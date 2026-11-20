"""Fine-tune DistilBERT to classify reviews as negative or positive.

What happens here, in plain words:
  1. Download a pretrained DistilBERT. It already "understands" English from
     reading lots of text, but knows nothing about sentiment yet.
  2. Put a small, randomly initialised classification layer on top (2 outputs).
  3. Show it ~60k labelled Amazon reviews for 2 passes (epochs) and nudge all the
     weights a little after every batch so the correct label gets more likely.
  4. After each epoch, measure accuracy / macro-F1 on a validation set it never
     trained on, and keep the best checkpoint in models/distilbert-sentiment/.

Usage:  python train.py                              (defaults: 60k reviews, 2 epochs, batch 16)
        python train.py --n-train 10000 --epochs 1   (quick try, a few minutes)
"""

from __future__ import annotations

import argparse
import json
import random
import time

from sentiment import BASE_MODEL, LABELS, MODEL_DIR, RESULTS_DIR  # isort: skip  (sets HF_HOME first)

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from sentiment.data import load_amazon
from sentiment.gpu import release_gpu_memory, wait_for_gpu


def set_seed(seed: int) -> None:
    """Make runs repeatable (same data order, same initial weights)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tokenizer, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    """Tokenize once up front; the collator pads each batch only to its longest review."""
    enc = tokenizer(list(df["text"]), truncation=True, max_length=max_length)
    rows = [
        {"input_ids": ids, "attention_mask": mask, "labels": int(label)}
        for ids, mask, label in zip(enc["input_ids"], enc["attention_mask"], df["label"])
    ]
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer))


@torch.inference_mode()
def evaluate(model, loader, device: str) -> dict[str, float]:
    """Accuracy and macro-F1 on a validation loader."""
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        preds += logits.argmax(-1).tolist()
        golds += batch["labels"].tolist()
    return {"accuracy": accuracy_score(golds, preds), "macro_f1": f1_score(golds, preds, average="macro")}


def train(args, splits, tokenizer, device: str, batch_size: int, accum_steps: int) -> list[dict]:
    """One full training run. Raises torch.cuda.OutOfMemoryError if the GPU fills up."""
    set_seed(args.seed)
    train_loader = make_loader(splits["train"], tokenizer, args.max_length, batch_size, shuffle=True)
    val_loader = make_loader(splits["val"], tokenizer, args.max_length, batch_size, shuffle=False)

    # Pretrained encoder + a fresh 2-class head. The label names are stored in the
    # saved config, so the app and pipelines show "positive" instead of "LABEL_1".
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)

    # AdamW + linear warm-up/decay of the learning rate: the standard fine-tuning recipe.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    # GradScaler keeps tiny fp16 gradients from rounding to zero.
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                # Passing `labels` makes the model compute the cross-entropy loss for us.
                loss = model(**batch).loss / accum_steps
            scaler.scale(loss).backward()
            # Gradient accumulation: with a smaller batch (after an OOM) we add up the
            # gradients of several batches before one optimizer step, so the maths matches.
            if step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * accum_steps
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}", flush=True)

        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, train_loss=running / len(train_loader), minutes=round((time.time() - start) / 60, 2))
        history.append(metrics)
        print(
            f"Epoch {epoch}: val accuracy {metrics['accuracy']:.4f}  val macro-F1 {metrics['macro_f1']:.4f}  "
            f"({metrics['minutes']} min)",
            flush=True,
        )

        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  saved best model to {MODEL_DIR}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--n-train", type=int, default=60_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    splits = load_amazon(n_train=args.n_train, seed=args.seed)
    print({name: len(df) for name, df in splits.items()})
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # The GPU is shared with other jobs: wait for room first, and on "CUDA out of
    # memory" retry with half the batch (and twice the accumulation).
    batch_size, accum_steps = args.batch_size, 1
    start = time.time()
    while True:
        if device == "cuda":
            wait_for_gpu(min_free_mb=2500)
        try:
            history = train(args, splits, tokenizer, device, batch_size, accum_steps)
            break
        except torch.cuda.OutOfMemoryError:
            release_gpu_memory()
            if batch_size == 1:
                raise
            batch_size, accum_steps = batch_size // 2, accum_steps * 2
            print(f"CUDA out of memory - waiting 60 s, then retrying with batch {batch_size} x {accum_steps} accumulation.")
            time.sleep(60)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {
        "args": vars(args),
        "final_batch_size": batch_size,
        "grad_accum_steps": accum_steps,
        "total_minutes": round((time.time() - start) / 60, 1),
        "history": history,
    }
    (RESULTS_DIR / "train_history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done in {summary['total_minutes']} min.")


if __name__ == "__main__":
    main()
