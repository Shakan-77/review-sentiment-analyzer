"""Small helpers for sharing one laptop GPU with other programs.

The GPU (6 GB) may be used by other training jobs at the same time, so before
we start we wait until enough memory is free, and if we still run out of memory
("CUDA OOM") the training script retries with a smaller batch.
"""

from __future__ import annotations

import gc
import time

import torch


def free_gpu_memory_mb() -> float:
    """Free memory on the GPU right now, in MB (counts other processes too). 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / 1024**2


def wait_for_gpu(min_free_mb: float = 2500, timeout_min: float = 60, poll_s: float = 30) -> bool:
    """Block until at least `min_free_mb` MB are free. Returns False on timeout (caller decides)."""
    if not torch.cuda.is_available():
        return False
    deadline = time.time() + timeout_min * 60
    while True:
        free = free_gpu_memory_mb()
        if free >= min_free_mb:
            print(f"GPU has {free:.0f} MB free - starting.")
            return True
        if time.time() > deadline:
            print(f"GPU still only has {free:.0f} MB free after {timeout_min} min.")
            return False
        print(f"GPU has {free:.0f} MB free, waiting for {min_free_mb:.0f} MB (another job is running)...")
        time.sleep(poll_s)


def release_gpu_memory() -> None:
    """Give cached GPU memory back after an OOM so a retry starts clean."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_device(min_free_mb: float = 1000) -> str:
    """Use the GPU for inference if it has room, otherwise fall back to the CPU."""
    return "cuda" if torch.cuda.is_available() and free_gpu_memory_mb() >= min_free_mb else "cpu"
