"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def bench(fn: Callable, warmup: int = 10, iters: int = 100, sync: Optional[Callable] = None) -> Dict[str, Any]:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo lặp lại có đồng bộ
    times_ms = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "p99": float(np.percentile(times_arr, 99)),
        "mean": float(np.mean(times_arr)),
        "std": float(np.std(times_arr, ddof=1)),
        "n": iters,
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}
    """
    if torch is None:
        raise ImportError("Cần cài đặt PyTorch để benchmark")

    target_device = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
    model = model.to(target_device)
    model.eval()

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=target_device)

    sync_fn = torch.cuda.synchronize if target_device.type == "cuda" else None
    gpu_name = torch.cuda.get_device_name(0) if target_device.type == "cuda" else "CPU"

    if dtype == "fp16":
        model = model.half()
        dummy = dummy.half()
        def forward_fn():
            with torch.inference_mode():
                _ = model(dummy)
    elif dtype == "amp":
        def forward_fn():
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(dummy)
    else:  # fp32
        def forward_fn():
            with torch.inference_mode():
                _ = model(dummy)

    stats = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    p50_sec = stats["p50"] / 1000.0
    throughput = (batch_size / p50_sec) if p50_sec > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "images_per_s": round(throughput, 2),
        "torch": torch.__version__,
    }


def tta_latency(model, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> Dict[str, Any]:
    """Độ trễ của TTA K view: chạy K lượt forward trên batch."""
    if torch is None:
        raise ImportError("Cần cài đặt PyTorch")

    target_device = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
    model = model.to(target_device)
    model.eval()

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=target_device)
    sync_fn = torch.cuda.synchronize if target_device.type == "cuda" else None

    def tta_forward():
        with torch.inference_mode():
            for _ in range(k_views):
                _ = model(dummy)

    stats = bench(tta_forward, warmup=warmup, iters=iters, sync=sync_fn)
    return stats
