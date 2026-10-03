"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).
Giao diện giữ nguyên:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
from typing import List, Tuple, Callable, Optional, Sequence
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def predict_logits(model, loader, device, view: Optional[Callable] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    if torch is None:
        raise ImportError("Cần cài đặt PyTorch")

    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            images, labels, filenames = batch
            images = images.to(device)

            if view is not None:
                images = view(images)

            # Hỗ trợ mixed precision khi suy luận nếu trên CUDA
            if device.type == "cuda" or (isinstance(device, str) and "cuda" in device):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)

            all_filenames.extend(filenames)
            all_y_true.append(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels))
            all_logits.append(logits.float().cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits


def view_identity(x):
    """Không biến đổi ảnh (1-view mốc)."""
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop` x `crop`."""
    _, _, H, W = x.size()
    crops = [
        x[:, :, :crop, :crop],                 # Trên - Trái
        x[:, :, :crop, W - crop:],             # Trên - Phải
        x[:, :, H - crop:, :crop],             # Dưới - Trái
        x[:, :, H - crop:, W - crop:],         # Dưới - Phải
        x[:, :, (H - crop) // 2:(H + crop) // 2, (W - crop) // 2:(W + crop) // 2], # Giữa
    ]
    return crops


def views_multiscale(x, sizes: Sequence[int]):
    """Resize batch về từng kích thước trong `sizes`."""
    scaled = []
    for s in sizes:
        res = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        scaled.append(res)
    return scaled


def _softmax(z: np.ndarray) -> np.ndarray:
    shift = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    if not logits_per_view:
        raise ValueError("Danh sách logits_per_view rỗng")

    if space == "prob":
        probs_list = [_softmax(log) for log in logits_per_view]
        return np.mean(probs_list, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return _softmax(avg_logits)
    else:
        raise ValueError(f"Không hỗ trợ space '{space}'. Chọn 'prob' hoặc 'logit'")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    if not list_of_probs:
        raise ValueError("Danh sách list_of_probs rỗng")
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69).
    Khớp nhiệt độ bằng scipy.optimize.
    """
    from scipy.optimize import minimize_scalar

    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    def nll_obj(log_t):
        t = np.exp(log_t)
        scaled_logits = val_logits / t
        # log_softmax ổn định số học
        shift = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shift), axis=-1, keepdims=True))
        log_probs = shift - log_sum_exp
        # NLL loss
        nll = -np.mean(log_probs[np.arange(len(val_labels)), val_labels])
        return nll

    res = minimize_scalar(nll_obj, bounds=(-3.0, 3.0), method="bounded")
    best_t = float(np.exp(res.x))
    return best_t


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    scaled = np.asarray(logits, dtype=np.float64) / max(float(T), 1e-6)
    return _softmax(scaled)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).
    Chỉ áp dụng cho mô hình có Conv2d và BatchNorm2d liền kề.
    """
    if torch is None:
        raise ImportError("Cần cài đặt PyTorch")

    fused_model = copy.deepcopy(model)
    fused_model.eval()

    # Sử dụng tiện ích tích hợp sẵn của PyTorch nếu có
    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
    except ImportError:
        fuse_conv_bn_eval = None

    def _fuse_modules(module):
        prev_name = None
        prev_child = None
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d) and isinstance(prev_child, nn.Conv2d):
                if fuse_conv_bn_eval is not None:
                    fused_conv = fuse_conv_bn_eval(prev_child, child)
                else:
                    # Tự tính theo công thức slide trang 71
                    w = prev_child.weight
                    b = prev_child.bias if prev_child.bias is not None else torch.zeros(w.size(0), device=w.device)
                    gamma = child.weight
                    beta = child.bias
                    mean = child.running_mean
                    var = child.running_var
                    eps = child.eps

                    scale = gamma / torch.sqrt(var + eps)
                    w_prime = w * scale.view(-1, 1, 1, 1)
                    b_prime = beta + scale * (b - mean)

                    fused_conv = nn.Conv2d(
                        in_channels=prev_child.in_channels,
                        out_channels=prev_child.out_channels,
                        kernel_size=prev_child.kernel_size,
                        stride=prev_child.stride,
                        padding=prev_child.padding,
                        dilation=prev_child.dilation,
                        groups=prev_child.groups,
                        bias=True,
                    )
                    fused_conv.weight.data.copy_(w_prime)
                    fused_conv.bias.data.copy_(b_prime)

                setattr(module, prev_name, fused_conv)
                setattr(module, name, nn.Identity())
            else:
                _fuse_modules(child)
            prev_name = name
            prev_child = child

    _fuse_modules(fused_model)
    return fused_model
