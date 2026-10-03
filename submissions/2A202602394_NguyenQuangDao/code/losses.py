"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).
Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

from typing import Union, Sequence
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    BaseModule = nn.Module
except ImportError:
    class BaseModule:
        pass
    torch = None
    nn = None
    F = None


class LabelSmoothingCE(BaseModule):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1, weight=None):
        if BaseModule is not object:
            super().__init__()
        self.smoothing = float(smoothing)
        self.weight = weight
        if nn is not None:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.smoothing, weight=self.weight)

    def forward(self, logits, target):
        return self.criterion(logits, target)

    def __call__(self, logits, target):
        return self.forward(logits, target)


class FocalLoss(BaseModule):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).

    Khi gamma = 0 và alpha = None: tương đương chính xác Cross-Entropy thông thường.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        if BaseModule is not object:
            super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        target_expanded = target.unsqueeze(-1)
        log_pt = log_p.gather(-1, target_expanded).squeeze(-1)
        pt = p.gather(-1, target_expanded).squeeze(-1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                alpha_tensor = torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            else:
                alpha_tensor = self.alpha.to(device=logits.device, dtype=logits.dtype)
            at = alpha_tensor.gather(-1, target)
            loss = at * loss

        return loss.mean()

    def __call__(self, logits, target):
        return self.forward(logits, target)


def class_weights(counts: Sequence[int] | np.ndarray, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng: w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp.
    """
    counts = np.asarray(counts, dtype=np.float64)
    num_classes = len(counts)

    if beta <= 0.0:
        inv_counts = 1.0 / np.maximum(counts, 1.0)
        weights = inv_counts / inv_counts.mean()
    else:
        # Effective number of samples: En = (1 - beta^n) / (1 - beta)
        # Weight w = 1 / En = (1 - beta) / (1 - beta^n)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        # Chuẩn hoá tổng trọng số về đúng số lớp
        weights = weights * (num_classes / weights.sum())

    if torch is not None:
        return torch.tensor(weights, dtype=torch.float32)
    return weights


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.
    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, điều chỉnh lam theo diện tích thực.
    """
    if torch is None:
        raise ImportError("Cần cài đặt PyTorch")

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        _, _, H, W = x.size()
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Toạ độ tâm ngẫu nhiên
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lại lam theo diện tích thực của bounding box
        actual_lam = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1)) / float(W * H)
        return x_mixed, (y_a, y_b, actual_lam)

    else:
        raise ValueError(f"Không hỗ trợ mode '{mode}'. Chọn 'mixup' hoặc 'cutmix'")


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: 'ce', 'ls', 'focal', 'ce_weighted'."""
    if nn is None:
        raise ImportError("Cần cài đặt PyTorch")

    weight = kw.get("weight", None)

    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    elif kind == "ls":
        smoothing = kw.get("label_smoothing", kw.get("smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing, weight=weight)
    elif kind == "focal":
        gamma = kw.get("focal_gamma", kw.get("gamma", 2.0))
        alpha = kw.get("alpha", weight)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ hàm loss: '{kind}'. Chọn trong ['ce', 'ls', 'focal', 'ce_weighted']")
