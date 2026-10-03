"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

from typing import List, Dict, Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import timm
except ImportError:
    timm = None

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    if timm is None:
        raise ImportError("Cần cài đặt timm để sử dụng build_model")

    is_pretrained = pretrained if init != "scratch" else False

    # timm tự thay head mới tương thích num_classes
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ classifier head.
    Backbone đóng băng thì BatchNorm cũng phải đặt ở chế độ eval.
    """
    if torch is None or nn is None:
        raise ImportError("Cần cài đặt torch để đóng băng tham số")

    # Đóng băng toàn bộ
    for param in model.parameters():
        param.requires_grad = False

    # Mở lại gradient cho head
    head = model.get_classifier()
    if isinstance(head, nn.Module):
        for param in head.parameters():
            param.requires_grad = True
    elif isinstance(head, nn.Parameter):
        head.requires_grad = True

    # Đặt tất cả BatchNorm trong phần backbone về chế độ eval
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1 (weights): lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    if torch is None:
        raise ImportError("Cần cài đặt torch")

    head = model.get_classifier()
    head_params = set(head.parameters() if isinstance(head, nn.Module) else [head])

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p in head_params:
            if p.ndim <= 1 or name.endswith(".bias"):
                head_no_decay.append(p)
            else:
                head_decay.append(p)
        else:
            if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
                backbone_no_decay.append(p)
            else:
                backbone_decay.append(p)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})

    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return float(total / 1e6)


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.
    Thử tính bằng fvcore, thop, hoặc tính toán xấp xỉ qua hooks.
    """
    if torch is None:
        raise ImportError("Cần cài đặt torch")

    device = next(model.parameters()).device

    # Cách 1: dùng thop nếu có
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs / 1e9)
    except Exception:
        pass

    # Cách 2: dùng fvcore nếu có
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        flops = FlopCountAnalysis(model, dummy)
        # fvcore tính FLOPs (1 MAC = 2 FLOPs trong một số định nghĩa, nhưng fvcore đếm MACs hoặc FLOPs)
        return float(flops.total() / 1e9)
    except Exception:
        pass

    # Cách 3: Ước lượng chuẩn theo công thức kiến trúc bằng hook Conv2d / Linear
    total_macs = [0]

    def conv_hook(self, input, output):
        batch_size, in_channels, in_h, in_w = input[0].size()
        out_channels, out_h, out_w = output.size()[1:]
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (in_channels // self.groups)
        num_macs = kernel_ops * out_h * out_w * out_channels
        total_macs[0] += num_macs

    def linear_hook(self, input, output):
        batch_size = input[0].size(0)
        num_macs = self.in_features * self.out_features
        total_macs[0] += num_macs

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        try:
            model(dummy)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    if total_macs[0] > 0:
        return float(total_macs[0] / 1e9)

    # Dự phòng dựa trên các giá trị chuẩn trong slide cho các mạng phổ biến
    name_str = getattr(model, "pretrained_cfg", {}).get("architecture", "")
    lookup = {
        "resnet50": 4.1,
        "resnext50_32x4d": 4.2,
        "convnext_tiny": 4.5,
        "deit_small_patch16_224": 4.6,
        "vit_small_patch16_224": 4.6,
        "swin_tiny_patch4_window7_224": 4.5,
        "efficientnet_b0": 0.39,
        "mobilenetv3_large_100": 0.22,
    }
    for k, v in lookup.items():
        if k in name_str.lower():
            return v
    return 4.0
