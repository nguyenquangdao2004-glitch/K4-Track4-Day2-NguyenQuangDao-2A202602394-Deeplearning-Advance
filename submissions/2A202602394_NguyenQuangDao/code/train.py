"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình: đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng `eval.compute_metrics` của repo gốc.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Đảm bảo import được eval.py từ repo gốc
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from eval import compute_metrics, save_predictions
except ImportError:
    # Dự phòng nếu đường dẫn tương đối khác
    try:
        import eval as ev
        compute_metrics = ev.compute_metrics
        save_predictions = ev.save_predictions
    except Exception:
        compute_metrics = None
        save_predictions = None

try:
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import LambdaLR
except ImportError:
    torch = None
    nn = None
    LambdaLR = None

import dataset as dataset_lib
import model as model_lib
import losses as losses_lib


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | heavy
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: Optional[EMA] = None) -> Dict[str, float]:
    """Một epoch huấn luyện."""
    model.train()
    # Nếu backbone bị đóng băng, giữ các lớp BatchNorm của backbone ở chế độ eval
    if cfg.init == "frozen":
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    total_loss = 0.0
    num_batches = len(loader)

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Áp dụng Mixup hoặc CutMix nếu cấu hình
        if cfg.mix:
            mixed_images, targets = losses_lib.mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)
            if cfg.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(mixed_images)
                    loss = losses_lib.mixed_loss(criterion, outputs, targets)
            else:
                outputs = model(mixed_images)
                loss = losses_lib.mixed_loss(criterion, outputs, targets)
        else:
            if cfg.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

        if cfg.amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += float(loss.item())

    avg_loss = total_loss / max(1, num_batches)
    current_lr = float(optimizer.param_groups[0]["lr"])
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []
    total_loss = 0.0

    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += float(loss.item()) * len(labels)
            all_filenames.extend(filenames)
            all_y_true.append(labels.cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, len(y_true))
    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: List[Dict[str, Any]], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h.get("val_macro_f1", 0.0) for h in history]
    val_acc = [h.get("val_top1", 0.0) for h in history]
    lrs = [h.get("lr", 0.0) for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Loss Curve
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="royalblue")
    ax1.plot(epochs, val_loss, "s--", label="Val Loss", color="crimson")
    ax1.set_title("Loss theo Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Metrics (Macro-F1 & Top-1 Acc)
    ax2.plot(epochs, val_f1, "^-", label="Val Macro-F1", color="darkgreen")
    ax2.plot(epochs, val_acc, "d-", label="Val Top-1 Acc", color="orange")
    ax2.set_title("Chỉ số Val (Macro-F1 & Top-1)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Điểm (0-1)")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # 3. Learning Rate Schedule
    ax3.plot(epochs, lrs, ".-", label="Learning Rate", color="purple")
    ax3.set_title("Lịch trình Learning Rate")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("LR")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _to_probs(logits: np.ndarray) -> np.ndarray:
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run(cfg: Config) -> Dict[str, Any]:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    if torch is None:
        raise ImportError("Cần cài đặt PyTorch")

    # 1. Cố định seed & tạo thư mục lưu kết quả
    set_seed(cfg.seed)
    save_run_dir = run_dir(cfg)
    save_run_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)

    with open(save_run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Nạp dữ liệu và kiểm tra tính toàn vẹn (S1-S6)
    train_df, val_df, test_df = dataset_lib.load_split(cfg.labels_dir, fold=cfg.fold)
    split_stats = dataset_lib.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Dựng transform và data loader
    train_transform = dataset_lib.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_transform = dataset_lib.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset_lib.make_loader(
        train_df, cfg.images_dir, train_transform,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset_lib.make_loader(
        val_df, cfg.images_dir, val_transform,
        batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers
    )

    # 4. Dựng model, criterion, optimizer, scheduler, EMA
    model = model_lib.build_model(
        name=cfg.backbone,
        num_classes=dataset_lib.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    ).to(device)

    # Thiết lập loss function
    loss_kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        class_counts = train_df["Label"].value_counts().sort_index().to_numpy()
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        w = losses_lib.class_weights(class_counts, beta=beta)
        loss_kw["weight"] = w.to(device)

    criterion = losses_lib.build_criterion(kind=cfg.loss, **loss_kw)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    if cfg.amp and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    else:
        scaler = None

    # 5. Vòng lặp huấn luyện từng epoch
    history = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_model_path = save_run_dir / "best_model.pt"

    start_train_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=ema
        )
        epoch_train_time = time.time() - t0

        # Nếu có EMA, dùng trọng số EMA để đánh giá
        if ema is not None:
            ema.apply_shadow(model)

        val_fns, val_y, val_logits, val_loss = evaluate(model, val_loader, criterion, device)

        if ema is not None:
            ema.restore(model)

        val_probs = _to_probs(val_logits)
        val_pred = val_probs.argmax(axis=1)

        # Tính macro-F1 val và top-1 acc
        if compute_metrics is not None:
            met = compute_metrics(val_y, val_pred, val_probs)
            val_f1 = met["macro_f1"]
            val_acc = met["top1"]
        else:
            val_acc = float((val_pred == val_y).mean())
            from sklearn.metrics import f1_score
            val_f1 = float(f1_score(val_y, val_pred, average="macro"))

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_top1": val_acc,
            "lr": train_metrics["lr"],
            "epoch_time_s": epoch_train_time,
        }
        history.append(epoch_record)

        # Lưu checkpoint theo macro-F1 val tốt nhất (nếu hòa thì giữ epoch sớm hơn)
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            # Lưu state_dict (nếu có EMA thì lưu trọng số shadow)
            if ema is not None:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), best_model_path)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), best_model_path)

    total_train_time = time.time() - start_train_time

    # 6. Nạp lại checkpoint tốt nhất và xuất file dự đoán val
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_fns, val_y, best_val_logits, _ = evaluate(model, val_loader, criterion, device)
    best_val_probs = _to_probs(best_val_logits)
    np.save(save_run_dir / "val_logits.npy", best_val_logits)

    if save_predictions is not None:
        save_predictions(pred_path(cfg, "val"), val_fns, val_y, best_val_probs)

    # 7. NẾU cfg.save_test_predictions (Chung kết - Bước 4): Đánh giá test ĐÚNG MỘT LẦN
    test_metrics = {}
    if cfg.save_test_predictions:
        test_transform = dataset_lib.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset_lib.make_loader(
            test_df, cfg.images_dir, test_transform,
            batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers
        )
        test_fns, test_y, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = _to_probs(test_logits)
        np.save(save_run_dir / "test_logits.npy", test_logits)

        if save_predictions is not None:
            save_predictions(pred_path(cfg, "test"), test_fns, test_y, test_probs)

        if compute_metrics is not None:
            test_metrics = compute_metrics(test_y, test_probs.argmax(axis=1), test_probs)

    # 8. Lưu history.csv và vẽ biểu đồ curves/
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_run_dir / "history.csv", index=False)

    curve_path = Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"Exp: {cfg.exp_id} - {cfg.backbone} (Seed {cfg.seed})")

    params_m = model_lib.count_params(model)
    gmacs = model_lib.count_gmacs(model, cfg.img_size)

    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_top1": history[best_epoch - 1]["val_top1"] if best_epoch > 0 else 0.0,
        "params_m": params_m,
        "gmacs": gmacs,
        "avg_epoch_time_s": total_train_time / cfg.epochs,
        "test_metrics": test_metrics,
    }
    return summary


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    field_types = {f.name: f.type for f in dataclasses.fields(Config)}
    overrides = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không đúng định dạng key=value: '{pair}'")
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k not in field_types:
            raise KeyError(f"Trường '{k}' không tồn tại trong Config. Các trường hợp lệ: {list(field_types.keys())}")

        # Ép kiểu giá trị
        if v.lower() in ("none", "null"):
            overrides[k] = None
        elif v.lower() == "true":
            overrides[k] = True
        elif v.lower() == "false":
            overrides[k] = False
        else:
            try:
                # Thử parse số hoặc literal
                overrides[k] = ast.literal_eval(v)
            except Exception:
                overrides[k] = v

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0."""
    parser = argparse.ArgumentParser(description="Chạy huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="+", default=[], help="Cặp KEY=VALUE để ghi đè Config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"=== Bắt đầu huấn luyện Exp: {cfg.exp_id} ({cfg.backbone}) - Seed: {cfg.seed} ===")
    summary = run(cfg)
    print(f"=== Hoàn thành Exp: {cfg.exp_id} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
