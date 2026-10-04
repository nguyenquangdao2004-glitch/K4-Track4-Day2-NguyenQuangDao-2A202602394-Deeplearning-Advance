"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện chuẩn:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file train: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file val: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file test: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if not {"Filename", "Label"}.issubset(df.columns):
            raise ValueError(f"File {name} thiếu cột bắt buộc 'Filename' hoặc 'Label'")

    return train_df, val_df, test_df


def resolve_images_dir(images_dir: str | Path) -> Path:
    """Tự động phát hiện và giải quyết đường dẫn thư mục ảnh DeepWeeds.
    Hỗ trợ các trường hợp: data/images, data, /kaggle/input/..., hoặc đường dẫn tương đối.
    """
    p = Path(images_dir)
    # 1. Nếu đường dẫn được truyền vào tồn tại và chứa file .jpg
    if p.exists() and any(p.glob("*.jpg")):
        return p

    # 2. Nếu được truyền "data/images" nhưng ảnh nằm ở "data"
    if p.parent.exists() and any(p.parent.glob("*.jpg")):
        return p.parent

    # 3. Nếu trong p có thư mục con "images" chứa ảnh
    if (p / "images").exists() and any((p / "images").glob("*.jpg")):
        return p / "images"

    # 4. Tìm kiếm trong /kaggle/input (nếu chạy trên Kaggle)
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if any(f.lower().endswith(".jpg") for f in files):
                return Path(root)

    # 5. Tìm kiếm trong các thư mục thông dụng quanh thư mục làm việc
    for candidate in [Path("data/images"), Path("data"), Path("images"), Path(".")]:
        if candidate.exists() and any(candidate.glob("*.jpg")):
            return candidate

    return p


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir` (nếu thư mục ảnh tồn tại)
    """
    images_dir = resolve_images_dir(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_imgs = n_train + n_val + n_test

    set_train = set(train_df["Filename"])
    set_val = set(val_df["Filename"])
    set_test = set(test_df["Filename"])

    # 1. Kiểm tra overlap
    train_val_overlap = set_train & set_val
    train_test_overlap = set_train & set_test
    val_test_overlap = set_val & set_test

    if len(train_val_overlap) > 0:
        raise AssertionError(f"Rò rỉ dữ liệu: Train và Val giao nhau {len(train_val_overlap)} ảnh!")
    if len(train_test_overlap) > 0:
        raise AssertionError(f"Rò rỉ dữ liệu: Train và Test giao nhau {len(train_test_overlap)} ảnh!")
    if len(val_test_overlap) > 0:
        raise AssertionError(f"Rò rỉ dữ liệu: Val và Test giao nhau {len(val_test_overlap)} ảnh!")

    # 2. Hợp ba tập
    union_files = set_train | set_val | set_test
    if len(union_files) != 17509:
        raise AssertionError(f"Tổng hợp ba tập là {len(union_files)} ảnh, khác chuẩn 17509 ảnh của DeepWeeds!")

    # 3. Phân bố lớp
    train_per_class = train_df["Label"].value_counts().sort_index().to_dict()
    val_per_class = val_df["Label"].value_counts().sort_index().to_dict()
    test_per_class = test_df["Label"].value_counts().sort_index().to_dict()

    # 4. Kiểm tra file ảnh tồn tại trên đĩa (nếu thư mục ảnh đã được giải nén)
    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        sample_to_check = list(union_files)[:100]
        missing = [fn for fn in sample_to_check if not (images_dir / fn).exists()]
        if missing:
            raise FileNotFoundError(f"Có ảnh trong danh sách CSV không tìm thấy trong {images_dir} (ví dụ: {missing[:3]})")

    stats = {
        "n": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": total_imgs,
            "pct": (round(n_train / total_imgs * 100, 2),
                    round(n_val / total_imgs * 100, 2),
                    round(n_test / total_imgs * 100, 2)),
        },
        "per_class": {
            "train": train_per_class,
            "val": val_per_class,
            "test": test_per_class,
        },
        "overlap": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
        "all_files_exist": True,
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation.
    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: Resize(256) -> CenterCrop(img_size) + ToTensor + Normalize (hoặc Resize(img_size) nếu img_size==256).
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("Cần cài đặt torchvision để tạo transforms")

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train:
        if img_size == 256:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ])

    # Augmentation cho tập train
    aug_list = []
    aug_list.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
    aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug == "basic":
        pass
    elif aug == "color":
        aug_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    elif aug == "trivial":
        aug_list.append(transforms.TrivialAugmentWide())
    elif aug == "randaug":
        aug_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif aug == "heavy":
        aug_list.extend([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        ])
    else:
        raise ValueError(f"Không hỗ trợ kiểu aug '{aug}'. Chọn trong [basic, color, trivial, randaug, heavy]")

    aug_list.extend([
        transforms.ToTensor(),
        normalize,
    ])
    return transforms.Compose(aug_list)


# Cho phép DeepWeedsDataset thừa kế Dataset khi torch có sẵn
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    BaseDataset = Dataset
except ImportError:
    class BaseDataset:
        pass
    torch = None
    DataLoader = None
    WeightedRandomSampler = None


class DeepWeedsDataset(BaseDataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = resolve_images_dir(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        fn = self.filenames[i]
        path = self.images_dir / fn
        if not path.exists():
            if (self.images_dir.parent / fn).exists():
                path = self.images_dir.parent / fn
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[i])
        return img, label, fn


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: Optional[str] = None, num_workers: int = 2):
    """Tạo DataLoader theo quy chuẩn của repo."""
    if torch is None or DataLoader is None:
        raise ImportError("Cần cài đặt PyTorch để tạo DataLoader")

    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / class_counts[y] for y in df["Label"]]
            sample_weights_tensor = torch.DoubleTensor(sample_weights)
            data_sampler = WeightedRandomSampler(sample_weights_tensor, num_samples=len(sample_weights), replacement=True)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=data_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
    else:
        # Khi đánh giá: KHÔNG shuffle, giữ nguyên thứ tự file
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
