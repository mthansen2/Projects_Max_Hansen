from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..labels import CLASS_TO_IDX
from ..utils import find_train_dir


class SplitListDataset(Dataset):
    """
    split file lines:
      relative/path.jpg <label_string>
    relative path is relative to train_dir (the folder that contains class dirs).
    """
    def __init__(self, data_root: Path, split_file: Path, tfm):
        self.train_dir = find_train_dir(data_root)
        self.items: List[Tuple[Path, int]] = []
        self.tfm = tfm

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, label = line.rsplit(" ", 1)
                self.items.append((Path(rel), CLASS_TO_IDX[label]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel, y = self.items[idx]
        img_path = self.train_dir / rel
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)
        return x, y


def build_transforms(img_size: int):
    # Don't use horizontal flip (can change handedness/meaning).
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfm, val_tfm


def make_loaders(data_root: Path, split_dir: Path, img_size: int, batch_size: int, num_workers: int):
    train_tfm, val_tfm = build_transforms(img_size)
    train_ds = SplitListDataset(data_root, split_dir / "train.txt", train_tfm)
    val_ds = SplitListDataset(data_root, split_dir / "val.txt", val_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
