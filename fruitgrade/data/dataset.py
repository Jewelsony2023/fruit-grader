"""
PyTorch Dataset: returns (image tensor, mask tensor, grade index, severity 0-1).

Augmentations are applied to image AND mask together (torchvision v2 handles
this automatically when the mask is wrapped as tv_tensors.Mask).
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_labels(root, split=None, sources=None, mask_quality=None):
    rows = list(csv.DictReader(open(Path(root) / "labels.csv", encoding="utf-8")))
    if split:
        rows = [r for r in rows if r["split"] == split]
    if sources:
        rows = [r for r in rows if r["source"] in sources]
    if mask_quality:
        rows = [r for r in rows if r["mask_quality"] in mask_quality]
    return rows


def build_transforms(size, train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(30)], p=0.5),
            # colour jitter changes the IMAGE only (masks are untouched automatically)
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess(rgb: np.ndarray, size: int) -> torch.Tensor:
    """Single numpy RGB image -> normalised 1x3xHxW tensor (used at inference)."""
    img = tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
    return build_transforms(size, train=False)(img).unsqueeze(0)


class FruitDataset(Dataset):
    def __init__(self, root, split, grades, size=256, train=False, **filters):
        self.root = Path(root)
        self.rows = read_labels(root, split, **filters)
        if not self.rows:
            raise ValueError(f"No rows for split='{split}' in {self.root/'labels.csv'}")
        self.grades = list(grades)
        self.tf = build_transforms(size, train)

    def __len__(self):
        return len(self.rows)

    def grade_counts(self):
        c = np.zeros(len(self.grades))
        for r in self.rows:
            c[self.grades.index(r["grade"])] += 1
        return c

    def __getitem__(self, i):
        r = self.rows[i]
        bgr = cv2.imread(str(self.root / r["image"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.root / r["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        img = tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
        msk = tv_tensors.Mask(torch.from_numpy(mask).long())
        img, msk = self.tf(img, msk)
        # severity is recomputed from the (possibly cropped) mask so it stays consistent
        m = msk
        fruit = (m > 0).sum().item()
        sev = ((m >= 2).sum().item() / fruit) if fruit else 0.0
        return {
            "image": img.as_subclass(torch.Tensor),
            "mask": m.as_subclass(torch.Tensor),
            "grade": torch.tensor(self.grades.index(r["grade"])),
            "severity": torch.tensor(sev, dtype=torch.float32),   # 0..1
            "name": r["image"],
        }
