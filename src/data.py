from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np

@dataclass
class DataConfig:
    data_dir: Path
    img_size: int = 224  # ResNet18 expects 224x224
    batch_size: int = 32
    num_workers: int = 2
    val_split: float = 0.2  # validation split

def build_transforms(img_size=224):
    # Training transforms with stronger augmentation
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Validation / evaluation transforms (no augmentation)
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_tfms, eval_tfms

def create_dataloaders(cfg: DataConfig):
    train_tfm, eval_tfm = build_transforms(cfg.img_size)

    train_dir = cfg.data_dir / "train"
    val_dir = cfg.data_dir / "val"

    if val_dir.exists():
        train_ds = datasets.ImageFolder(train_dir, train_tfm)
        val_ds = datasets.ImageFolder(val_dir, eval_tfm)
        ref_ds = train_ds
    else:
        full = datasets.ImageFolder(train_dir, train_tfm)
        n = len(full)
        n_val = int(n * cfg.val_split)
        n_train = n - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val])
        ref_ds = full

    # -----------------------------
    # Weighted sampler to balance classes
    # -----------------------------
    targets = [y for _, y in train_ds]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    return train_loader, val_loader, ref_ds