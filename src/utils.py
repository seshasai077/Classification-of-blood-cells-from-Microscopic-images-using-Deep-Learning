from __future__ import annotations

import json, torch, random, os, time
from pathlib import Path
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now_run_dir():
    p = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    p.mkdir(parents=True)
    return p


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def device_from_arg(device):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_labels(run_dir):
    return load_json(Path(run_dir) / "labels.json")


def accuracy_top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


class AverageMeter:
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        self.total += val * n
        self.count += n

    @property
    def avg(self):
        return self.total / self.count