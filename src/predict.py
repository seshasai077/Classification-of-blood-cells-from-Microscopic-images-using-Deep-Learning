from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.model import build_model
from src.utils import read_labels


def predict(run_dir, image_path):
    labels = read_labels(run_dir)

    model = build_model(len(labels))
    model.load_state_dict(torch.load(Path(run_dir) / "model_best.pt"))
    model.eval()

    img = Image.open(image_path).convert("RGB")

    tfm = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    x = tfm(img).unsqueeze(0)

    out = model(x)
    pred = out.argmax(1).item()

    print("Prediction:", labels[pred])