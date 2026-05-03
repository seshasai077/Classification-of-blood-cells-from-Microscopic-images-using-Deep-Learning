from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.data import DataConfig, create_dataloaders
from src.model import build_model
from src.utils import save_json, now_run_dir, device_from_arg


def main():
    # -----------------------------
    # Argument parsing
    # -----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to dataset folder")
    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    args = ap.parse_args()

    device = device_from_arg("auto")

    # -----------------------------
    # Data loaders
    # -----------------------------
    cfg = DataConfig(data_dir=Path(args.data_dir))
    train_loader, val_loader, ref_ds = create_dataloaders(cfg)
    labels = list(ref_ds.classes)

    # -----------------------------
    # Model with Dropout
    # -----------------------------
    model = build_model(len(labels)).to(device)
    
    # -----------------------------
    # Optimizer & Scheduler
    # -----------------------------
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)  # L2 regularization
    scheduler = StepLR(opt, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # Save labels and run directory
    # -----------------------------
    run_dir = now_run_dir()
    save_json(run_dir / "labels.json", labels)

    # -----------------------------
    # Early stopping setup
    # -----------------------------
    best_val_acc = 0.0
    patience = 5
    counter = 0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        # -----------------------------
        # Save best model
        # -----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_dir / "model_best.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

        # Step the LR scheduler
        scheduler.step()

    print("Training done!")

if __name__ == "__main__":
    main()