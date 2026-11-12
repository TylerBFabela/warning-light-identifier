"""Train the warning-light classifier on the synthetic images.

Usage:  python src/train.py data --arch small_cnn --epochs 15 --out models/v1
Saves to the --out folder: model.pt (weights), classes.json, config.json, metrics.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import MEAN, STD, build_model, eval_transform, get_device


def train_transform(img):
    """Training images get light random changes each time they are used, so the model cannot memorize them."""
    return transforms.Compose([
        transforms.Resize((img, img)),
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),  # hue=0 on purpose: color means severity
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])                                                                          # no horizontal flip: left/right arrows differ


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = correct = count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        scores = model(xb)
        total_loss += loss_fn(scores, yb).item() * len(yb)
        correct += (scores.argmax(1) == yb).sum().item()
        count += len(yb)
    return total_loss / count, correct / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--arch", default="small_cnn")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/v1")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)    # repeatable runs
    device = get_device()

    # 1) Data: one folder per class. Hold out 10% of the images to check the model on pictures it never trained on.
    classes = datasets.ImageFolder(a.data_dir).classes
    order = np.random.RandomState(a.seed).permutation(len(datasets.ImageFolder(a.data_dir)))
    n_val = max(1, len(order) // 10)
    train_ds = Subset(datasets.ImageFolder(a.data_dir, transform=train_transform(a.img)), order[n_val:])
    val_ds = Subset(datasets.ImageFolder(a.data_dir, transform=eval_transform(a.img)), order[:n_val])
    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=a.batch, num_workers=0)

    # 2) Model, loss, optimizer, learning-rate schedule.
    model = build_model(a.arch, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=a.lr, total_steps=a.epochs * len(train_loader))
    print(f"device={device} arch={a.arch} params={n_params:,} train={len(train_ds)} val={len(val_ds)} classes={len(classes)}")

    # 3) Train. Keep the weights from the epoch with the lowest validation loss.
    best_loss, best_state, best_epoch, epochs_without_gain = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        start = time.time()
        run_loss = seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            schedule.step()
            run_loss += loss.item() * len(yb)
            seen += len(yb)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": run_loss / seen, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d}/{a.epochs}  {time.time() - start:5.1f}s  train_loss={run_loss / seen:.3f}  val_loss={val_loss:.3f}  val_acc={val_acc:.3f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, epochs_without_gain = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= a.patience:
                print("early stopping: validation loss stopped improving")
                break

    # 4) Save everything needed to use the model later.
    model.load_state_dict(best_state)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "classes.json").write_text(json.dumps(classes))
    (out / "config.json").write_text(json.dumps({"arch": a.arch, "img": a.img, "mean": MEAN, "std": STD, "best_epoch": best_epoch,
                                                 "seed": a.seed, "lr": a.lr, "batch": a.batch, "params": n_params, "threshold": 0.8}, indent=2))
    (out / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"saved the best epoch ({best_epoch}) to {out}/")


if __name__ == "__main__":
    main()