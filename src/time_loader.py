"""Time how long it takes to feed images to a model, so you know what is slow.

Usage:  python time_loader.py data
        (data/ must contain one subfolder per class, e.g. data/battery/*.jpg -- make_synth.py creates this)
"""
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

tfm = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
ds = datasets.ImageFolder(data_dir, transform=tfm)  # one subfolder per class
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)  # <- this is the "loader"
print("images:", len(ds), "| classes:", ds.classes)

# 1) Only load and prepare the images (no model). If this is most of the time, data loading is the bottleneck.
t = time.time()
for xb, yb in loader:
    pass
data_s = time.time() - t
print(f"data only: {data_s:.1f}s  ({len(ds) / data_s:.0f} images/sec)")

# 2) One full training epoch with a tiny CNN.
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(64, len(ds.classes)),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

t = time.time()
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    loss_fn(model(xb), yb).backward()
    opt.step()
if device == "cuda":
    torch.cuda.synchronize()
elif device == "mps":
    torch.mps.synchronize()
epoch_s = time.time() - t
print(f"one training epoch: {epoch_s:.1f}s  ({len(ds) / epoch_s:.0f} images/sec)")