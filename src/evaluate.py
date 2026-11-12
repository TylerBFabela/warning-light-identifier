"""Test a trained model on REAL cropped icons.

Usage:  python src/evaluate.py models/v1 real_test
real_test/ needs one folder per class, named exactly like the model's classes (e.g. real_test/battery/carA_01.jpg).
Prints accuracy, per-class precision/recall, a confusion matrix, and what happens at different "unsure" thresholds.
Never train on these images or use them to choose settings: they are your honest final test.
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from common import eval_transform, get_device, load_model

model_dir, test_dir = sys.argv[1], sys.argv[2]
device = get_device()
model, classes, cfg = load_model(model_dir, device)

ds = datasets.ImageFolder(test_dir, transform=eval_transform(cfg["img"]))
to_model_index = {i: classes.index(name) for i, name in enumerate(ds.classes)}   # folder order -> model class order

all_probs, truth = [], []
with torch.no_grad():
    for xb, yb in DataLoader(ds, batch_size=64):
        all_probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
        truth += [to_model_index[int(i)] for i in yb]
probs, truth = np.concatenate(all_probs), np.array(truth)
guess = probs.argmax(axis=1)

n = len(classes)
confusion = np.zeros((n, n), dtype=int)                                          # rows = true class, columns = guess
for t, g in zip(truth, guess):
    confusion[t, g] += 1

print(f"images: {len(truth)}   accuracy: {(guess == truth).mean():.3f}")
print(f"{'class':15s} {'precision':>9s} {'recall':>7s} {'images':>7s}")
for i, name in enumerate(classes):
    hit, false_alarm, miss = confusion[i, i], confusion[:, i].sum() - confusion[i, i], confusion[i].sum() - confusion[i, i]
    if confusion[i].sum() == 0 and confusion[:, i].sum() == 0:
        continue                                                                 # class not present in this test set
    print(f"{name:15s} {hit / max(hit + false_alarm, 1):9.2f} {hit / max(hit + miss, 1):7.2f} {confusion[i].sum():7d}")

print("\nmost common mistakes (true -> guessed):")
mistakes = [(confusion[t, g], classes[t], classes[g]) for t in range(n) for g in range(n) if t != g and confusion[t, g] > 0]
for count, t, g in sorted(mistakes, reverse=True)[:8]:
    print(f"  {count:3d} x  {t} -> {g}")
if not mistakes:
    print("  none")

none_index = classes.index("none")
print("\nthreshold  kept  accuracy_of_kept   (below the threshold the answer becomes 'unsure', counted as none)")
confidence = probs.max(axis=1)
for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
    kept = confidence >= threshold
    acc_kept = (guess[kept] == truth[kept]).mean() if kept.any() else float("nan")
    print(f"{threshold:9.2f} {kept.mean():5.2f} {acc_kept:17.3f}")