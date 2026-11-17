"""Does the model know more than it says?  Two questions about REAL lights that the model calls 'none'.

Usage:  python src/probe_no_none.py models/v3 external_rf2
Question 1: if we ignore the 'none' score and take the best of the other 12 classes, how often is that right?
Question 2: if we push the 'none' score down by a penalty, how many real lights do we win,
            and how many true 'none' crops start being called lights (false alarms)?
Test-only: nothing is trained and no weights change.
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
none_index = classes.index("none")

ds = datasets.ImageFolder(test_dir, transform=eval_transform(cfg["img"]))
to_model_index = {i: classes.index(name) for i, name in enumerate(ds.classes)}

all_probs, truth = [], []
with torch.no_grad():
    for xb, yb in DataLoader(ds, batch_size=64):
        all_probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
        truth += [to_model_index[int(i)] for i in yb]
probs, truth = np.concatenate(all_probs), np.array(truth)

is_light = truth != none_index   # TODO 1: True for crops whose TRUE class is not none (hint: compare truth with none_index)
is_none = ~is_light
print(f"real lights: {is_light.sum()}   true none crops: {is_none.sum()}")

# ---- Question 1: ignore 'none' ----
normal_guess = probs.argmax(axis=1)
without_none = probs.copy()                  # copy, so the original probs stay untouched for Question 2
without_none[:, none_index] = 0         # TODO 2: what number makes the none column unable to win an argmax?
no_none_guess = without_none.argmax(axis=1)    # TODO 3: the best class per row, now that none cannot win
top3 = np.argsort(-without_none, axis=1)[:, :3]

print("\nQUESTION 1: accuracy on the real lights only")
print(f"  normal guess (none allowed):       {(normal_guess[is_light] == truth[is_light]).mean():.3f}")
print(f"  best guess with none ignored:      {(no_none_guess[is_light] == truth[is_light]).mean():.3f}")
print(f"  true class among best 3 (no none): {np.mean([truth[i] in top3[i] for i in np.where(is_light)[0]]):.3f}")
print(f"  always guessing the most common class: {np.bincount(truth[is_light]).max() / is_light.sum():.3f}")

# ---- Question 2: the 'none' penalty ----
print("\nQUESTION 2: subtract a penalty from the none score (in log-probability units)")
print(f"{'penalty':>8s} {'lights right':>13s} {'none crops called a light':>26s}")
log_probs = np.log(probs + 1e-9)             # the 1e-9 avoids log(0)
for penalty in (0, 1, 2, 3, 4, 6):
    shifted = log_probs.copy()
    shifted[:, none_index] -= penalty
    guess = shifted.argmax(axis=1)
    lights_right = (guess[is_light] == truth[is_light]).mean()
    false_alarms = (guess[is_none] != none_index).mean()  # TODO 4: fraction of the true-none crops whose guess is NOT none
    print(f"{penalty:8d} {lights_right:13.3f} {false_alarms:26.3f}")

# ---- Per class: where does the signal live? ----
print("\nPER CLASS (real lights only)")
print(f"{'class':15s} {'crops':>5s} {'right now':>9s} {'none ignored':>12s} {'in best 3':>9s}   most common guess (none ignored)")
for c in np.unique(truth[is_light]):
    rows = truth == c                                    # True for crops whose real class is c
    right_now = (normal_guess[rows] == c).sum()
    right_no_none = (no_none_guess[rows] == c).sum()                                  # TODO A: same as the line above, but using no_none_guess
    in_best_3 = sum(c in top3[i] for i in np.where(rows)[0])
    common_guess = np.bincount(no_none_guess[rows], minlength=len(classes)).argmax()                                  # TODO B: the most common value in no_none_guess[rows]
    print(f"{classes[c]:15s} {rows.sum():5d} {right_now:9d} {right_no_none:12d} {in_best_3:9d}   {classes[common_guess]}")

# ---- What does each real class get called? (none ignored) ----
print("\nWHAT EACH REAL CLASS GETS CALLED (none ignored)")
for c in np.unique(truth[is_light]):
    rows = truth == c
    counts = np.bincount(no_none_guess[rows], minlength=len(classes))
    ranked = np.argsort(-counts)        # TODO: class numbers sorted by count, biggest first (hint: np.argsort(-counts))
    parts = [f"{classes[k]} {counts[k]}" for k in ranked if counts[k] > 0]
    print(f"{classes[c]:15s} ({rows.sum():2d}):  " + ", ".join(parts))

# ---- Is blur the reason? ----
from PIL import Image


def sharpness(path):
    """How much fine detail a crop has: the variance of its Laplacian (a high value = crisp edges, a low value = blurry)."""
    g = np.asarray(Image.open(path).convert("L").resize((96, 96)), dtype=float)
    laplacian = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return laplacian.var()

paths = [p for p, _ in ds.samples]        # TODO 1: the file path of every crop (hint: ds.samples is a list of (path, class_number) pairs)
sharp = np.array([sharpness(p) for p in paths])        # TODO 2: an np.array with sharpness(path) for every path
side = np.array([Image.open(p).size[0] for p in paths])   # width in pixels of each saved crop
median_sharp = np.median(sharp[is_light])
print(f"\nIS BLUR THE REASON?  (real lights only, split at the median sharpness {median_sharp:.0f})")
print(f"{'group':15s} {'crops':>5s} {'right now':>9s} {'none ignored':>12s} {'in best 3':>9s} {'median width px':>16s}")
for name, group in (("sharper half", is_light & (sharp >= median_sharp)), ("blurrier half", is_light & (sharp < median_sharp))):
    print(f"{name:15s} {group.sum():5d} {(normal_guess[group] == truth[group]).sum():9d} "
          f"{(no_none_guess[group] == truth[group]).sum():12d} "
          f"{sum(truth[i] in top3[i] for i in np.where(group)[0]):9d} {np.median(side[group]):16.0f}")