"""Compare groups of models (for example the seeds of v4 and of v5a) on several real crop sets at once.

Usage:  python src/compare_models.py
Edit GROUPS, SETS and MATRIX_FOR below. Everything is pooled over all SETS. Prints:
  1. per class: crops right with 'none' allowed (average over a group's seeds, and the group's ensemble)
  2. false alarms: true 'none' crops called a light, and how many of those were at the website's 'identified' confidence
  3. a confusion matrix (rows = true class, columns = guess) for one model
Test-only: nothing is trained and no weights change.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from common import eval_transform, get_device, load_model

GROUPS = {                                       # name -> model folders (several = several seeds)
    "v3": ["models/v3"],
    "v4": ["models/v4_pre", "models/v4_pre_s1", "models/v4_pre_s2"],
    "v5a": ["models/v5a_s0", "models/v5a_s1", "models/v5a_s2"],
    "v5b": ["models/v5b_s0", "models/v5b_s1", "models/v5b_s2"],
}
SETS = ["external_dev", "real_dev", "real_test", "real_night"]
MATRIX_FOR = "models/v5b_s1"                     # the model whose confusion matrix is printed
SURE = 0.9                                       # the website's "identified" cutoff
SHORT = {"abs": "abs", "airbag": "air", "battery": "bat", "brake_system": "brk", "check_engine": "eng", "headlights": "hdl",
         "high_beam": "hib", "none": "non", "oil_pressure": "oil", "parking_brake": "prk", "tire_pressure": "tir",
         "turn_left": "t_l", "turn_right": "t_r"}
device = get_device()


def log_probs(model_dir):
    """Log-probabilities for every crop of every set (always in the same order), the true classes, and the class names."""
    model, classes, cfg = load_model(model_dir, device)
    out, truth = [], []
    for s in SETS:
        ds = datasets.ImageFolder(s, transform=eval_transform(cfg["img"]), allow_empty=True)
        to_model = {i: classes.index(name) for i, name in enumerate(ds.classes)}
        with torch.no_grad():
            for xb, yb in DataLoader(ds, batch_size=64):
                out.append(torch.log_softmax(model(xb.to(device)), dim=1).cpu().numpy())
                truth += [to_model[int(i)] for i in yb]
    return np.concatenate(out), np.array(truth), classes


cache = {d: log_probs(d) for dirs in GROUPS.values() for d in dirs}
if MATRIX_FOR not in cache:
    cache[MATRIX_FOR] = log_probs(MATRIX_FOR)
_, truth, classes = cache[MATRIX_FOR]
assert all(c == classes for _, _, c in cache.values()), "all models must have the same classes in the same order"
none = classes.index("none")
is_none = truth == none


def summary(lp):
    """Per-class right counts, false alarms, and false alarms at SURE confidence or more."""
    guess, conf = lp.argmax(1), np.exp(lp.max(1))
    right = np.array([(guess[truth == c] == c).sum() for c in range(len(classes))])
    alarm = is_none & (guess != none)
    return right, alarm.sum(), (alarm & (conf >= SURE)).sum()


columns = []                                     # (title, right per class, false alarms, sure false alarms)
for name, dirs in GROUPS.items():
    runs = [summary(cache[d][0]) for d in dirs]
    columns.append((f"{name} avg" if len(dirs) > 1 else name, np.mean([r[0] for r in runs], 0),
                    np.mean([r[1] for r in runs]), np.mean([r[2] for r in runs])))
    if len(dirs) > 1:
        lp = np.mean([cache[d][0] for d in dirs], 0)
        lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))      # renormalize the averaged log-probabilities
        columns.append((f"{name} ens", *summary(lp)))

present = [c for c in range(len(classes)) if (truth == c).any()]
print(f"pooled over {', '.join(SETS)}: {(~is_none).sum()} real lights, {is_none.sum()} none crops\n")
print("1. RIGHT ANSWERS PER CLASS ('none' allowed)")
print(f"{'class':15s} {'crops':>5s}" + "".join(f"{t:>10s}" for t, *_ in columns))
for c in present:
    print(f"{classes[c]:15s} {(truth == c).sum():5d}" + "".join(f"{r[c]:10.1f}" for _, r, _, _ in columns))
lights = [c for c in present if c != none]
print(f"{'ALL LIGHTS':15s} {(~is_none).sum():5d}" + "".join(f"{r[lights].sum():10.1f}" for _, r, _, _ in columns))

print(f"\n2. FALSE ALARMS (of {is_none.sum()} none crops)")
print(f"{'called a light':21s}" + "".join(f"{fa:10.1f}" for _, _, fa, _ in columns))
print(f"{'... at >= ' + str(SURE) + ' conf':21s}" + "".join(f"{fs:10.1f}" for _, _, _, fs in columns))

print(f"\n3. CONFUSION MATRIX for {MATRIX_FOR} (rows = true class, columns = guess, 'none' allowed)")
guess = cache[MATRIX_FOR][0].argmax(1)
short = [SHORT.get(n, n[:3]) for n in classes]
print(f"{'':15s}" + "".join(f"{s:>5s}" for s in short))
for c in present:
    row = [(guess[truth == c] == g).sum() for g in range(len(classes))]
    print(f"{classes[c]:15s}" + "".join(f"{v:5d}" if v else f"{'.':>5s}" for v in row))


print(f"\n4. 'NONE' PENALTY for {MATRIX_FOR} (subtracted from the none score before choosing, as the website would do)")
print(f"{'penalty':>8s} {'lights right':>13s} {'false alarms':>13s} {'identified >= ' + str(SURE):>18s} {'of those wrong':>15s}")
for penalty in (0, 0.5, 1, 1.5, 2, 3):
    lp = cache[MATRIX_FOR][0].copy()
    lp[:, none] -= penalty
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))         # back to probabilities that add up to 1
    guess, conf = lp.argmax(1), np.exp(lp.max(1))
    named = guess != none
    sure = named & (conf >= SURE)                               # what the website would show as "identified"
    print(f"{penalty:8.1f} {((guess == truth) & ~is_none).sum():13d} {(named & is_none).sum():13d} "
          f"{sure.sum():18d} {(sure & (guess != truth)).sum():15d}")


SITE_PENALTY = 1.0                               # the none penalty the website uses (docs/model/config.json)
lp = cache[MATRIX_FOR][0].copy()
lp[:, none] -= SITE_PENALTY
lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
p_none, guess = np.exp(lp[:, none]), lp.argmax(1)
print(f"\n5. HIDING SPOTS CALLED 'none' ({MATRIX_FOR}, penalty {SITE_PENALTY}): hide only when 'none' confidence >= cutoff")
print(f"{'cutoff':>7s} {'junk hidden':>12s} {'real lights hidden':>19s}   hidden lights by class")
for cutoff in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    hidden = (guess == none) & (p_none >= cutoff)
    lost = hidden & ~is_none
    names = ", ".join(f"{classes[c]} {n}" for c, n in enumerate(np.bincount(truth[lost], minlength=len(classes))) if n)
    print(f"{cutoff:7.2f} {(hidden & is_none).sum():8d}/{is_none.sum():<3d} {lost.sum():12d}/{(~is_none).sum():<5d}   {names}")