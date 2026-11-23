"""Measure the light finder (src/find_lights.py) against the hand-drawn boxes of a YOLO dataset.

Usage:  python src/finder_eval.py external_raw_rf2 check_icons/finder_eval [--params '{"merge_cap_frac": 0.15}'] [--splits valid,test]
For every labeled light, the finder's boxes are checked:
  clean     = a finder box is centered on the light and fits it (overlap, IoU, of at least MIN_IOU)
  poor      = a finder box covers the light's center but fits badly (merged with neighbours, far too big, or a fragment)
  missed    = no finder box covers the light's center
It also reports finder boxes per photo and how many touch no labeled light. Those are not all mistakes: the dataset labels
only some classes, so other lit lights get boxes too. A picture is saved for every photo with a poor or missed light:
green = labeled light found cleanly, yellow = poorly framed, red = missed, orange = the finder's boxes.
Use --splits valid,test to skip Roboflow's 'train' copies: they are rotated and cropped a little, and their boxes can be off.
"""
import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from find_lights import propose_lights

DETECT_SIDE = 1024          # the website looks for lights on a copy no bigger than this
MIN_IOU = 0.15              # finder squares carry a 25% margin, so a perfect box on a square icon scores about 0.44
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def evaluate(dataset, params=None, splits=("valid", "test", "train"), out=None):
    """Run the finder with `params` on one photo per source and score it. Returns a dict of counts."""
    params = params or {}
    names = yaml.safe_load(next(dataset.rglob("data.yaml")).read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    chosen = {}                                               # one photo per source, in the order of `splits`
    for split in splits:
        for label_file in sorted(dataset.rglob(f"{split}/labels/*.txt")):
            images = label_file.parent.parent / "images"
            photo = next((p for p in images.iterdir() if p.stem == label_file.stem and p.suffix.lower() in IMAGE_TYPES), None)
            source = label_file.stem.split(".rf.")[0]
            if photo is not None and source not in chosen:
                chosen[source] = (photo, label_file)
    result = {"photos": len(chosen), "clean": 0, "poor": 0, "missed": 0, "boxes": 0, "unmatched": 0, "per_class": {}}
    for source, (photo, label_file) in sorted(chosen.items()):
        img = Image.open(photo).convert("RGB")
        k = min(1.0, DETECT_SIDE / max(img.size))
        small = img.resize((round(img.width * k), round(img.height * k))) if k < 1 else img
        finder = [tuple(c / k for c in b) for b in propose_lights(small, **params)]      # back to full-size pixels
        labels = []
        for line in label_file.read_text().splitlines():
            cls, cx, cy, w, h = line.split()[:5]
            cx, w, cy, h = float(cx) * img.width, float(w) * img.width, float(cy) * img.height, float(h) * img.height
            labels.append((names[int(cls)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        used, verdicts = set(), []
        for cls, box in labels:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            covering = [i for i, f in enumerate(finder) if f[0] <= cx <= f[2] and f[1] <= cy <= f[3]]
            best = max(covering, key=lambda i: iou(box, finder[i]), default=None)
            verdict = "missed" if best is None else ("clean" if iou(box, finder[best]) >= MIN_IOU else "poor")
            used.update(covering)
            verdicts.append((cls, box, verdict))
            result[verdict] += 1
            result["per_class"].setdefault(cls, {"clean": 0, "poor": 0, "missed": 0})[verdict] += 1
        result["boxes"] += len(finder)
        result["unmatched"] += len(finder) - len(used)
        if out is not None and any(v != "clean" for _, _, v in verdicts):
            pen = ImageDraw.Draw(img)
            for f in finder:
                pen.rectangle(f, outline=(255, 150, 0), width=2)
            for cls, box, verdict in verdicts:
                colour = {"clean": (0, 220, 0), "poor": (255, 230, 0), "missed": (255, 0, 0)}[verdict]
                pen.rectangle(box, outline=colour, width=4)
                pen.text((box[0] + 2, box[1] + 2), f"{cls}: {verdict}", fill=colour)
            out.mkdir(parents=True, exist_ok=True)
            img.save(out / f"{source[:60]}.jpg", quality=90)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("out")
    ap.add_argument("--params", default="{}", help='finder options as JSON, for example \'{"merge_cap_frac": 0.15}\'')
    ap.add_argument("--splits", default="valid,test,train")
    a = ap.parse_args()
    r = evaluate(Path(a.dataset), json.loads(a.params), a.splits.split(","), Path(a.out))
    n = r["clean"] + r["poor"] + r["missed"]
    print(f"photos: {r['photos']}   labeled lights: {n}")
    print(f"clean {r['clean']} ({r['clean'] / max(n, 1):.0%})   poor {r['poor']} ({r['poor'] / max(n, 1):.0%})   "
          f"missed {r['missed']} ({r['missed'] / max(n, 1):.0%})")
    print(f"finder boxes per photo: {r['boxes'] / max(r['photos'], 1):.1f}   boxes touching no labeled light: {r['unmatched']}")
    print(f"\n{'dataset class':40s} {'clean':>6s} {'poor':>6s} {'missed':>7s}")
    for cls, c in sorted(r["per_class"].items()):
        print(f"{cls[:40]:40s} {c['clean']:6d} {c['poor']:6d} {c['missed']:7d}")
    print(f"\npictures of photos with poor or missed lights: {a.out}/")