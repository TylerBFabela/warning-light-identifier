"""Turn a downloaded YOLO-format dataset (photos + box labels) into class folders of crops our classifier can use.

Usage:  python src/yolo_to_crops.py DATASET_FOLDER MAPPING.json OUT_FOLDER
DATASET_FOLDER has data.yaml (class names) and train/ valid/ test/ folders, each with images/ and labels/.
MAPPING.json maps THEIR class names to OUR class folder names, e.g. {"Check Engine": "check_engine"}.
Their classes that are not in the mapping are skipped (and counted).
Each box becomes a SQUARE crop with a 25% margin, the same shape the light finder produces, so training and testing match.
Roboflow exports often contain several augmented copies of one source photo (names like  photo_jpg.rf.<hash>.jpg);
only the first copy of each source photo is used, so the same light is not counted twice.
"""
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

dataset, mapping_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mapping = json.loads(mapping_file.read_text())
names = yaml.safe_load(next(dataset.rglob("data.yaml")).read_text())["names"]
names = list(names.values()) if isinstance(names, dict) else list(names)       # YOLO files number the classes 0, 1, 2...
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MARGIN = 0.25
STOCK_HINTS = ("shutterstock", "istock", "alamy", "dreamstime", "123rf", "gettyimages", "adobestock", "depositphotos", "260nw")
stock_skipped = 0
kept, skipped, seen_sources, duplicates = {}, {}, set(), 0

for label_file in sorted(dataset.rglob("labels/*.txt")):
    split = label_file.parent.parent.name
    images_dir = label_file.parent.parent / "images"
    image_path = next((p for p in images_dir.iterdir() if p.stem == label_file.stem and p.suffix.lower() in IMAGE_TYPES), None)
    if image_path is None:
        continue
    if any(hint in image_path.name.lower() for hint in STOCK_HINTS):
        stock_skipped += 1
        continue
    source_name = image_path.stem.split(".rf.")[0]
    if source_name in seen_sources:
        duplicates += 1
        continue
    seen_sources.add(source_name)
    photo = Image.open(image_path).convert("RGB")
    for n, line in enumerate(label_file.read_text().splitlines()):
        class_id, cx, cy, w, h = line.split()[:5]                              # centre x/y and width/height, as 0..1 of the image
        their_name = names[int(class_id)]
        if their_name not in mapping:
            skipped[their_name] = skipped.get(their_name, 0) + 1
            continue
        cx, w = float(cx) * photo.width, float(w) * photo.width
        cy, h = float(cy) * photo.height, float(h) * photo.height
        side = int(max(w, h) * (1 + 2 * MARGIN))
        left = int(cx - side / 2)
        top = int(cy - side / 2)
        right = left + side
        bottom = top + side
        crop = photo.crop((left, top, right, bottom))
        folder = out / mapping[their_name]
        folder.mkdir(parents=True, exist_ok=True)
        crop.save(folder / f"{split}_{len(seen_sources):04d}_{image_path.stem[:30]}_{n}.jpg", quality=95)
        kept[mapping[their_name]] = kept.get(mapping[their_name], 0) + 1
print("kept crops per class:", kept)
print("skipped (not in mapping):", skipped)
print("augmented duplicate photos ignored:", duplicates)
print("stock-photo names skipped:", stock_skipped)