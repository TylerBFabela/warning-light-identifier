"""Draw a YOLO dataset's boxes onto its photos so you can check the labels by eye.

Usage:  python src/draw_yolo_boxes.py DATASET_FOLDER OUT_FOLDER [HOW_MANY]
Finds every .../labels/*.txt anywhere under DATASET_FOLDER, pairs it with the photo of the same name in the
neighbouring images/ folder, and prints what it found so a problem is never silent.
"""
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

dataset, out = Path(sys.argv[1]), Path(sys.argv[2])
how_many = int(sys.argv[3]) if len(sys.argv) > 3 else 10
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

yaml_files = sorted(dataset.rglob("data.yaml"))
if not yaml_files:
    sys.exit(f"No data.yaml under {dataset}. Is this the folder you unzipped?")
names = yaml.safe_load(yaml_files[0].read_text())["names"]
names = list(names.values()) if isinstance(names, dict) else list(names)

label_files = sorted(dataset.rglob("labels/*.txt"))                       # works however deep the folders are nested
print(f"classes in data.yaml: {names}")
print(f"label files found: {len(label_files)}")
if not label_files:
    sys.exit("No .../labels/*.txt files found. Send me: find YOUR_FOLDER -maxdepth 3 -type d")

out.mkdir(parents=True, exist_ok=True)
saved = boxes_drawn = without_photo = 0
for label_file in label_files[:how_many]:
    images_dir = label_file.parent.parent / "images"
    image_path = next((p for p in images_dir.iterdir() if p.stem == label_file.stem and p.suffix.lower() in IMAGE_TYPES), None) if images_dir.is_dir() else None
    if image_path is None:
        without_photo += 1
        continue
    photo = Image.open(image_path).convert("RGB")
    pen = ImageDraw.Draw(photo)
    for line in label_file.read_text().splitlines():
        class_id, cx, cy, w, h = line.split()[:5]
        cx, w = float(cx) * photo.width, float(w) * photo.width                 # fractions of the image -> pixels
        cy, h = float(cy) * photo.height, float(h) * photo.height
        pen.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(0, 255, 0), width=3)
        pen.text((cx - w / 2 + 3, cy - h / 2 + 3), names[int(class_id)], fill=(255, 255, 0))
        boxes_drawn += 1
    photo.save(out / f"{label_file.parent.parent.name}_{label_file.stem}_boxes.jpg", quality=90)
    saved += 1
print(f"previews saved: {saved} | boxes drawn: {boxes_drawn} | labels with no matching photo: {without_photo}")
print("saved previews to", out)