"""Cut out every lit light the app would find, so you can label them and build a real test set.

Usage:  python src/save_crops.py frames crops_to_label
Reads every photo in frames/ (JPEG, PNG, HEIC...), saves each candidate crop (cut from the FULL-resolution photo)
to crops_to_label/<photo>_<ext>_<n>.jpg and a preview showing the boxes to crops_to_label/previews/.
Afterwards, move each crop into real_test/<class>/ by hand (use real_test/none/ if it is not a warning light).
Compare each preview with the real dashboard: a lit icon with no box is a miss by the light finder.
"""
import sys
from pathlib import Path

from PIL import ImageDraw

from find_lights import propose_lights
from load_photo import load_photo

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
(dst / "previews").mkdir(parents=True, exist_ok=True)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}

for path in sorted(p for p in src.iterdir() if p.suffix.lower() in EXTENSIONS):
    try:
        full = load_photo(path)
    except ValueError as problem:
        print(f"skipped {path.name}: {problem}")
        continue
    name = f"{path.stem}_{path.suffix[1:].lower()}"     # e.g. frame_0001_jpg, so two files with the same stem never collide
    small = full.copy()
    small.thumbnail((1024, 1024))                      # find the lights on a small copy (fast)...
    scale = full.width / small.width
    boxes = propose_lights(small)
    preview = small.copy()
    pen = ImageDraw.Draw(preview)
    for number, box in enumerate(boxes, start=1):
        crop = full.crop(tuple(round(c * scale) for c in box))   # ...but cut the crop from the original (more detail)
        crop.thumbnail((256, 256))                     # a review copy; plenty for labeling and testing at 96 px
        crop.save(dst / f"{name}_{number}.jpg", quality=95)
        pen.rectangle(box, outline=(0, 255, 0), width=2)
        pen.text((box[0] + 3, box[1] + 3), str(number), fill=(0, 255, 0))
    preview.save(dst / "previews" / f"{name}_boxes.jpg", quality=90)
    print(f"{path.name}: {len(boxes)} candidate light(s)")