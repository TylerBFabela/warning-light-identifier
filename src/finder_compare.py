"""Draw the current finder's boxes next to a new setting's boxes on your own photos (no labels needed).

Usage:  python src/finder_compare.py frames check_icons/finder_compare '{"merge_cap_frac": 0.15}'
Saves one picture per photo: left = current finder, right = the new setting, with the number of boxes on each.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from find_lights import propose_lights
from load_photo import load_photo

src, dst, params = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}
for path in sorted(p for p in src.iterdir() if p.suffix.lower() in EXTENSIONS):
    try:
        small = load_photo(path)
    except ValueError as problem:
        print(f"skipped {path.name}: {problem}")
        continue
    small.thumbnail((1024, 1024))                                  # the same small copy the website searches
    panels = []
    for title, kw in (("current", {}), ("new", params)):
        boxes = propose_lights(small, **kw)
        panel = small.copy()
        pen = ImageDraw.Draw(panel)
        for box in boxes:
            pen.rectangle(box, outline=(0, 255, 255), width=3)
        pen.text((8, 8), f"{title}: {len(boxes)} boxes", fill=(255, 255, 0))
        panels.append(panel)
    both = Image.new("RGB", (small.width * 2 + 10, small.height), (30, 30, 30))
    both.paste(panels[0], (0, 0)); both.paste(panels[1], (small.width + 10, 0))
    both.save(dst / f"{path.stem}.jpg", quality=88)
    print(f"{path.name}: current {len(propose_lights(small))} boxes, new {len(propose_lights(small, **params))} boxes")