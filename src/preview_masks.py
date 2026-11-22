"""One picture showing what the generator 'sees' for every icon: left = your file, right = the shape it extracts.

Usage:  python src/preview_masks.py icons_holdout sheet_holdout.png
White in the right-hand tile = the symbol. A good icon shows the full symbol, with its inner details, centered.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from make_synth import mask_from_file

folder, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p for p in folder.rglob("*.png"))
TILE = 128
COLUMNS = 4
rows = (len(files) + COLUMNS - 1) // COLUMNS
sheet = Image.new("RGB", (COLUMNS * (2 * TILE + 12), rows * (TILE + 22)), (30, 30, 30))
for n, path in enumerate(files):
    original = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", original.size, (128, 128, 128, 255))     # mid-gray so white and black icons both show
    background.alpha_composite(original)
    mask = mask_from_file(path, size=TILE)
    x, y = (n % COLUMNS) * (2 * TILE + 12), (n // COLUMNS) * (TILE + 22)
    sheet.paste(background.convert("RGB").resize((TILE, TILE)), (x + 2, y + 18))
    sheet.paste(mask.convert("RGB"), (x + TILE + 6, y + 18))                  # What would the picture look like if this line were missing?
    ImageDraw.Draw(sheet).text((x + 2, y + 3), f"{path.parent.name}/{path.name}"[:34], fill=(255, 255, 0))
sheet.save(out)
print(f"{len(files)} icons -> {out}")