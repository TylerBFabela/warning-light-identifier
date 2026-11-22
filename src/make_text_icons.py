"""Draw text-only warning lights (some cars show the word instead of a symbol) as icon PNGs for the generator.

Usage:  python src/make_text_icons.py icons
Writes icons/brake_system/text_BRAKE.png and icons/abs/text_ABS.png: bold black text on white, with a margin.
Uses Pillow's built-in font, so there is no extra font file to download.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WORDS = {"brake_system": "BRAKE", "abs": "ABS", "parking_brake": "PARK"}   # class folder -> the word some cars light up instead of a symbol
out_dir = Path(sys.argv[1])
try:
    font = ImageFont.load_default(size=200)
except TypeError:
    sys.exit("This Pillow is too old for a sizable built-in font: pip install --upgrade pillow")

for cls, word in WORDS.items():
    box = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), word, font=font, stroke_width=8)
    width, height = box[2] - box[0], box[3] - box[1]
    pad = int(0.15 * max(width, height))                                       # plain margin around the word
    img = Image.new("RGB", (width + 2 * pad, height + 2 * pad), "white")
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), word, fill="black", font=font,
                             stroke_width=8, stroke_fill="black")              # the stroke makes the letters bold
    target = out_dir / cls / f"text_{word}.png"
    img.save(target)
    print("wrote", target, img.size)