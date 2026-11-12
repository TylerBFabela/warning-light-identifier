"""Synthetic training images for the warning-light classifier.

Usage:  python make_synth.py OUT_DIR PER_CLASS [--icons icons] [--bg backgrounds] [--size 128] [--seed 0]

--icons  folder like icons/<class>/*.png plus icons/classes.json ({"battery": "red", ...}).
         Without it, three PLACEHOLDER shapes are used so you can test the pipeline.
--bg     optional folder of real dashboard/interior photos used as backgrounds (check their licenses).
Output:  OUT_DIR/<class>/*.jpg, one folder per class, plus a "none" class.
"""
import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

SIZE = 128
COLORS = {"red": (230, 40, 40), "amber": (255, 170, 0), "green": (40, 200, 90), "blue": (60, 120, 255)}


# ---- placeholder icons (replace with real icon masks) -------------------------------------------
def draw_battery(d, s):
    d.rectangle([s * .15, s * .30, s * .85, s * .75], outline=255, width=max(3, s // 20))
    d.rectangle([s * .25, s * .20, s * .35, s * .30], fill=255)
    d.rectangle([s * .65, s * .20, s * .75, s * .30], fill=255)


def draw_oil(d, s):
    d.polygon([(s * .15, s * .45), (s * .55, s * .40), (s * .80, s * .55), (s * .55, s * .70), (s * .20, s * .65)], outline=255, width=max(3, s // 20))
    d.ellipse([s * .78, s * .60, s * .92, s * .80], fill=255)


def draw_tire(d, s):
    d.arc([s * .15, s * .15, s * .85, s * .85], 200, 340, fill=255, width=max(3, s // 18))
    d.arc([s * .15, s * .15, s * .85, s * .85], 20, 160, fill=255, width=max(3, s // 18))
    d.rectangle([s * .47, s * .30, s * .53, s * .58], fill=255)
    d.ellipse([s * .46, s * .64, s * .54, s * .72], fill=255)


CLASSES = {"battery": (draw_battery, "red"), "oil_pressure": (draw_oil, "red"), "tire_pressure": (draw_tire, "amber")}


def mask_for(fn, size=SIZE):
    m = Image.new("L", (size, size), 0)
    fn(ImageDraw.Draw(m), size)
    return m


# ---- real icons -----------------------------------------------------------------------------------
def mask_from_file(path, size=256):
    """Icon file -> white-on-black mask. Handles transparent PNGs and dark-on-white / white-on-dark icons."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") and np.asarray(im.getchannel("A")).std() > 1:
        m = im.getchannel("A")
    else:
        g = im.convert("L")
        m = ImageOps.invert(g) if np.asarray(g).mean() > 127 else g
    return ImageOps.autocontrast(ImageOps.pad(m, (size, size), color=0))


def load_icons(icons_dir):
    cfg = json.loads((Path(icons_dir) / "classes.json").read_text())
    out = {}
    for cls, color in cfg.items():
        files = sorted((Path(icons_dir) / cls).glob("*.png"))
        if not files:
            raise SystemExit(f"no PNG icons found in {Path(icons_dir) / cls}")
        out[cls] = ([mask_from_file(f) for f in files], color)
    return out


# ---- scene pieces ---------------------------------------------------------------------------------
def background(size, bg_files):
    if bg_files and random.random() < 0.7:
        im = Image.open(random.choice(bg_files)).convert("RGB")
        w, h = im.size
        s = random.randint(max(8, min(w, h) // 3), min(w, h))
        x, y = random.randint(0, w - s), random.randint(0, h - s)
        return ImageEnhance.Brightness(im.crop((x, y, x + s, y + s)).resize((size, size))).enhance(random.uniform(0.2, 0.6))
    base = random.randint(8, 55)
    bg = Image.new("RGB", (size, size), (base, base, base + random.randint(0, 12)))
    d = ImageDraw.Draw(bg)
    for _ in range(random.randint(0, 5)):  # distractors: gauge arcs, ticks, text-like bars
        x, y = random.randint(-20, size), random.randint(-20, size)
        d.arc([x, y, x + size // 2, y + size // 2], random.randint(0, 180), random.randint(181, 359), fill=(random.randint(60, 140),) * 3, width=random.randint(1, 3))
    for _ in range(random.randint(0, 3)):
        x, y = random.randint(0, size), random.randint(0, size)
        d.rectangle([x, y, x + random.randint(6, 30), y + 3], fill=(random.randint(70, 150),) * 3)
    return bg


def warp(m):
    j = lambda: random.uniform(-0.12, 0.12)
    coeffs = (1 + j() * .5, j() * .5, 0, j() * .5, 1 + j() * .5, 0, j() * .002, j() * .002)
    return m.transform(m.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def place_icon(bg, mask, color, brightness, size=SIZE):
    s = int(size * random.uniform(0.45, 0.8))
    m = warp(mask.resize((s, s)).rotate(random.uniform(-12, 12)))
    col = tuple(int(c * brightness) for c in COLORS[color])
    layer = Image.new("RGB", (s, s), col)
    x, y = random.randint(0, size - s), random.randint(0, size - s)
    out = bg.copy()
    bloom = m.filter(ImageFilter.GaussianBlur(max(2, s * random.uniform(0.04, 0.10)))).point(lambda v: int(v * 0.6))
    out.paste(layer, (x, y), bloom)                                                       # LED glow
    out.paste(layer, (x, y), m.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.2))))  # the icon
    return out


def add_decoys(img):
    """Bright colored things that are NOT warning icons (needles, text, dots). Teaches the model to say 'none'."""
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 3)):
        col = tuple(int(c * random.uniform(0.6, 1)) for c in random.choice(list(COLORS.values())))
        x, y, kind = random.randint(5, img.width - 40), random.randint(5, img.height - 40), random.choice(["bar", "dot", "arc", "text"])
        if kind == "bar":
            d.rectangle([x, y, x + random.randint(15, 60), y + random.randint(2, 5)], fill=col)
        elif kind == "dot":
            r = random.randint(2, 8); d.ellipse([x, y, x + r, y + r], fill=col)
        elif kind == "arc":
            d.arc([x, y, x + random.randint(20, 60), y + random.randint(20, 60)], random.randint(0, 180), random.randint(181, 330), fill=col, width=random.randint(2, 4))
        else:
            for k in range(random.randint(3, 6)):
                d.rectangle([x + k * 7, y, x + k * 7 + 4, y + random.randint(6, 12)], fill=col)
    return img


def degrade(img, size=SIZE):
    if random.random() < 0.4:  # glare
        glare = Image.new("RGB", img.size, (0, 0, 0))
        x, y = random.randint(0, size), random.randint(0, size)
        ImageDraw.Draw(glare).ellipse([x, y, x + random.randint(20, 70), y + random.randint(10, 40)], fill=(random.randint(60, 140),) * 3)
        img = ImageChops.add(img, glare.filter(ImageFilter.GaussianBlur(10)))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
    arr = np.asarray(img).astype(np.int16) + np.random.randint(-10, 11, (size, size, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(35, 90))  # compression artifacts, like a phone photo
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def generate(out_dir, per_class, icons=None, bg_dir=None, size=SIZE, seed=0):
    random.seed(seed); np.random.seed(seed)
    out = Path(out_dir)
    bg_files = sorted(p for p in Path(bg_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if bg_dir else []
    icon_set = load_icons(icons) if icons else {n: ([mask_for(fn)], c) for n, (fn, c) in CLASSES.items()}
    counts = {}
    for name, (masks, color) in icon_set.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = place_icon(background(size, bg_files), random.choice(masks), color, random.uniform(0.75, 1.0), size)
            degrade(img, size).save(out / name / f"{i:05d}.jpg")
        counts[name] = per_class
    (out / "none").mkdir(parents=True, exist_ok=True)
    for i in range(per_class):  # "none": empty clusters, UNLIT icons (very dim), and bright decoys
        img = background(size, bg_files)
        r = random.random()
        if r < 0.4:
            masks, color = random.choice(list(icon_set.values()))
            img = place_icon(img, random.choice(masks), color, random.uniform(0.04, 0.12), size)
        elif r < 0.75:
            img = add_decoys(img)
        degrade(img, size).save(out / "none" / f"{i:05d}.jpg")
    counts["none"] = per_class
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("per_class", type=int)
    ap.add_argument("--icons"); ap.add_argument("--bg"); ap.add_argument("--size", type=int, default=SIZE); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(generate(a.out_dir, a.per_class, a.icons, a.bg, a.size, a.seed))