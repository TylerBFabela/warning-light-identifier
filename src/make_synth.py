"""Synthetic training images for the warning-light classifier.

Usage:  python make_synth.py OUT_DIR PER_CLASS [--icons icons] [--bg backgrounds] [--size 128] [--seed 0]
                             [--hard-negatives] [--lowres] [--centered] [--none-factor 1.0]
                             [--color-jitter] [--bright-bg] [--labels]

--icons  folder like icons/<class>/*.png plus icons/classes.json ({"battery": "red", ...}).
         Without it, three PLACEHOLDER shapes are used so you can test the pipeline.
         Optional icons/none/*.png = real symbols that are NOT in classes.json (seat belt, master warning, ...);
         with --hard-negatives they are shown lit as "not a known light" examples.
--bg     optional folder of real dashboard/interior photos used as backgrounds (check their licenses).
Output:  OUT_DIR/<class>/*.jpg, one folder per class, plus a "none" class.

Realism switches (all off by default, so old runs are reproducible; turn them on one at a time to see what helps):
--hard-negatives  "none" gets what real dashboards show: other lit symbols, gauge needles and colored gauge zones,
                  orange LCD panels, text like AT/AWD, and glare, instead of mostly empty pictures.
--lowres          images are shrunk to 24-64 px and blown back up, like the small crops the light finder cuts out.
--centered        the icon sits near the middle at about 2/3 of the crop, like the finder's crops (it centers its box).
--color-jitter    LED colors vary within their family (orange..yellow-amber, lime..green, pinkish..pure red) instead of one fixed RGB.
--bright-bg       some backgrounds are pale blue-gray with a gradient, like glare on the cluster glass.
--labels          brake_system and parking_brake sometimes get their printed word (BRAKE / PARK) under the icon, as many cars do.
--none-factor     make this many times more "none" images than per-class images (none is the broadest class).
"""
import argparse
import io
import colorsys
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

SIZE = 128
OPTIONS = {"color_jitter": False, "bright_bg": False}      # switched on by generate(); read by the drawing helpers
HUE_RANGES = {"red": (-0.03, 0.025), "amber": (0.06, 0.14), "green": (0.24, 0.42), "blue": (0.55, 0.66)}   # hue as a 0..1 turn
LABELS = {"brake_system": "BRAKE", "parking_brake": "PARK"}
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
    """Icon file -> white-on-black mask (white = the symbol), cropped to the symbol and centered.
    Handles: 1) symbol on a transparent background, 2) colored/white symbol on a dark tile,
    3) dark symbol on a white or light background."""
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im).astype(np.float32)
    alpha = arr[..., 3] / 255
    if (alpha < 0.5).mean() > 0.25:
        # Lots of transparency: the symbol is whatever is NOT transparent.
        m = Image.fromarray((alpha * 255).astype(np.uint8))
    else:
        # A solid picture (a tile): the symbol is whatever differs from the background colour.
        rgb = arr[..., :3]
        solid = alpha > 0.5
        keys = (rgb // 32).astype(int) @ np.array([64, 8, 1])           # coarse colour buckets
        bg_key = np.bincount(keys[solid]).argmax()                      # most common colour = background
        bg = rgb[solid & (keys == bg_key)].mean(axis=0)
        dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2)) * solid           # distance from the background colour
        m = Image.fromarray(np.clip(dist / max(dist.max(), 1) * 255, 0, 255).astype(np.uint8))
    box = m.point(lambda v: 255 if v > 32 else 0).getbbox()             # tight box around the symbol
    if box:
        m = m.crop(box)
    inner = int(size * 0.92)                                            # small margin all around
    m = ImageOps.contain(m, (inner, inner))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(m, ((size - m.width) // 2, (size - m.height) // 2))
    return ImageOps.autocontrast(canvas)


def load_icons(icons_dir):
    cfg = json.loads((Path(icons_dir) / "classes.json").read_text())
    out = {}
    for cls, color in cfg.items():
        files = sorted((Path(icons_dir) / cls).glob("*.png"))
        if not files:
            raise SystemExit(f"no PNG icons found in {Path(icons_dir) / cls}")
        out[cls] = ([mask_from_file(f) for f in files], color)
    return out


def load_extra_icons(icons_dir):
    """Optional icons/none/*.png: real symbols that are not one of the known classes (used as lit negatives)."""
    if not icons_dir:
        return []
    return [mask_from_file(f) for f in sorted((Path(icons_dir) / "none").glob("*.png"))]


def lit_rgb(color, brightness=1.0):
    """RGB for an LED of this color family. With --color-jitter the hue and saturation vary within the family."""
    if not OPTIONS["color_jitter"]:
        return tuple(int(c * brightness) for c in COLORS[color])
    lo, hi = HUE_RANGES[color]
    r, g, b = colorsys.hsv_to_rgb(random.uniform(lo, hi) % 1.0, random.uniform(0.65, 1.0), min(1.0, brightness))
    return (int(r * 255), int(g * 255), int(b * 255))


# ---- scene pieces ---------------------------------------------------------------------------------
def background(size, bg_files):
    if bg_files and random.random() < 0.7:
        im = Image.open(random.choice(bg_files)).convert("RGB")
        w, h = im.size
        s = random.randint(max(8, min(w, h) // 3), min(w, h))
        x, y = random.randint(0, w - s), random.randint(0, h - s)
        return ImageEnhance.Brightness(im.crop((x, y, x + s, y + s)).resize((size, size))).enhance(random.uniform(0.2, 0.6))
    if OPTIONS["bright_bg"] and random.random() < 0.4:     # pale bluish-gray glare with a gradient
        base = random.randint(70, 150)
        row = np.linspace(0, random.uniform(-40, 40), size)[None, :, None] * np.ones((size, size, 3))
        tint = np.array([-random.randint(0, 20), 0, random.randint(10, 35)])
        bg = Image.fromarray(np.clip(base + row + tint, 0, 255).astype(np.uint8))
    else:
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


def place_icon(bg, mask, color, brightness, size=SIZE, centered=False, shrink=1.0, lift=0):
    # centered=True mimics the light finder: its square box sits on the icon, which fills about 2/3 of the crop
    s = int(size * (random.uniform(0.55, 0.75) if centered else random.uniform(0.45, 0.8)) * shrink)
    m = warp(mask.resize((s, s)).rotate(random.uniform(-12, 12)))
    col = lit_rgb(color, brightness)
    layer = Image.new("RGB", (s, s), col)
    if centered:
        x = min(max((size - s) // 2 + random.randint(-size // 8, size // 8), 0), size - s)
        y = min(max((size - s) // 2 + random.randint(-size // 8, size // 8), 0), size - s)
    else:
        x, y = random.randint(0, size - s), random.randint(0, size - s)
    y = max(0, y - lift)                                   # lift = leave room for a printed word below the icon
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


# ---- hard negatives: things real dashboards show that are NOT one of the known lights -------------------
def _font(px):
    """Text font about px pixels tall (falls back to Pillow's tiny built-in font if sizing is unsupported)."""
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def _text_mask(text, size, px):
    """White text on black, roughly centered, as a mask."""
    m = Image.new("L", (size, size), 0)
    try:
        ImageDraw.Draw(m).text((size // 2 + random.randint(-size // 10, size // 10), size // 2 + random.randint(-size // 10, size // 10)),
                               text, fill=255, font=_font(px), anchor="mm")
    except (ValueError, TypeError):
        pass                                                                      # bitmap font cannot center: leave empty
    return m


def _paste_lit(img, mask, rgb):
    """Paste a glowing shape: a blurred copy underneath (LED bloom) and a sharp copy on top."""
    layer = Image.new("RGB", img.size, rgb)
    bloom = mask.filter(ImageFilter.GaussianBlur(random.uniform(2, 5))).point(lambda v: int(v * 0.6))
    img.paste(layer, (0, 0), bloom)
    img.paste(layer, (0, 0), mask.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.0))))
    return img


def _lit_color(names=("red", "amber", "green", "blue")):
    if OPTIONS["color_jitter"]:
        return lit_rgb(random.choice(names), random.uniform(0.7, 1.0))
    return tuple(int(c * random.uniform(0.7, 1.0)) for c in COLORS[random.choice(names)])   # original behavior (a little tint noise)


def random_glyph(size=256):
    """A made-up symbol (white on black): an outline shape plus a few inner marks. Never a '!' inside a ring or a 'P'."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    w = random.randint(size // 16, size // 9)
    pad = size * 0.08
    shape = random.choice(["ring", "box", "blob", "none", "none"])
    if shape == "ring":
        d.ellipse([pad, pad, size - pad, size - pad], outline=255, width=w)
    elif shape == "box":
        d.rounded_rectangle([pad, size * .2, size - pad, size * .8], radius=size // 8, outline=255, width=w)
    elif shape == "blob":
        n = random.randint(5, 8)
        d.polygon([(size / 2 + math.cos(2 * math.pi * k / n) * size * random.uniform(.25, .45),
                    size / 2 + math.sin(2 * math.pi * k / n) * size * random.uniform(.25, .45)) for k in range(n)], fill=255)
    for _ in range(random.randint(1, 3)):
        kind = random.choice(["bar", "dot", "zigzag", "letter", "wrench", "arc"])
        if kind == "dot" and shape == "ring":
            continue                                                              # ring + dot could look like a brake light
        a, b = (random.uniform(size * .2, size * .8) for _ in range(2))
        if kind == "bar":
            d.line([a, size * .25, b, size * .75], fill=255, width=w)
        elif kind == "dot":
            r = size * random.uniform(.05, .11); d.ellipse([a - r, b - r, a + r, b + r], fill=255)
        elif kind == "zigzag":
            d.line([(size * k / 4, size * random.uniform(.3, .7)) for k in range(5)], fill=255, width=w)
        elif kind == "wrench":
            d.line([size * .2, size * .8, size * .7, size * .3], fill=255, width=w)
            d.ellipse([size * .6, size * .15, size * .85, size * .4], outline=255, width=w)
        elif kind == "arc":
            d.arc([pad, pad, size - pad, size - pad], random.randint(0, 180), random.randint(181, 340), fill=255, width=w)
        else:
            d.text((size / 2, size / 2), random.choice(["A", "AT", "S", "R", "T", "AWD", "OFF", "ECO", "W"]),
                   fill=255, font=_font(random.randint(size // 3, size // 2)), anchor="mm")
    box = m.getbbox()
    if box:
        m = m.crop(box)
    m = ImageOps.contain(m, (int(size * .92), int(size * .92)))
    out = Image.new("L", (size, size), 0)
    out.paste(m, ((size - m.width) // 2, (size - m.height) // 2))
    return out


def neg_glyph(img, extra, centered):
    return place_icon(img, random_glyph(), random.choice(list(COLORS)), random.uniform(0.7, 1.0), img.width, centered)


def neg_extra(img, extra, centered):
    if not extra:
        return neg_glyph(img, extra, centered)
    return place_icon(img, random.choice(extra), random.choice(list(COLORS)), random.uniform(0.75, 1.0), img.width, centered)


def neg_unlit(img, extra, centered, icon_set=None):
    masks, color = random.choice(list(icon_set.values()))
    return place_icon(img, random.choice(masks), color, random.uniform(0.04, 0.12), img.width, centered)


def neg_needle(img, extra=None, centered=False):
    """A thin tapered gauge needle (mostly red) crossing the crop."""
    s = img.width
    cx, cy, ang = s * random.uniform(.25, .75), s * random.uniform(.25, .75), random.uniform(0, 2 * math.pi)
    length, base = s * random.uniform(.5, 1.1), random.uniform(3, 9)
    dx, dy = math.cos(ang), math.sin(ang)
    tail, tip = (cx - dx * length / 2, cy - dy * length / 2), (cx + dx * length / 2, cy + dy * length / 2)
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).polygon([(tail[0] - dy * base / 2, tail[1] + dx * base / 2), (tail[0] + dy * base / 2, tail[1] - dx * base / 2), tip], fill=255)
    col = (235, 235, 235) if random.random() < 0.3 else _lit_color(("red",))
    return _paste_lit(img, m, col)


def neg_stripe(img, extra=None, centered=False):
    """A short red or blue bar like the hot/cold zone of a temperature gauge, with pale tick marks beside it."""
    s = img.width
    m = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(m)
    cx, cy, length, width = s / 2 + random.randint(-s // 8, s // 8), s / 2 + random.randint(-s // 8, s // 8), random.uniform(.3, .7) * s, random.uniform(.06, .13) * s
    d.rectangle([cx - length / 2, cy - width / 2, cx + length / 2, cy + width / 2], fill=255)
    ang = random.uniform(0, 180)
    m = m.rotate(ang, center=(s / 2, s / 2))
    ticks = Image.new("L", img.size, 0)
    td = ImageDraw.Draw(ticks)
    for k in range(random.randint(1, 4)):
        off = width * (1.6 + k * 1.2)
        td.line([cx - length / 2, cy - off, cx - length / 2 + random.uniform(.05, .2) * s, cy - off], fill=200, width=2)
    img.paste((190, 195, 200), (0, 0), ticks.rotate(ang, center=(s / 2, s / 2)))
    return _paste_lit(img, m, _lit_color(("red", "blue")))


def neg_lcd(img, extra=None, centered=False):
    """A big orange LCD panel with dark digits or letters (odometer / gear display)."""
    s = img.width
    w, h = int(s * random.uniform(.6, 1.0)), int(s * random.uniform(.35, .8))
    x, y = (s - w) // 2 + random.randint(-s // 10, s // 10), (s - h) // 2 + random.randint(-s // 10, s // 10)
    k = random.uniform(.55, .95)
    ImageDraw.Draw(img).rounded_rectangle([x, y, x + w, y + h], radius=s // 20, fill=(int(255 * k), int(random.randint(110, 170) * k), int(random.randint(20, 70) * k)))
    dark = _text_mask(random.choice(["69", "163566", "ODO", "R N D", "12.5", "TRIP A", "P R N D", "88:88"]), s, random.randint(s // 6, s // 3))
    img.paste((25, 12, 0), (0, 0), dark)
    return img


def neg_text(img, extra=None, centered=False):
    """Bare text like AT, AWD, HOLD, OFF (no class words such as PARK or BRAKE)."""
    word = random.choice(["AT", "AWD", "HOLD", "OFF", "ECO", "A", "4WD", "2WD", "SPORT", "MAINT", "S", "R", "N", "D", "x1000"])
    m = _text_mask(word, img.width, random.randint(img.width // 6, img.width // 3))
    col = (235, 235, 235) if random.random() < 0.25 else _lit_color(("amber", "green", "blue"))
    return _paste_lit(img, m, col)


def neg_glare(img, extra=None, centered=False):
    """Pale blurry reflections on the cluster glass or trim."""
    layer = Image.new("RGB", img.size, (0, 0, 0))
    d = ImageDraw.Draw(layer)
    tint = (random.randint(150, 255), random.randint(170, 255), random.randint(190, 255))
    for _ in range(random.randint(1, 3)):
        x, y = random.randint(0, img.width), random.randint(0, img.height)
        d.ellipse([x, y, x + random.randint(15, 60), y + random.randint(10, 40)], fill=tint)
    return ImageChops.add(img, layer.filter(ImageFilter.GaussianBlur(random.uniform(2, 8))))


def hard_negative(img, extra, icon_set, centered):
    """Pick one kind of 'not a known light' picture. Weights roughly follow what the real dashboards produced."""
    kinds = [("unlit", .10), ("empty", .06), ("decoy", .08), ("glyph", .24), ("extra", .14),
             ("stripe", .10), ("needle", .10), ("lcd", .06), ("text", .08), ("glare", .04)]
    kind = random.choices([k for k, _ in kinds], weights=[w for _, w in kinds])[0]
    if kind == "unlit":
        return neg_unlit(img, extra, centered, icon_set)
    if kind == "decoy":
        return add_decoys(img)
    if kind == "glyph":
        return neg_glyph(img, extra, centered)
    if kind == "extra":
        return neg_extra(img, extra, centered)
    if kind == "empty":
        return img
    return {"stripe": neg_stripe, "needle": neg_needle, "lcd": neg_lcd, "text": neg_text, "glare": neg_glare}[kind](img)


def add_label(img, word, color):
    """Print a word (BRAKE / PARK) under the icon in the icon's color, like many dashboards do."""
    m = Image.new("L", img.size, 0)
    try:
        ImageDraw.Draw(m).text((img.width // 2 + random.randint(-4, 4), int(img.height * 0.84)), word, fill=255,
                               font=_font(int(img.height * random.uniform(0.13, 0.18))), anchor="mm")
    except (ValueError, TypeError):
        return img
    return _paste_lit(img, m, lit_rgb(color, random.uniform(0.75, 1.0)))


def degrade(img, size=SIZE, lowres=False):
    if random.random() < 0.4:  # glare
        glare = Image.new("RGB", img.size, (0, 0, 0))
        x, y = random.randint(0, size), random.randint(0, size)
        ImageDraw.Draw(glare).ellipse([x, y, x + random.randint(20, 70), y + random.randint(10, 40)], fill=(random.randint(60, 140),) * 3)
        img = ImageChops.add(img, glare.filter(ImageFilter.GaussianBlur(10)))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if lowres and random.random() < 0.7:  # the finder's crops come from 25-85 px icons that get enlarged: imitate that
        small = random.randint(24, 64)
        img = img.resize((small, small), Image.BILINEAR).resize((size, size), random.choice([Image.BILINEAR, Image.BICUBIC]))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
    arr = np.asarray(img).astype(np.int16) + np.random.randint(-10, 11, (size, size, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(35, 90))  # compression artifacts, like a phone photo
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def generate(out_dir, per_class, icons=None, bg_dir=None, size=SIZE, seed=0,
             hard_negatives=False, lowres=False, centered=False, none_factor=1.0,
             color_jitter=False, bright_bg=False, labels=False):
    random.seed(seed); np.random.seed(seed)
    OPTIONS["color_jitter"], OPTIONS["bright_bg"] = color_jitter, bright_bg
    out = Path(out_dir)
    bg_files = sorted(p for p in Path(bg_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if bg_dir else []
    icon_set = load_icons(icons) if icons else {n: ([mask_for(fn)], c) for n, (fn, c) in CLASSES.items()}
    extra = load_extra_icons(icons) if hard_negatives else []
    if hard_negatives:
        print(f"hard negatives on; {len(extra)} extra 'none' icon(s) found in icons/none")
    counts = {}
    for name, (masks, color) in icon_set.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            if labels and name in LABELS and random.random() < 0.5:
                img = place_icon(background(size, bg_files), random.choice(masks), color, random.uniform(0.75, 1.0), size, centered, shrink=0.8, lift=size // 10)
                img = add_label(img, LABELS[name], color)
            else:
                img = place_icon(background(size, bg_files), random.choice(masks), color, random.uniform(0.75, 1.0), size, centered)
            degrade(img, size, lowres).save(out / name / f"{i:05d}.jpg")
        counts[name] = per_class
    (out / "none").mkdir(parents=True, exist_ok=True)
    n_none = round(per_class * none_factor)
    for i in range(n_none):  # "none": empty clusters, UNLIT icons (very dim), and bright decoys
        img = background(size, bg_files)
        if hard_negatives:
            img = hard_negative(img, extra, icon_set, centered)
        else:
            r = random.random()
            if r < 0.4:
                masks, color = random.choice(list(icon_set.values()))
                img = place_icon(img, random.choice(masks), color, random.uniform(0.04, 0.12), size)
            elif r < 0.75:
                img = add_decoys(img)
        degrade(img, size, lowres).save(out / "none" / f"{i:05d}.jpg")
    counts["none"] = n_none
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("per_class", type=int)
    ap.add_argument("--icons"); ap.add_argument("--bg"); ap.add_argument("--size", type=int, default=SIZE); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-negatives", action="store_true"); ap.add_argument("--lowres", action="store_true")
    ap.add_argument("--centered", action="store_true"); ap.add_argument("--none-factor", type=float, default=1.0)
    ap.add_argument("--color-jitter", action="store_true"); ap.add_argument("--bright-bg", action="store_true"); ap.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    print(generate(a.out_dir, a.per_class, a.icons, a.bg, a.size, a.seed, a.hard_negatives, a.lowres, a.centered, a.none_factor,
                   a.color_jitter, a.bright_bg, a.labels))