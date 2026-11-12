"""Open a photo the way the app should: any common format, upright, plain RGB, and a sane size.

Phones store a rotation flag (EXIF) instead of rotating the pixels, and iPhones save HEIC by default.
Every part of the project (predict.py, save_crops.py, the future web app) should open photos through load_photo().
"""
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener   # optional: pip install pillow-heif  (lets Pillow open iPhone HEIC photos)
    register_heif_opener()
except ImportError:
    pass

MAX_PIXELS = 100_000_000                           # refuse absurdly large images (a 12-megapixel photo is 12,000,000)
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP", "HEIF"}   # MPO is how some phones label JPEGs


def load_photo(source):
    """source: a file path or the raw bytes of an upload. Returns an upright RGB image, or raises ValueError."""
    try:
        img = Image.open(BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    except (UnidentifiedImageError, OSError):
        raise ValueError("That file is not an image I can read. Try a JPEG or PNG (iPhone HEIC needs: pip install pillow-heif).")
    if img.format not in ALLOWED_FORMATS:               # judged by the file's contents, not its name
        raise ValueError(f"Unsupported image type: {img.format}")
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("That image is too large.")
    img = ImageOps.exif_transpose(img)                  # apply the phone's rotation flag
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")                      # put transparent areas on black, like a dark dashboard
        flat = Image.new("RGB", rgba.size, (0, 0, 0))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return img.convert("RGB")                           # also drops metadata such as GPS location