"""The whole job in one function: photo -> find each lit light -> label it, or say "unsure".

The web app and the tests call analyze(); predict.py does the same steps for the command line.
"""
from dataclasses import dataclass

import torch
from PIL import ImageDraw

from common import eval_transform
from find_lights import propose_lights

MAX_LIGHTS = 40          # more candidate boxes than this means the photo is probably not a clean shot of a dashboard


@dataclass
class Light:
    box: tuple           # (x0, y0, x1, y1) in the ORIGINAL photo's pixels
    status: str          # "identified" or "unsure"
    label: str           # the class name when identified, otherwise "unsure"
    best_guess: str      # the model's top class even when unsure (it can be "none")
    confidence: float    # probability of that top class, 0..1


def analyze(photo, model, classes, cfg, device="cpu", threshold=None, detect_side=1024):
    """photo: an upright RGB PIL image. Returns a list of Light, top to bottom then left to right.

    A light is "identified" only if the model is confident (>= threshold) AND its top class is a real light.
    Everything else stays in the list as "unsure": a lit spot we could not name is never silently dropped."""
    threshold = cfg.get("threshold", 0.8) if threshold is None else threshold
    prepare = eval_transform(cfg["img"])
    small = photo.copy()
    small.thumbnail((detect_side, detect_side))                       # find the lights on a small copy (fast) ...
    scale = photo.width / small.width
    boxes = sorted(propose_lights(small), key=lambda b: (b[1] // 40, b[0]))   # rows about 40 px tall, left to right
    if len(boxes) > MAX_LIGHTS:
        raise ValueError(f"Found {len(boxes)} bright spots, more than a dashboard normally shows. "
                         "Try a straighter, closer photo of just the instrument cluster.")
    if not boxes:
        return []
    boxes = [tuple(round(c * scale) for c in b) for b in boxes]       # back to the original photo's pixels
    crops = [photo.crop(b) for b in boxes]                            # ... but cut the crops from the original (more detail)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.stack([prepare(c) for c in crops]).to(device)), dim=1).cpu()
    lights = []
    for box, p in zip(boxes, probs):
        confidence, index = p.max(dim=0)
        guess = classes[int(index)]
        sure = float(confidence) >= threshold and guess != "none"
        lights.append(Light(box, "identified" if sure else "unsure", guess if sure else "unsure", guess, float(confidence)))
    return lights


def annotate(photo, lights, max_side=1024):
    """A shrunken copy of the photo with a numbered box on every light (green = identified, orange = unsure)."""
    img = photo.copy()
    img.thumbnail((max_side, max_side))
    k = img.width / photo.width
    pen = ImageDraw.Draw(img)
    for number, light in enumerate(lights, start=1):
        color = (60, 200, 90) if light.status == "identified" else (255, 170, 0)
        box = [c * k for c in light.box]
        pen.rectangle(box, outline=color, width=3)
        pen.text((box[0] + 4, box[1] + 3), str(number), fill=color)
    return img