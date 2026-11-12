"""Find each lit warning light in a dashboard photo and say what it is.

Usage:  python src/predict.py models/v1 my_dashboard.jpg [--threshold 0.8]
Steps: shrink the photo -> find bright colorful blobs (find_lights.py) -> crop each one -> classify -> draw the result.
If no blobs are found, the whole image is treated as one icon (for photos you already cropped).
Saves my_dashboard_pred.png next to the photo.
"""
import argparse

import torch
from PIL import Image, ImageDraw

from common import eval_transform, get_device, load_model
from find_lights import propose_lights
from load_photo import load_photo

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("photo")
ap.add_argument("--threshold", type=float)
a = ap.parse_args()

device = get_device()
model, classes, cfg = load_model(a.model_dir, device)
threshold = a.threshold if a.threshold is not None else cfg["threshold"]
prepare = eval_transform(cfg["img"])

full = load_photo(a.photo)                             # upright RGB, any common format
photo = full.copy()
photo.thumbnail((1024, 1024))                          # find the lights on a small copy (fast)...
scale = full.width / photo.width
boxes = propose_lights(photo) or [(0, 0, photo.width, photo.height)]
annotated = photo.copy()
pen = ImageDraw.Draw(annotated)

print(f"{len(boxes)} candidate light(s); answers below {threshold:.0%} confidence are shown as 'unsure'")
for number, box in enumerate(boxes, start=1):
    crop = full.crop(tuple(round(c * scale) for c in box))   # ...but cut the crop from the original photo (more detail)
    with torch.no_grad():
        probs = torch.softmax(model(prepare(crop).unsqueeze(0).to(device)), dim=1)[0].cpu()
    top = probs.topk(min(3, len(classes)))
    best_name, best_prob = classes[int(top.indices[0])], float(top.values[0])
    shown = best_name if (best_prob >= threshold and best_name != "none") else "unsure / not a known light"
    runners_up = ", ".join(f"{classes[int(i)]} {float(p):.0%}" for p, i in zip(top.values, top.indices))
    print(f"  #{number}: {shown} ({best_prob:.0%})   top 3: {runners_up}")
    pen.rectangle(box, outline=(0, 255, 0), width=2)
    pen.text((box[0] + 3, box[1] + 3), f"{number}: {shown}", fill=(0, 255, 0))

destination = a.photo.rsplit(".", 1)[0] + "_pred.png"
annotated.save(destination)
print("saved", destination)