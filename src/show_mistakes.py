"""Show the real crops a model gets wrong, as one picture: each tile has the crop and "true -> guessed (confidence)".

Usage:  python src/show_mistakes.py models/v2 real_dev ~/Desktop/mistakes.png
Tiles are grouped by what the model GUESSED, so you can see what kinds of pictures end up under each wrong answer.
"""
import sys

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import datasets

from common import eval_transform, get_device, load_model

model_dir, test_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
device = get_device()
model, classes, cfg = load_model(model_dir, device)
prepare = eval_transform(cfg["img"])
ds = datasets.ImageFolder(test_dir, allow_empty=True)        # no transform here: we keep the original pictures to draw them

wrong = []
with torch.no_grad():
    for path, folder_index in ds.samples:
        true = ds.classes[folder_index]
        img = Image.open(path).convert("RGB")
        probs = torch.softmax(model(prepare(img).unsqueeze(0).to(device)), dim=1)[0].cpu()
        best = int(probs.argmax())
        if classes[best] != true:
            wrong.append((classes[best], true, float(probs[best]), img))
wrong.sort(key=lambda w: (w[0], w[1]))                       # group by the wrong answer

TILE, CAPTION, COLS = 130, 26, 7
rows = -(-len(wrong) // COLS)
sheet = Image.new("RGB", (COLS * TILE, max(rows, 1) * (TILE + CAPTION)), (25, 25, 25))
pen = ImageDraw.Draw(sheet)
for n, (guess, true, conf, img) in enumerate(wrong):
    x, y = (n % COLS) * TILE, (n // COLS) * (TILE + CAPTION)
    sheet.paste(ImageOps.contain(img, (TILE - 4, TILE - 4)), (x + 2, y + 2))
    pen.text((x + 2, y + TILE), f"true: {true}", fill=(255, 255, 255))
    pen.text((x + 2, y + TILE + 11), f"said: {guess} {conf:.0%}", fill=(255, 200, 80))
sheet.save(out)
print(f"{len(wrong)} wrong out of {len(ds.samples)} -> {out}")
