"""Find lit warning lights in a dashboard photo so each one can be classified separately.

A lit light is a bright, colorful blob on a dark background. This is a simple starting heuristic;
a trained detector (e.g. on Roboflow boxes) can replace it later. Run `python find_lights.py` for a demo.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _merge(boxes, gap):
    """Merge boxes that overlap or sit within `gap` pixels of each other (parts of one icon)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def propose_lights(img, min_area=60, margin=0.25, gap_frac=0.08):
    """Return SQUARE boxes (x0, y0, x1, y1) around bright, saturated blobs (one box per icon).
    Boxes can extend beyond the photo; crop with img.crop(box), which pads with black. Squares avoid stretching."""
    hsv = np.asarray(img.convert("HSV")).astype(np.float32) / 255
    mask = (hsv[..., 1] > 0.5) & (hsv[..., 2] > 0.45)   # colorful and bright
    labels, _ = ndimage.label(mask)
    raw = []
    for ys, xs in ndimage.find_objects(labels):
        if (ys.stop - ys.start) * (xs.stop - xs.start) >= min_area:
            raw.append((xs.start, ys.start, xs.stop, ys.stop))
    boxes = []
    for x0, y0, x1, y1 in _merge(raw, gap=int(gap_frac * min(img.size))):
        side = int(max(x1 - x0, y1 - y0) * (1 + 2 * margin))   # square, with a margin around the icon
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        sx, sy = cx - side // 2, cy - side // 2
        boxes.append((sx, sy, sx + side, sy + side))   # may extend past the photo edge: img.crop() pads with black, keeping crops square
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


if __name__ == "__main__":
    import make_synth as ms

    cluster = Image.new("RGB", (480, 150), (20, 20, 26))
    truth = {}
    for name, x, color in [("battery", 20, "red"), ("oil_pressure", 170, "red"), ("tire_pressure", 320, "amber")]:
        fn = ms.CLASSES[name][0]
        mask = ms.mask_for(fn).resize((110, 110))
        cluster.paste(Image.new("RGB", (110, 110), ms.COLORS[color]), (x, 20), mask)
        truth[name] = (x, 20, x + 110, 130)
    ImageDraw.Draw(cluster).rectangle([440, 60, 470, 64], fill=(240, 240, 240))  # white speck: should be ignored

    boxes = propose_lights(cluster)
    print("lights found:", len(boxes), "| expected: 3")
    for name, tb in truth.items():
        best = max((iou(tb, b) for b in boxes), default=0)
        print(f"  {name:14s} best overlap with a proposed box: {best:.2f}")
    out = cluster.copy(); d = ImageDraw.Draw(out)
    for b in boxes:
        d.rectangle(b, outline=(0, 255, 0), width=2)
    out.save("clusters_demo.png")
    print("saved clusters_demo.png")