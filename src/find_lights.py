"""Find lit warning lights in a dashboard photo so each one can be classified separately.

A lit light is a bright, colorful blob on a dark background. This is a simple starting heuristic;
a trained detector (e.g. on Roboflow boxes) can replace it later. Run `python find_lights.py` for a demo.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _merge(boxes, gap, cap=None):
    """Merge boxes that overlap or sit within `gap` pixels of each other (parts of one icon).
    With `cap`, two boxes are not merged if the result would be wider or taller than `cap` pixels (too big for one light)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    union = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    if cap is not None and max(union[2] - union[0], union[3] - union[1]) > cap:
                        continue                                          # would be too big for one light: keep them apart
                    boxes[i] = union
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def propose_lights(img, min_area=25, margin=0.25, gap_frac=0.005, sat=0.5, val=0.45,
                   max_side_frac=None, sparse_frac=0.10, merge_cap_frac=None, white_core=False):
    """Return SQUARE boxes (x0, y0, x1, y1) around bright, saturated blobs (one box per icon).
    Boxes can extend beyond the photo; crop with img.crop(box), which pads with black. Squares avoid stretching.
    The options after val are off by default (the original behaviour); fractions are of the photo's shorter side:
      max_side_frac   drop a blob wider or taller than this (gauge backlight, big arcs, skin)
      sparse_frac     drop a blob bigger than this that fills less than 15% of its box (needles, scale arcs)
      merge_cap_frac  never merge pieces into a box bigger than this (stops neighbouring lights chaining together)
      white_core      count a white, over-exposed center as part of the colored light around it"""
    hsv = np.asarray(img.convert("HSV")).astype(np.float32) / 255
    mask = (hsv[..., 1] > sat) & (hsv[..., 2] > val)     # colorful and bright
    if white_core:
        white = (hsv[..., 2] > 0.85) & ~mask
        mask = ndimage.binary_propagation(mask, mask=mask | white)   # white pixels joined to colored ones
    labels, _ = ndimage.label(mask)
    short = min(img.size)
    raw = []
    for n, (ys, xs) in enumerate(ndimage.find_objects(labels), start=1):
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if h * w < min_area:
            continue
        if max_side_frac is not None and max(h, w) > max_side_frac * short:
            continue
        if sparse_frac is not None and max(h, w) > sparse_frac * short and (labels[ys, xs] == n).mean() < 0.15:
            continue
        raw.append((xs.start, ys.start, xs.stop, ys.stop))
    cap = merge_cap_frac * short if merge_cap_frac is not None else None
    boxes = []
    for x0, y0, x1, y1 in _merge(raw, gap=int(gap_frac * short), cap=cap):
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