"""Convert downloaded SVG icons to PNG.  Usage: python svg_to_png.py icons_raw icons
Reads icons_raw/<class>/*.svg and writes icons/<class>/<name>.png (512 px wide, keeps transparency).
Removes the gray (#999) corner brackets that ISO 7000 files include as registration marks."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf  # pip install pymupdf

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for svg in sorted(src.glob("*/*.svg")):
    root = ET.fromstring(svg.read_bytes())
    parents = {child: parent for parent in root.iter() for child in parent}
    for el in list(root.iter()):
        if el.get("stroke") == "#999" and el in parents:      # the corner marks
            parents[el].remove(el)
    page = pymupdf.open(stream=ET.tostring(root), filetype="svg")[0]
    zoom = 512 / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
    out_dir = dst / svg.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pix.save(out_dir / (svg.stem + ".png"))
    print(f"{svg.parent.name}/{svg.name} -> {pix.width}x{pix.height}")