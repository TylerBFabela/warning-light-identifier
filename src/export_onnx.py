"""Export a trained model to ONNX so it can run in a web browser (or anywhere, without PyTorch).

Usage:  python src/export_onnx.py models/v3 docs/model
Writes model.onnx, classes.json and config.json into the output folder, then checks that the ONNX model gives the
same answers as the PyTorch model.  Needs:  pip install onnx onnxruntime
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from common import load_model

source, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
model, classes, cfg = load_model(source, "cpu")
size = cfg["img"]
dummy = torch.zeros(1, 3, size, size)
options = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(model, dummy, out / "model.onnx", dynamo=False, **options)   # the older, simpler exporter
except TypeError:                                                                  # older PyTorch has no dynamo option
    torch.onnx.export(model, dummy, out / "model.onnx", **options)
shutil.copy(source / "classes.json", out / "classes.json")
(out / "config.json").write_text(json.dumps({"arch": cfg["arch"], "img": size, "threshold": cfg.get("threshold", 0.8)}, indent=2))
print(f"wrote {out / 'model.onnx'} ({(out / 'model.onnx').stat().st_size / 1024:.0f} KB)")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("could not check the export (pip install onnxruntime), but the files were written")
x = torch.randn(8, 3, size, size)
with torch.no_grad():
    expected = model(x).numpy()
got = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
print(f"largest difference between PyTorch and ONNX answers: {np.abs(expected - got).max():.2e}  (should be tiny, under 1e-4)")