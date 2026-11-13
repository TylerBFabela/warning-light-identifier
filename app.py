"""Web service: upload a dashboard photo, get back every lit warning light with a label or "unsure".

Run it locally:  uvicorn app:app --port 8000        then open http://localhost:8000
The API docs page is at /docs. The model folder comes from the MODEL_DIR environment variable (default: deploy_model).
Photos are processed in memory and are not saved by this app.
"""
import base64
import io
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402  (after the sys.path line on purpose)
from fastapi.responses import FileResponse  # noqa: E402

from common import load_model  # noqa: E402
from load_photo import load_photo  # noqa: E402
from pipeline import analyze, annotate  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024                         # refuse uploads bigger than 10 MB
NOTE = ("General information only, not a diagnosis. Your owner's manual is the authority for your car. "
        "If a warning light is on and you are unsure, have the car checked by a mechanic.")

app = FastAPI(title="Warning-light identifier")


@lru_cache(maxsize=1)
def get_model():
    """Load the model once, on first use, then reuse it for every request."""
    return load_model(os.environ.get("MODEL_DIR", str(ROOT / "deploy_model")), "cpu")


@lru_cache(maxsize=1)
def get_meanings():
    return json.loads((ROOT / "meanings.json").read_text())


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    _, classes, cfg = get_model()
    return {"status": "ok", "classes": classes, "arch": cfg["arch"], "threshold": cfg["threshold"]}


@app.post("/analyze")
def analyze_photo(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)             # read one byte more than allowed, to know if it is too big
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "That file is larger than 10 MB. Please send a smaller photo.")
    try:
        photo = load_photo(data)                     # fixes rotation, handles HEIC/PNG/WebP, refuses non-images
        model, classes, cfg = get_model()
        lights = analyze(photo, model, classes, cfg)
    except ValueError as err:                        # the friendly messages written in load_photo.py and pipeline.py
        raise HTTPException(400, str(err))
    meanings = get_meanings()
    buffer = io.BytesIO()
    annotate(photo, lights).save(buffer, "JPEG", quality=85)
    return {
        "lights": [{"id": n, "box": list(light.box), "status": light.status, "label": light.label,
                    "best_guess": light.best_guess, "confidence": round(light.confidence, 3),
                    "meaning": meanings.get(light.label) if light.status == "identified" else None}
                   for n, light in enumerate(lights, start=1)],
        "image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "threshold": cfg["threshold"],
        "note": NOTE,
    }