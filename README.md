# Dashboard warning-light identifier

**[Live demo](https://tylerbfabela.github.io/warning-light-identifier/)**: upload a photo of a car's instrument cluster. It runs in your
browser, and your photo never leaves your device.

The app finds each lit warning light in a photo and names it, or says it is unsure. It is a portfolio prototype, **not
safety advice**: when a warning light is on and you are unsure, consult your owner's manual or a mechanic.

## Status: working end to end, accuracy on new dashboards is limited
The full pipeline works and is deployed. The model is the weak part, and I measured exactly how weak.

| Test set | What it is | Crops | Overall | Real lights right | Non-lights right |
|---|---|---|---|---|---|
| Mazda CX-30, day | used while tuning (optimistic) | 92 (34 lights, 58 non-lights) | 77% | 22 of 34 (65%) | 84% |
| Toyota RAV4, day | a different car, evaluated once | 50 (20 lights, 30 non-lights) | 70% | 6 of 20 (30%) | 97% |
| Both cars, night | newer photos | 54 (20 lights, 34 non-lights) | 61% | 3 of 20 (15%) | 88% |

Read the "real lights right" column, not "overall": non-lights are over half of every set. Crops were sorted by hand using
[LABELING.md](LABELING.md). Only two cars, and one class (parking brake) has 11 near-duplicate crops, so these are small
samples. Numbers are the model's top guess, before the confidence threshold.

**End to end on one fully labeled photo** (Mazda bulb check, 7 real lights, 26 boxes): the top guess was right for 4 of 7
(airbag, oil pressure, parking brake, check engine) but only at 40-79% confidence. ABS and tire pressure were called "not a
light" at 94-96%, and BRAKE was read as parking brake. At the app's 0.9 threshold it made two confident claims and both
were wrong: a clipped, pink fragment read as "parking brake" and clipped "HOLD" text read as "headlights".

The demo shows three tiers: *identified* (confidence 90% or more, with advice), *possible* (40-90%, a low-confidence guess
with no advice), and *unsure* (everything else, including "not a known light").

## Known limitations
- The light finder is a color-and-brightness rule. It misses washed-out or dim icons, merges icons into glow
  (the Toyota's orange backlight), boxes gauge needles and numbers, and sometimes cuts an icon in half. A clipped icon
  can be named confidently and wrongly.
- Trained only on synthetic images; tested on two cars, so new dashboards often fall outside what it has seen.
- 12 light types. Red and amber versions of a symbol are treated as different lights. The circled "!" (BRAKE) and circled
  "P" (PARK) are not told apart reliably, so the app gives the same advice for both.
- Phone cameras shift colors (red lights photograph orange at night), which hurts night photos.

## How it works
1. **Find** bright, colorful spots (`src/find_lights.py`, ported to JavaScript in `docs/app.js`).
2. **Crop** a square around each spot from the full-resolution photo.
3. **Classify** each crop with a small CNN (242k parameters, 96x96 input) into 12 light types or "none".
4. **Decide**: name the light only if the model is confident and the answer is a real light, otherwise "unsure".

The model is trained on synthetic images made from open-licensed icons (`src/make_synth.py`): icons are recolored, warped,
blurred and glowed on dashboard-like backgrounds, with extra "not a known light" examples (other symbols, gauge needles
and zones, orange displays, text, glare). Real photos are used only for validation and testing, split by car.

## What I tried (and what the evidence said)
| Step | Result on real Mazda crops |
|---|---|
| v1: plain synthetic icons | 18% overall; never said "none" (0 of 58) |
| v2: realistic negatives, 3x "none", low-res, centered icons | 41% |
| v3: + color jitter, pale glare backgrounds, printed BRAKE/PARK words | 77% (the deployed model) |
| Small CNN vs residual network | same accuracy on synthetic data, 3.3x slower, so small CNN |
| Wider color range + blown-out glow (small proxy set) | about +3 points, not adopted |
| Real crops from one car in training, tested on the other (small proxy set) | no gain (21% to 21%) |
| Finder fixes tried: pixel-count floor, local contrast, re-checking oversized blobs, top-hat filter | none shipped: each dropped real icons, left glow merged, or boxed gauge numbers |

What it points to: the bottleneck is how many different dashboards the model has seen, not tuning on these two.

## Decisions I can defend
- **PyTorch and a small CNN**: fits a laptop, trains in minutes, ResNet gave no gain.
- **Synthetic training data**: no free, licensed dataset of dashboard icons with "none" examples.
- **A "none" class plus a confidence threshold**: refusing to guess is safer than guessing.
- **Test split by car, not by photo**: crops from one car look alike, so mixing them would inflate results.
- **No horizontal flips** (left and right arrows differ); color jitter only within a light's color family.
- **Merged advice for BRAKE and PARK**: I could not tell them apart, and the safe advice is the same.
- **In-browser deployment (ONNX + GitHub Pages)**: free, permanent, no server, and photos stay on the user's device.

## Run it
In-browser demo, locally:

    python -m http.server 8000 --directory docs      # then open http://localhost:8000

Python side (needs the icons: see ATTRIBUTION.md, then `python src/svg_to_png.py icons_raw icons`):

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python src/make_synth.py data_v3 600 --icons icons --hard-negatives --none-factor 3 --lowres --centered --color-jitter --bright-bg --labels --seed 2
    python src/train.py data_v3 --out models/v3
    python src/evaluate.py models/v3 real_dev            # your own sorted crops, see LABELING.md
    python src/show_mistakes.py models/v3 real_dev mistakes.png
    python src/export_onnx.py models/v3 docs/model       # the file the web demo loads
    python src/predict.py models/v3 photo.jpg            # command-line version

There is also a FastAPI version (`uvicorn app:app`, with a `Dockerfile` that I have not built or tested yet).

## Data, licenses and privacy
Icons come from Wikimedia Commons: see [ATTRIBUTION.md](ATTRIBUTION.md). My dashboard photos and sorted crops are not
included. Code is MIT-licensed (see LICENSE). The trained model in `docs/model/` is derived from CC BY-SA icons and is
released under CC BY-SA 4.0 with the credits in ATTRIBUTION.md. The demo processes your photo on your device only.

## Next steps
Photograph many more cars; test a pretrained backbone; replace the finder with a trained detector; add more icon artwork
variants per class; add automated tests and CI.
