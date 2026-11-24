# Telltale: dashboard warning light identifier

**[Try it live](https://tylerbfabela.github.io/warning-light-identifier/)**: take a photo of a car's instrument cluster, and
Telltale finds each lit warning light, names the ones it knows, and says how urgent each one is. Everything runs in the
browser: the photo never leaves the device.

It is a learning project, **not a safety tool**. When a warning light is on and you are unsure, check your owner's manual or
ask a mechanic.

## Why
Many drivers never read their owner's manual, and many dashboard symbols look alike. A light's color says how urgent it is,
but only the symbol says what is wrong, and a symbol is hard to describe to a search engine. Telltale lets you search with a
photo instead, and it says so when it is not sure.

## How it works
1. **Find** (`src/find_lights.py`, ported to `docs/app.js`): bright, colorful spots are grouped into one box per icon. Big,
   thin shapes (needles, gauge arcs) are dropped, and each box is made square with a margin around the icon.
2. **Crop**: each box is cut from the full-resolution photo and resized to 96×96 pixels.
3. **Classify**: MobileNetV3-small, pretrained on ImageNet and fine-tuned here, scores 12 light types plus "none" (not one
   of the 12).
4. **Decide**: 1.0 is first subtracted from the "none" score, so real lights are not dismissed too readily. Then:
   - **identified**: a light type at 90% confidence or more. Only these get advice.
   - **possible**: 40-90%, shown as a low-confidence guess without advice.
   - **unsure** or **probably not a warning light**: everything else.
5. **Draw a box**: if the finder misses a light (for example in bright daylight), the visitor can draw a box around it, and
   only that area is classified.

The model runs in the browser with ONNX Runtime Web, so the site is a static page on GitHub Pages, with no server.

## Results
Measured on development sets: 113 real lights and 134 spots that are not warning lights, cropped from a public Roboflow
dataset and from photos of two cars (day and night). The settings were chosen on these same sets, so the numbers are
optimistic; a fresh test set is the next step. Every experiment is in [EXPERIMENTS.md](EXPERIMENTS.md).

| With the site's settings (model v5b) | Count |
|---|---|
| Real lights named correctly | 87 of 113 (77%) |
| ... of those, shown as "identified" | 39, none of them wrong |
| Non-lights shown as a light | 10 of 134, all as a low-confidence "possible" |

How the model improved (real lights named correctly, before the "none" adjustment, average of 3 training runs):

| Version | Change | Lights named |
|---|---|---|
| v3 | small CNN trained from scratch on synthetic images | 39 of 114 (one run) |
| v4 | MobileNetV3-small, pretrained on ImageNet | 56.3 of 114 |
| v5a | solid and outline versions of each drawing | 68.0 of 113 |
| v5b | 14 more drawings, text-only BRAKE / ABS / PARK, side lights | 75.0 of 113 |

One crop was relabeled between v5a and v5b, so the total went from 114 to 113; v5a was re-scored on the corrected labels.

The light finder, on 16 Roboflow photos with hand-drawn boxes (53 lights):

| Finder | Clean box | Poor box (merged, too big or cut off) | Missed |
|---|---|---|---|
| first version | 19 | 29 | 5 |
| current | 31 | 15 | 7 |

## Decisions and why
- **Synthetic training data.** No free, licensed set of dashboard-icon crops with "not a light" examples existed, so
  `src/make_synth.py` draws licensed icons thousands of times: recolored within their color family, warped, blurred and
  glowing, on generated backgrounds, with phone-like noise and compression. Each option was added to fix a measured failure.
- **A "none" class with realistic examples.** The finder also boxes needles, gauge zones, displays, text and glare, so the
  model learns to answer "not one of mine".
- **Real photos for evaluation only**, split by source photo or by car, so near-identical crops never end up on both sides
  of a comparison.
- **A pretrained MobileNetV3-small.** The same network trained from a random start did no better than the small CNN;
  starting from ImageNet weights did. It is about 3.7 MB and fast in a browser.
- **Three training runs per version.** One run can be lucky, so versions are compared on the average.
- **No mirroring, and hues kept within each color family**: left and right arrows differ, and color means urgency.
- **A rule-based finder plus "Draw a box".** A trained detector needs box-labeled photos that may be used for training, and
  none were available. The rules are fast, explainable and measured.
- **Only "identified" lights get advice**: a refusal is safer than a confident wrong answer.
- **In the browser (ONNX + GitHub Pages)**: free to host, and the photo stays on the visitor's device. The JavaScript finder
  gives the same boxes as the Python one on all 36 test photos.

## Limitations
- Bright daylight washes out the lights' colors, so the finder can miss them.
- It knows 12 light types; any other light is shown as unsure.
- The circled "!" (brake system) and circled "P" (parking brake) are still the most common mix-up between two lights.
- Phone cameras shift colors at night (red lights can photograph as orange).
- The advice text in `docs/meanings.json` is general and is still being checked against owner's manuals.
- Tested on two cars and one public dataset.

## Repository
| Path | What it is |
|---|---|
| `docs/` | the website: `index.html`, `app.js`, the model (`docs/model/`), the advice text (`meanings.json`), sample photos |
| `src/make_synth.py`, `src/make_text_icons.py`, `src/svg_to_png.py` | synthetic training images, and preparing the icons for them |
| `src/train.py`, `src/common.py` | training, and the model definitions |
| `src/export_onnx.py` | converts a trained model for the browser and checks that it gives the same answers |
| `src/find_lights.py`, `src/pipeline.py`, `src/predict.py` | the finder and the photo-to-results pipeline in Python |
| `src/evaluate.py`, `src/probe_no_none.py`, `src/compare_models.py`, `src/show_mistakes.py` | testing on real crops |
| `src/finder_eval.py`, `src/finder_sweep.py`, `src/finder_compare.py` | measuring the light finder |
| `src/yolo_to_crops.py`, `mapping.json`, `LABELING.md` | turning a box-labeled dataset into labeled crops, and the labeling rules |
| `icons/classes.json` | the 12 light types and their colors |
| `EXPERIMENTS.md` | every experiment, with the numbers |

## Run it
The website, locally (then open http://localhost:8000):

```bash
python -m http.server 8000 --directory docs
```

The Python side. The icons are not in the repository; [ATTRIBUTION.md](ATTRIBUTION.md) lists the files.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/svg_to_png.py icons_raw icons            # downloaded SVG icons -> PNG
python src/make_text_icons.py icons                 # text-only BRAKE / ABS / PARK icons
python src/make_synth.py data_v5b 600 --icons icons --hard-negatives --none-factor 3 --lowres --centered --color-jitter --bright-bg --labels --styles --seed 2
python src/train.py data_v5b --arch mobilenet_v3_small --pretrained --epochs 12 --lr 1e-3 --seed 1 --out models/v5b_s1
python src/evaluate.py models/v5b_s1 real_dev       # a folder of your own labeled crops (see LABELING.md)
python src/export_onnx.py models/v5b_s1 docs/model  # the files the website loads; keeps the site's settings
python src/predict.py models/v5b_s1 photo.jpg       # the whole pipeline from the command line
```

## Licenses, credits and privacy
- Code: MIT (see [LICENSE](LICENSE)).
- Icons: Wikimedia Commons, credited in [ATTRIBUTION.md](ATTRIBUTION.md). The trained model in `docs/model/` is derived from
  CC BY-SA icons, so it is released under CC BY-SA 4.0.
- The text-only icons are drawn with Pillow's built-in font. The site's heading font is Barlow Semi Condensed (SIL Open Font
  License), loaded from Google Fonts. ONNX Runtime Web (MIT) is in `docs/ort/`.
- Photos: the author's own dashboard photos and crops are not in the repository, except the two sample photos in
  `docs/samples/`. The Roboflow crops were used for evaluation only and are not distributed.
- The website processes photos on the visitor's device only.

## Next steps
- A fresh test set, never used for any decision.
- Check the advice text against owner's manuals.
- More distinct drawings for the weakest classes (battery, oil pressure).
- A trained light finder, if box-labeled photos that allow training become available.
- Automated tests: the browser-against-Python finder check and the sample photos, run on every change.