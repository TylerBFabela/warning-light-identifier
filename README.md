# Dashboard warning-light identifier

Photograph a car dashboard, find each lit warning light, and identify it, with an honest "unsure" when the model
is not confident.

## Status: work in progress
- Early scaffold. The synthetic-image generator, light finder and a timing script are in place.
- Training, evaluation and prediction scripts are coming next.
- There are NO real-photo results yet; numbers will be published here with the sample size.
- Not safety advice. When a warning light is on and you are unsure, consult your owner's manual or a mechanic.

## Quick start
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python make_synth.py data 500          # placeholder shapes, no icon files needed
    python time_loader.py data             # is loading images or training the slow part?
    python find_lights.py                  # demo: finds 3 lights on a fake cluster

## Design decisions (so far)
- PyTorch, 96x96 RGB input, no horizontal flips and no hue shifts (arrows have direction; color encodes severity).
- Synthetic images for training; real crops for validation and a final test; never tune on the test set.
- Data and model weights are NOT stored in this repo; see .gitignore.
