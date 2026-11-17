"""Split labeled external crops into a dev half and a final-test half, by SOURCE PHOTO (not by crop).

Usage:  python src/split_external.py external_rf2 external_dev external_test
File names look like  train_0012_<stem>_<n>.jpg : the 4-digit number identifies the source photo.
Crops cut from one photo always land on the same side, so the test half never contains 'the same photo again'.
Copies the files (the originals stay where they are). Refuses to run if the output folders already exist.
"""
import random
import shutil
import sys
from pathlib import Path

source_dir, dev_dir, test_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if dev_dir.exists() or test_dir.exists():
    sys.exit("The dev or test folder already exists. Remove it yourself first if you really want to redo the split.")

crops = sorted(source_dir.rglob("*.jpg"))


def photo_id(path):
    return path.name.split("_")[1]      # TODO 1: the 4-digit photo number from the file name (hint: path.name.split("_") cuts the name at each underscore)


photos = sorted({photo_id(p) for p in crops})
random.Random(0).shuffle(photos)                 # fixed seed: the same split every time you run it
dev_photos = set(photos[:len(photos) // 2])    # TODO 2: a set holding the first half of the shuffled photos (hint: slicing, len(photos) // 2)

counts = {"dev": {}, "test": {}}
for crop in crops:
    side, folder = ("dev", dev_dir) if photo_id(crop) in dev_photos else ("test", test_dir)
    target = folder / crop.parent.name / crop.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(crop, target)
    counts[side][crop.parent.name] = counts[side].get(crop.parent.name, 0) + 1

print(f"photos: {len(photos)}  ->  dev {len(dev_photos)}, test {len(photos) - len(dev_photos)}")
for side in ("dev", "test"):
    print(f"{side:5s}", dict(sorted(counts[side].items())), " total", sum(counts[side].values()))