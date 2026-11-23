"""Try several finder settings on the same labeled photos and print one line per setting.

Usage:  python src/finder_sweep.py external_raw_rf2 [--splits valid,test]
Fractions are of the photo's shorter side. 'current' is what the website uses today.
"""
import argparse
from pathlib import Path

from finder_eval import evaluate

VARIANTS = {
    "current": {},
    "gap 0.005": {"gap_frac": 0.005},
    "gap 0.003": {"gap_frac": 0.003},
    "gap 0.005 + cap 0.15": {"gap_frac": 0.005, "merge_cap_frac": 0.15},
    "gap 0.005 + sparse 0.10": {"gap_frac": 0.005, "sparse_frac": 0.10},
    "gap 0.005 + white core": {"gap_frac": 0.005, "white_core": True},
    "gap 0.005 + sparse 0.10 + white": {"gap_frac": 0.005, "sparse_frac": 0.10, "white_core": True},
}

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--splits", default="valid,test,train")
a = ap.parse_args()
print(f"{'setting':34s} {'clean':>6s} {'poor':>6s} {'missed':>7s} {'boxes/photo':>12s}")
for name, params in VARIANTS.items():
    r = evaluate(Path(a.dataset), params, a.splits.split(","))
    print(f"{name:34s} {r['clean']:6d} {r['poor']:6d} {r['missed']:7d} {r['boxes'] / max(r['photos'], 1):12.1f}")