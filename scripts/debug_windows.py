"""
debug_windows.py - show what the window model thinks for every window of one file.

    python scripts/debug_windows.py path/to/audio.wav [--top 3] [--threshold 0.20]

Prints, per window: time span, top-N species with probability, and whether the
top one would pass the serving threshold. Use it to see WHY a stretch of audio
is not detected (probability spread over many species, wrong species, or below
threshold).
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdear.core import (  # noqa: E402
    FEATURES_DIR, class_names_from_label_map, extract_segments, load_analyzer,
    load_label_map, pool_segments, read_meta, window_ranges,
)

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.20)
    args = ap.parse_args()

    names = class_names_from_label_map(load_label_map())
    model = joblib.load(FEATURES_DIR / "window_svm.pkl")
    w = read_meta()["window"]
    segs = extract_segments(load_analyzer(), args.audio)
    emb = np.stack([s[2] for s in segs])
    ranges = window_ranges(len(emb), w["window_size"], w["step_size"])
    probs = model.predict_proba(np.stack([pool_segments(emb[a:b]) for a, b in ranges]))

    print(f"{len(segs)} segments, {len(ranges)} windows (threshold {args.threshold})\n")
    for (a, b), p in zip(ranges, probs):
        order = np.argsort(p)[::-1][: args.top]
        cells = "  ".join(f"{names[model.classes_[i]].replace('_', ' ')} {p[i]:.2f}" for i in order)
        flag = "DETECT" if p[order[0]] >= args.threshold else "  --  "
        print(f"{segs[a][0]:6.1f}-{segs[b - 1][1]:6.1f}s  {flag}  {cells}")


if __name__ == "__main__":
    main()
