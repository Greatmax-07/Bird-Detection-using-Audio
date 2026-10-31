"""
inspect_clip.py - per-3s-segment scores for ONE audio file (debugging).

    python scripts/inspect_clip.py path/to/audio.mp3 [--top 4]

For each segment prints the top species by blended score, with the head and
BirdNET parts, and marks whether it passes t_low / t_high.
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdear.core import (  # noqa: E402
    FEATURES_DIR, analyze_audio, birdnet_label_index, class_names_from_label_map, l2_normalize,
    load_analyzer, load_label_map,
)
from birdear.detect import DetectParams, detect_events, fuse  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()

    P = DetectParams.from_env()
    names = class_names_from_label_map(load_label_map())
    head = joblib.load(FEATURES_DIR / "segment_head.pkl")
    analyzer = load_analyzer()
    label_idx = birdnet_label_index(analyzer, names)
    has_bn = label_idx >= 0
    times, emb, bn = analyze_audio(analyzer, args.audio, label_idx)
    ph = head.predict_proba(l2_normalize(emb, axis=1))
    fused = fuse(ph, bn, has_bn, P.w_birdnet)

    print(f"\n{P}\n")
    print("segment      species (fused = head/birdnet)")
    for s in range(len(times)):
        order = np.argsort(fused[s])[::-1][: args.top]
        cells = "  ".join(f"{names[k].replace('_', ' ')} {fused[s, k]:.2f} ({ph[s, k]:.2f}/{bn[s, k]:.2f})"
                          for k in order)
        print(f"{times[s, 0]:5.0f}-{times[s, 1]:4.0f}s  {cells}")
    print("\ndetected species:")
    for k, events in detect_events(times, fused, P).items():
        print(f"  {names[k].replace('_', ' ')}: " +
              ", ".join(f"{e['start_time']:.0f}-{e['end_time']:.0f}s ({e['confidence']:.2f})" for e in events))
    if not detect_events(times, fused, P):
        print("  none (no confident bird)")


if __name__ == "__main__":
    main()
