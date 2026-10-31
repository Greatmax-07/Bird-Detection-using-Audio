"""
04_train_segment_head.py
========================
Per-3-second-segment head on BirdNET embeddings with INDEPENDENT sigmoid outputs
(one per species), so several species can score high in the same segment.
Replaces the old 15 s window SVM.

    python scripts/04_train_segment_head.py

Training data: every segment of the training recordings (capped per recording),
target = one-hot of the recording's species. Weak labels: a quiet segment still
carries its recording's label.

Writes features/segment_head.pkl and model_meta.json (section "segment").
Uses the same recording-level split as script 03.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdear.core import (  # noqa: E402
    FEATURES_DIR, class_names_from_label_map, l2_normalize, load_label_map, recording_split, update_meta,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=str(FEATURES_DIR))
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_segments_per_recording", type=int, default=40)
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    X = l2_normalize(np.load(fdir / "X.npy"), axis=1)
    y = np.load(fdir / "y.npy")
    manifest = pd.read_csv(fdir / "manifest.csv")
    names = class_names_from_label_map(load_label_map(fdir))
    C = len(names)
    if not (len(X) == len(y) == len(manifest)):
        raise SystemExit("X, y and manifest.csv lengths differ - rerun script 02")

    groups = manifest.groupby("filepath", sort=True).indices
    paths = np.array(list(groups), dtype=object)
    labels = np.array([int(y[groups[p][0]]) for p in paths])
    train, val = recording_split(paths, labels, args.test_frac, args.seed)

    rng = np.random.default_rng(args.seed)
    tr_idx, va_idx, va_rec = [], [], []
    for p in paths:
        idx = groups[p]
        if p in train:
            if len(idx) > args.max_segments_per_recording:
                idx = rng.choice(idx, args.max_segments_per_recording, replace=False)
            tr_idx.extend(idx.tolist())
        elif p in val:
            va_idx.extend(idx.tolist())
            va_rec.extend([p] * len(idx))
    tr_idx, va_idx, va_rec = np.array(tr_idx), np.array(va_idx), np.array(va_rec, dtype=object)
    print(f"{len(paths)} recordings -> train {len(tr_idx)} segments / val {len(va_idx)} segments, {C} classes")

    Y = np.zeros((len(tr_idx), C), dtype=np.int8)
    Y[np.arange(len(tr_idx)), y[tr_idx]] = 1
    model = MLPClassifier(hidden_layer_sizes=(args.hidden,), alpha=args.alpha, max_iter=args.max_iter,
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                          random_state=args.seed)
    model.fit(X[tr_idx], Y)
    print(f"stopped after {model.n_iter_} epochs")

    P = model.predict_proba(X[va_idx])
    pred = P.argmax(axis=1)
    seg_acc = float((pred == y[va_idx]).mean())
    macro = float(f1_score(y[va_idx], pred, average="macro", zero_division=0))
    rec_ok = []
    for p in val:
        m = va_rec == p
        rec_ok.append(int(P[m].mean(axis=0).argmax() == y[va_idx][m][0]))
    rec_acc = float(np.mean(rec_ok))
    print(f"segment top-1 accuracy {seg_acc:.4f}  macro-F1 {macro:.4f}  "
          f"recording accuracy (mean of segment scores) {rec_acc:.4f}")
    print("(segments inherit the recording label, so this understates the real segment accuracy)")

    joblib.dump(model, fdir / "segment_head.pkl")
    update_meta("segment", {
        "n_classes": C, "hidden": args.hidden, "alpha": args.alpha, "n_epochs": int(model.n_iter_),
        "n_train_segments": int(len(tr_idx)), "n_val_segments": int(len(va_idx)),
        "segment_accuracy": seg_acc, "segment_macro_f1": macro, "recording_accuracy": rec_acc,
        "seed": args.seed, "test_frac": args.test_frac,
        "max_segments_per_recording": args.max_segments_per_recording,
        "sklearn_version": sklearn.__version__,
    }, fdir)
    print(f"saved {fdir / 'segment_head.pkl'}")


if __name__ == "__main__":
    main()
