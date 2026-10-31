"""
03_train_classifier_head.py
===========================
Recording-level species classifier: pooled (mean|max|std) BirdNET embedding of
the WHOLE recording -> RBF SVM. Answers "which species is this recording?".

    python scripts/03_train_classifier_head.py

Reads  features/{X,y}.npy, manifest.csv, label_map.json  (from script 02)
Writes features/recording_svm.pkl, recording_validation_predictions.csv,
       model_meta.json (section "recording")
Uses the SAME recording-level split as script 04 (birdear.core.recording_split).
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdear.core import (  # noqa: E402
    FEATURES_DIR, class_names_from_label_map, load_label_map, pool_segments,
    recording_split, update_meta,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=str(FEATURES_DIR))
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    X = np.load(fdir / "X.npy")
    y = np.load(fdir / "y.npy")
    manifest = pd.read_csv(fdir / "manifest.csv")
    names = class_names_from_label_map(load_label_map(fdir))
    if not (len(X) == len(y) == len(manifest)):
        raise SystemExit("X, y and manifest.csv lengths differ - rerun script 02")

    # one pooled vector per recording
    paths, labels, feats = [], [], []
    for path, idx in manifest.groupby("filepath", sort=True).indices.items():
        idx = idx[np.argsort(manifest["start_time"].to_numpy()[idx])]
        if len(set(y[idx])) != 1:
            raise SystemExit(f"recording with several labels: {path}")
        paths.append(path)
        labels.append(int(y[idx[0]]))
        feats.append(pool_segments(X[idx]))
    paths, labels, feats = np.array(paths, dtype=object), np.array(labels), np.stack(feats)
    print(f"{len(paths)} recordings, {len(names)} classes, feature dim {feats.shape[1]}")

    train, val = recording_split(paths, labels, args.test_frac, args.seed)
    tr = np.array([p in train for p in paths])
    va = np.array([p in val for p in paths])
    print(f"train {tr.sum()} / val {va.sum()} recordings")

    model = SVC(C=args.C, gamma=args.gamma, kernel="rbf", class_weight="balanced",
                probability=True, random_state=args.seed)
    model.fit(feats[tr], labels[tr])
    pred = model.predict(feats[va])

    acc = accuracy_score(labels[va], pred)
    macro = f1_score(labels[va], pred, average="macro", zero_division=0)
    weighted = f1_score(labels[va], pred, average="weighted", zero_division=0)
    print(classification_report(labels[va], pred, labels=np.arange(len(names)),
                                target_names=names, zero_division=0))
    print(f"accuracy {acc:.4f}  macro-F1 {macro:.4f}  weighted-F1 {weighted:.4f}")

    joblib.dump(model, fdir / "recording_svm.pkl")
    pd.DataFrame({
        "filepath": paths[va], "true_class": [names[i] for i in labels[va]],
        "predicted_class": [names[i] for i in pred],
    }).to_csv(fdir / "recording_validation_predictions.csv", index=False)
    update_meta("recording", {
        "n_classes": len(names), "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
        "C": args.C, "gamma": args.gamma, "seed": args.seed, "test_frac": args.test_frac,
        "sklearn_version": sklearn.__version__,
    }, fdir)
    print(f"saved {fdir / 'recording_svm.pkl'}")


if __name__ == "__main__":
    main()
