"""
04_train_window_svm.py
======================
Window-level classifier for temporal detection: 5 consecutive 3 s BirdNET
segments (15 s), step 2 segments (6 s), pooled like script 03 -> RBF SVM.

WEAK SUPERVISION: every window inherits its recording's species label, even if
the bird is silent in that window. Treat outputs as "acoustic activity", not
ground truth.

    python scripts/04_train_window_svm.py

Writes features/window_svm.pkl, window_validation_predictions.csv,
       model_meta.json (section "window", incl. window size/step used by the server)
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
    FEATURES_DIR, WINDOW_SIZE, WINDOW_STEP, class_names_from_label_map,
    load_label_map, pool_segments, recording_split, update_meta, window_ranges,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=str(FEATURES_DIR))
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--step_size", type=int, default=WINDOW_STEP)
    ap.add_argument("--max_train_windows_per_recording", type=int, default=2)
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    X = np.load(fdir / "X.npy")
    y = np.load(fdir / "y.npy")
    manifest = pd.read_csv(fdir / "manifest.csv")
    names = class_names_from_label_map(load_label_map(fdir))
    if not (len(X) == len(y) == len(manifest)):
        raise SystemExit("X, y and manifest.csv lengths differ - rerun script 02")

    starts_all = manifest["start_time"].to_numpy(dtype=float)
    ends_all = manifest["end_time"].to_numpy(dtype=float)

    feats, labels, rec_of, t0, t1 = [], [], [], [], []
    rec_windows = {}  # path -> list of window row indices
    for path, idx in manifest.groupby("filepath", sort=True).indices.items():
        idx = idx[np.argsort(starts_all[idx])]
        if len(set(y[idx])) != 1:
            raise SystemExit(f"recording with several labels: {path}")
        for a, b in window_ranges(len(idx), args.window_size, args.step_size):
            rec_windows.setdefault(path, []).append(len(feats))
            feats.append(pool_segments(X[idx[a:b]]))
            labels.append(int(y[idx[0]]))
            rec_of.append(path)
            t0.append(starts_all[idx[a]])
            t1.append(ends_all[idx[b - 1]])
    feats, labels = np.stack(feats), np.array(labels)
    rec_of = np.array(rec_of, dtype=object)
    print(f"{len(rec_windows)} recordings -> {len(feats)} windows "
          f"(size {args.window_size}, step {args.step_size})")

    rec_paths = list(rec_windows)
    rec_labels = [labels[rec_windows[p][0]] for p in rec_paths]
    train, val = recording_split(rec_paths, rec_labels, args.test_frac, args.seed)

    train_idx = []
    for p in train:
        w = np.array(rec_windows[p])
        if len(w) > args.max_train_windows_per_recording:
            pos = np.unique(np.linspace(0, len(w) - 1, args.max_train_windows_per_recording, dtype=int))
            w = w[pos]
        train_idx.extend(w.tolist())
    val_idx = np.array([i for p in val for i in rec_windows[p]])
    train_idx = np.array(train_idx)
    print(f"train windows {len(train_idx)} / val windows {len(val_idx)}")

    model = SVC(C=args.C, gamma=args.gamma, kernel="rbf", class_weight="balanced",
                probability=True, random_state=args.seed)
    model.fit(feats[train_idx], labels[train_idx])
    pred = model.predict(feats[val_idx])

    acc = accuracy_score(labels[val_idx], pred)
    macro = f1_score(labels[val_idx], pred, average="macro", zero_division=0)
    weighted = f1_score(labels[val_idx], pred, average="weighted", zero_division=0)
    print(classification_report(labels[val_idx], pred, labels=np.arange(len(names)),
                                target_names=names, zero_division=0))
    print(f"accuracy {acc:.4f}  macro-F1 {macro:.4f}  weighted-F1 {weighted:.4f}")
    print("(accuracy is agreement with the INHERITED recording label, not true presence)")

    joblib.dump(model, fdir / "window_svm.pkl")
    pd.DataFrame({
        "filepath": rec_of[val_idx],
        "start_time": np.array(t0)[val_idx], "end_time": np.array(t1)[val_idx],
        "true_class": [names[i] for i in labels[val_idx]],
        "predicted_class": [names[i] for i in pred],
    }).to_csv(fdir / "window_validation_predictions.csv", index=False)
    update_meta("window", {
        "n_classes": len(names), "window_size": args.window_size, "step_size": args.step_size,
        "n_train_windows": int(len(train_idx)), "n_val_windows": int(len(val_idx)),
        "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
        "C": args.C, "gamma": args.gamma, "seed": args.seed, "test_frac": args.test_frac,
        "max_train_windows_per_recording": args.max_train_windows_per_recording,
        "sklearn_version": sklearn.__version__,
    }, fdir)
    print(f"saved {fdir / 'window_svm.pkl'}")


if __name__ == "__main__":
    main()
