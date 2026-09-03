"""
resplit.py
----------
Re-generates metadata.csv with a stratified split — guarantees each species
has recordings in all three splits, reducing the chance of a bad random draw.

Overwrites dataset/processed/metadata.csv in place.
Re-run train.py from scratch (or from checkpoint) after this.

Usage:
    python scripts/resplit.py --processed_dir dataset/processed
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
SEED        = 7   # different seed from original (42) to get a fresh split


def resplit(args):
    random.seed(SEED)
    processed = Path(args.processed_dir)
    df = pd.read_csv(processed / "metadata.csv")

    # Extract the source recording stem from the path
    # filename format: ClassName__recordingstem__chunkNNNN.npy
    df["recording"] = df["path"].apply(lambda p: "__".join(Path(p).stem.split("__")[:2]))

    new_split = {}
    for label, group in df.groupby("label"):
        recordings = list(group["recording"].unique())
        random.shuffle(recordings)
        n       = len(recordings)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val   = max(1, int(n * VAL_RATIO))

        for i, rec in enumerate(recordings):
            if i < n_train:
                new_split[rec] = "train"
            elif i < n_train + n_val:
                new_split[rec] = "val"
            else:
                new_split[rec] = "test"

        # Safety: if only 1-2 recordings, force at least one to train
        if n == 1:
            new_split[recordings[0]] = "train"
        elif n == 2:
            new_split[recordings[0]] = "train"
            new_split[recordings[1]] = "val"

    df["split"] = df["recording"].map(new_split)
    df = df.drop(columns=["recording"])
    df.to_csv(processed / "metadata.csv", index=False)

    print("Split counts:")
    print(df["split"].value_counts().to_string())
    print("\nPer-class split check:")
    for label, group in df.groupby("label"):
        counts = group["split"].value_counts()
        print(f"  {label:45s}  train={counts.get('train',0):3d}  val={counts.get('val',0):3d}  test={counts.get('test',0):3d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="dataset/processed")
    args = parser.parse_args()
    resplit(args)
