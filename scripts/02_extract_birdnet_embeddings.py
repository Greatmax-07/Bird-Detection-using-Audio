"""
02_extract_birdnet_embeddings.py
==================================
Replace "train a CNN from scratch on 5,117 chunks" with "extract embeddings
from a model already pretrained on global bird calls, then train a small
head on top." With only ~614 recordings and several classes under 20
chunks, transfer learning is the highest-leverage change you can make.

Uses BirdNET-Analyzer (via the `birdnetlib` wrapper) as a frozen feature
extractor. BirdNET was trained on millions of bird recordings worldwide
(including plenty of South Asian species), so its embeddings already
encode useful acoustic structure even for species/classes with very
little data of your own.

Install:
    pip install birdnetlib librosa soundfile numpy tqdm
    # birdnetlib also needs ffmpeg on your system PATH

Expected input layout (works for both your existing processed dataset
AND the new xc_downloads/ folder from script 01 — just point --data_dirs
at both):
    some_root/
        Puff-throated_Babbler/
            rec1.wav
            rec2.mp3
        Blue-winged_Leafbird/
            xc_123456.mp3
        ...

Usage:
    python 02_extract_birdnet_embeddings.py \
        --data_dirs ./ibc53_processed ./xc_downloads \
        --out_dir ./features

Output:
    features/X.npy            -- (N, 1024) BirdNET embedding vectors
    features/y.npy            -- (N,) integer labels
    features/label_map.json   -- {class_name: integer_id}
    features/manifest.csv     -- source file + time offset for each row (for debugging)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def collect_files(data_dirs):
    """Returns list of (filepath, class_name) pairs, one per audio file."""
    items = []
    for root in data_dirs:
        root = Path(root)
        if not root.exists():
            print(f"  ! warning: {root} does not exist, skipping")
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            class_name = class_dir.name
            for f in class_dir.iterdir():
                if f.suffix.lower() in AUDIO_EXTS:
                    items.append((f, class_name))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True,
                     help="one or more dataset roots, each containing per-species subfolders")
    ap.add_argument("--out_dir", default="./features")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect_files(args.data_dirs)
    if not items:
        raise SystemExit("No audio files found under the given --data_dirs.")

    classes = sorted({c for _, c in items})
    label_map = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(items)} audio files across {len(classes)} classes.")

    analyzer = Analyzer()  # loads the BirdNET-Analyzer model once, reused for every file

    X, y, manifest_rows = [], [], []
    failures = 0

    for filepath, class_name in tqdm(items, desc="Extracting embeddings"):
        try:
            recording = Recording(analyzer, str(filepath))
            recording.extract_embeddings()
        except Exception as e:  # noqa: BLE001 - keep going on a single bad file
            failures += 1
            tqdm.write(f"  ! failed on {filepath}: {e}")
            continue

        for seg in recording.embeddings:
            vec = seg.get("embeddings") or seg.get("embedding")
            if vec is None:
                continue
            X.append(vec)
            y.append(label_map[class_name])
            manifest_rows.append([
                str(filepath), class_name,
                seg.get("start_time"), seg.get("end_time"),
            ])

    if not X:
        raise SystemExit("No embeddings were extracted — check that ffmpeg/librosa are installed correctly.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class", "start_time", "end_time"])
        writer.writerows(manifest_rows)

    print(f"\nDone. {X.shape[0]} embedding vectors (dim={X.shape[1]}) across {len(classes)} classes.")
    print(f"Failed to process {failures} files.")
    print("\nPer-class embedding counts:")
    counts = np.bincount(y, minlength=len(classes))
    for c, n in sorted(zip(classes, counts), key=lambda t: t[1]):
        print(f"  {c:35s} {n}")


if __name__ == "__main__":
    main()
