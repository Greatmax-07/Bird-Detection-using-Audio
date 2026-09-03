"""
preprocess.py
-------------
Converts raw bird audio recordings into mel spectrogram numpy arrays.

Usage:
    python preprocess.py --data_dir /path/to/iBC53 --out_dir ./processed

Output structure:
    processed/
        specs/          # .npy files, one per 5s chunk
        metadata.csv    # path, label, class_idx, split
        class_map.json  # {class_name: idx}
        stats.json      # mean/std per channel for normalization
"""

import os
import json
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

SAMPLE_RATE    = 22050      # Hz — standard for librosa / bird audio tasks
CHUNK_DURATION = 5.0        # seconds per chunk
HOP_DURATION   = 2.5        # overlap hop for Weak/Moderate classes; full hop for Strong
N_MELS         = 128
N_FFT          = 2048
HOP_LENGTH     = 512        # spectrogram hop (not chunk hop)
F_MIN          = 50         # Hz — below most bird calls
F_MAX          = 11025      # Hz — Nyquist at 22050 Hz

CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)

# Classes to exclude entirely
DROPPED_CLASSES = {
    "Mystery",
    "Black-browed_Reed_Warbler",
    "Asian_Emerald_Cuckoo",
    # Weak tier
    "White-tailed_Flycatcher",
    "Yellow-vented_Flowerpecker",
    "Grey-throated_Martin",
    "Cinnamon_Bittern",
    "Tickell's_Leaf_Warbler",
    "Blue-winged_Leafbird",
    # Moderate tier
    "Yellow-browed_Warbler",
    "Yellow-throated_Leaf_Warbler",
    "Asian_Palm_Swift",
    "Streaked_Spiderhunter",
    "Baikal_Bush_Warbler",
    "Scarlet-backed_Flowerpecker",
    "Chinspot_Wren-Babbler",
    "Long-tailed_Shrike",
    "Spot-breasted_Parrotbill",
    "Oriental_Dollarbird",
    "Mrs_Gould's_Sunbird",
    "Plain_Flowerpecker",
    "Black-bellied_Plover",
    "Ruddy_Kingfisher",
    "Cachar_Bulbul",
}

WEAK_CLASSES = set()       # empty — none left
MODERATE_CLASSES = set()   # empty — none left

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10 (remainder)

RANDOM_SEED = 42


# ── Audio helpers ───────────────────────────────────────────────────────────────

def load_audio(path: Path) -> np.ndarray | None:
    """Load audio file as mono, resampled to SAMPLE_RATE. Returns None on failure."""
    try:
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return y
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return None


def audio_to_melspec(y: np.ndarray) -> np.ndarray:
    """Convert audio array to log-mel spectrogram, shape (N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # → roughly -80..0 dB range
    return log_mel.astype(np.float32)


def chunk_audio(y: np.ndarray, hop_samples: int) -> list[np.ndarray]:
    """
    Slice audio into CHUNK_SAMPLES-long segments.
    - Short tail (<50% of chunk) is discarded.
    - Short tail (≥50% of chunk) is zero-padded to full length.
    """
    chunks = []
    start = 0
    while start + CHUNK_SAMPLES <= len(y):
        chunks.append(y[start : start + CHUNK_SAMPLES])
        start += hop_samples

    # Handle tail
    tail = y[start:]
    if len(tail) >= CHUNK_SAMPLES // 2:
        padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        padded[: len(tail)] = tail
        chunks.append(padded)

    return chunks


# ── Main preprocessing ──────────────────────────────────────────────────────────

def preprocess(data_dir: Path, out_dir: Path):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    specs_dir = out_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    # Discover classes
    class_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name not in DROPPED_CLASSES
    ])
    class_names = [d.name for d in class_dirs]
    class_map   = {name: idx for idx, name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes (after dropping {len(DROPPED_CLASSES)} excluded).")
    print(f"Output directory: {out_dir}\n")

    records = []  # list of dicts for metadata.csv

    for class_dir in class_dirs:
        label     = class_dir.name
        class_idx = class_map[label]

        if label in WEAK_CLASSES or label in MODERATE_CLASSES:
            hop_samples = int(SAMPLE_RATE * HOP_DURATION)   # 2.5s hop → 50% overlap
        else:
            hop_samples = CHUNK_SAMPLES                      # 5.0s hop → no overlap

        audio_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
        ])

        if not audio_files:
            print(f"  [SKIP] {label} — no audio files found")
            continue

        # Split files into train/val/test BEFORE chunking to prevent leakage
        random.shuffle(audio_files)
        n      = len(audio_files)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val   = max(1, int(n * VAL_RATIO))

        file_splits = {}
        for i, f in enumerate(audio_files):
            if i < n_train:
                file_splits[f] = "train"
            elif i < n_train + n_val:
                file_splits[f] = "val"
            else:
                file_splits[f] = "test"

        chunk_count = 0
        for audio_path in tqdm(audio_files, desc=f"{label:45s}", leave=False):
            y = load_audio(audio_path)
            if y is None or len(y) < SAMPLE_RATE:   # skip clips <1s
                continue

            split  = file_splits[audio_path]
            chunks = chunk_audio(y, hop_samples)

            for i, chunk in enumerate(chunks):
                spec     = audio_to_melspec(chunk)   # shape: (128, T)
                stem     = f"{label}__{audio_path.stem}__chunk{i:04d}"
                spec_path = specs_dir / f"{stem}.npy"
                np.save(str(spec_path), spec)

                records.append({
                    "path":      str(spec_path.relative_to(out_dir)),
                    "label":     label,
                    "class_idx": class_idx,
                    "split":     split,
                })
                chunk_count += 1

        print(f"  {label:45s} → {chunk_count:4d} chunks  (hop={'2.5s' if hop_samples < CHUNK_SAMPLES else '5.0s'})")

    # ── Save metadata ───────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metadata.csv", index=False)

    with open(out_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # ── Compute normalization stats (train split only) ──────────────────────────
    print("\nComputing normalization statistics (train split)...")
    train_paths = df[df["split"] == "train"]["path"].tolist()
    sample_paths = random.sample(train_paths, min(500, len(train_paths)))

    all_specs = [np.load(out_dir / p) for p in tqdm(sample_paths, desc="Loading sample specs")]
    stacked   = np.stack(all_specs)         # (N, 128, T)
    mean      = float(stacked.mean())
    std       = float(stacked.std())

    stats = {"mean": mean, "std": std}
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"  Total chunks:  {len(df)}")
    print(f"  Train chunks:  {len(df[df['split']=='train'])}")
    print(f"  Val chunks:    {len(df[df['split']=='val'])}")
    print(f"  Test chunks:   {len(df[df['split']=='test'])}")
    print(f"  Classes:       {df['label'].nunique()}")
    print(f"  Spec mean:     {mean:.4f}")
    print(f"  Spec std:      {std:.4f}")
    print(f"\n  Saved to: {out_dir}")

    # Per-class chunk counts
    print("\nPer-class chunk counts:")
    counts = df.groupby("label").size().sort_values(ascending=False)
    for label, count in counts.items():
        bar = "█" * (count // 20)
        print(f"  {label:45s} {count:4d}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess iBC53 bird audio dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to iBC53 root directory (contains one folder per species)")
    parser.add_argument("--out_dir",  type=str, default="./processed",
                        help="Output directory for spectrograms and metadata")
    args = parser.parse_args()

    preprocess(Path(args.data_dir), Path(args.out_dir))
