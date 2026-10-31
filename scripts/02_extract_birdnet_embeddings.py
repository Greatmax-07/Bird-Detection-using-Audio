"""
02_extract_birdnet_embeddings.py
================================
Run frozen BirdNET over every audio file and store one 1024-D embedding per
3-second segment.

Only folders named after species in config/species_maharashtra.txt are used
(override with --all_folders). Input layout (one folder per species, files directly inside):
    data/xc/<Species_Name>/xc_123456.mp3

    python scripts/02_extract_birdnet_embeddings.py                       # data/xc -> features/
    python scripts/02_extract_birdnet_embeddings.py --data_dirs data/xc data/my_recordings

Output in --out_dir (default features/):
    X.npy            (N, 1024) float32
    y.npy            (N,)      int64 class id per segment
    label_map.json   {class_name: id}   (ids = alphabetical order of folder names)
    manifest.csv     filepath,class,start_time,end_time per row of X
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdear.core import (  # noqa: E402
    AUDIO_EXTS, DEFAULT_DATA_DIR, FEATURES_DIR, extract_segments, load_analyzer, load_species, slugify,
)


def collect_files(data_dirs, allowed=None):
    """allowed: set of folder names to keep (None = every folder)."""
    items = []
    skipped = set()
    for root in map(Path, data_dirs):
        if not root.exists():
            print(f"  ! {root} does not exist, skipping")
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if allowed is not None and class_dir.name not in allowed:
                skipped.add(class_dir.name)
                continue
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in AUDIO_EXTS:
                    items.append((f, class_dir.name))
    if skipped:
        print(f"  skipped {len(skipped)} folder(s) not in the species list: {sorted(skipped)}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", default=[str(DEFAULT_DATA_DIR)])
    ap.add_argument("--out_dir", default=str(FEATURES_DIR))
    ap.add_argument("--all_folders", action="store_true",
                    help="use every folder, not just species in config/species_maharashtra.txt")
    args = ap.parse_args()

    allowed = None if args.all_folders else {slugify(n) for n in load_species()}
    items = collect_files(args.data_dirs, allowed)
    if not items:
        raise SystemExit("No audio files found under --data_dirs.")

    classes = sorted({c for _, c in items})
    label_map = {c: i for i, c in enumerate(classes)}
    print(f"{len(items)} audio files, {len(classes)} classes")

    analyzer = load_analyzer()
    X, y, rows = [], [], []
    failures = 0
    files_per_class = {c: 0 for c in classes}

    for path, cls in tqdm(items, desc="BirdNET embeddings"):
        try:
            segments = extract_segments(analyzer, path)
        except Exception as e:  # keep going on a single bad file
            failures += 1
            tqdm.write(f"  ! {path}: {e}")
            continue
        if not segments:
            continue
        files_per_class[cls] += 1
        for start, end, vec in segments:
            X.append(vec)
            y.append(label_map[cls])
            rows.append([path.as_posix(), cls, start, end])

    if not X:
        raise SystemExit("No embeddings extracted - is ffmpeg installed?")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", np.asarray(X, dtype=np.float32))
    np.save(out / "y.npy", np.asarray(y, dtype=np.int64))
    (out / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "class", "start_time", "end_time"])
        w.writerows(rows)

    print(f"\n{len(X)} segment embeddings, {len(classes)} classes, {failures} failed files")
    print("Classes with fewer than 5 usable recordings:")
    for c, n in sorted(files_per_class.items(), key=lambda t: t[1]):
        if n < 5:
            print(f"  {c:35s} {n}")
    print("NOTE: models trained earlier no longer match this label_map; rerun 03 and 04.")


if __name__ == "__main__":
    main()
