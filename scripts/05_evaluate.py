"""
05_evaluate.py
==============
Measure the multi-species detector and pick its parameters.

Test sets (all unseen by the head: validation recordings only):
  single   N validation recordings per species                  (one bird)
  mix      synthetic mixtures of 2 recordings of different species, random relative
           level -6..+6 dB                                       (two birds at once)
  real     OPTIONAL real clips you label yourself with --real_dir:
               <real_dir>/Rose-ringed_Parakeet/clip1.mp3          one species
               <real_dir>/Rose-ringed_Parakeet+House_Crow/c.wav   several ('+')
               <real_dir>/_none/traffic.mp3                       no bird at all
           folder names are class names (see features/label_map.json)

    python scripts/05_evaluate.py
    python scripts/05_evaluate.py --real_dir "testing audios/labelled"

For every combination of blend weight (head vs BirdNET) and threshold it prints
set-based precision / recall / F1 per test set, plus how many no-bird clips
trigger a false detection. BirdNET is run once per clip and cached.
Writes features/eval_sweep.csv.
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdear.core import (  # noqa: E402
    AUDIO_EXTS, FEATURES_DIR, analyze_audio, birdnet_label_index, class_names_from_label_map,
    l2_normalize, load_analyzer, load_label_map, recording_split, slugify,
)
from birdear.detect import DetectParams, detect_events, fuse  # noqa: E402

warnings.filterwarnings("ignore")
SR = 48000


def make_mixture(path_a, path_b, rng, out_path, seconds=20):
    a, _ = librosa.load(path_a, sr=SR, mono=True, duration=seconds)
    b, _ = librosa.load(path_b, sr=SR, mono=True, duration=seconds)
    n = max(len(a), len(b))
    a, b = np.pad(a, (0, n - len(a))), np.pad(b, (0, n - len(b)))
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-9)  # noqa: E731
    b = b * (rms(a) / rms(b)) * 10 ** (rng.uniform(-6, 6) / 20)
    mix = a + b
    sf.write(out_path, (0.9 * mix / (np.abs(mix).max() + 1e-9)).astype(np.float32), SR)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=str(FEATURES_DIR))
    ap.add_argument("--per_class", type=int, default=2, help="validation recordings per species")
    ap.add_argument("--mixtures", type=int, default=80)
    ap.add_argument("--real_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    names = class_names_from_label_map(load_label_map(fdir))
    cid = {n: i for i, n in enumerate(names)}
    head = joblib.load(fdir / "segment_head.pkl")
    analyzer = load_analyzer()
    label_idx = birdnet_label_index(analyzer, names)
    has_bn = label_idx >= 0
    print(f"{has_bn.sum()}/{len(names)} classes have a BirdNET species score")

    # ---- validation recordings (same split as scripts 03/04)
    manifest = pd.read_csv(fdir / "manifest.csv", usecols=["filepath", "class"]).drop_duplicates("filepath")
    paths = manifest["filepath"].to_numpy(dtype=object)
    labels = np.array([cid[c] for c in manifest["class"]])
    _, val = recording_split(paths, labels, args.test_frac, args.seed)
    rng = np.random.default_rng(args.seed)
    by_class: dict[int, list] = {}
    for p, l in zip(paths, labels):
        if p in val and Path(p).exists():
            by_class.setdefault(int(l), []).append(p)

    clips = []  # (group, path, true class-id set)
    for c, ps in by_class.items():
        for p in rng.permutation(ps)[: args.per_class]:
            clips.append(("single", p, {c}))
    tmp = Path(tempfile.mkdtemp())
    classes = list(by_class)
    for i in range(args.mixtures if len(classes) >= 2 else 0):
        a, b = rng.choice(classes, 2, replace=False)
        pa, pb = rng.choice(by_class[a]), rng.choice(by_class[b])
        out = tmp / f"mix{i}.wav"
        make_mixture(pa, pb, rng, out)
        clips.append(("mix", str(out), {int(a), int(b)}))
    if args.real_dir:
        for d in sorted(p for p in Path(args.real_dir).iterdir() if p.is_dir()):
            truth = set() if d.name == "_none" else {cid[s] for s in d.name.split("+")}
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in AUDIO_EXTS:
                    clips.append(("noise" if not truth else "real", str(f), truth))
    print(f"{len(clips)} clips: " + ", ".join(f"{g}={sum(1 for c in clips if c[0] == g)}" for g in
                                              ('single', 'mix', 'real', 'noise')))

    # ---- one BirdNET pass per clip
    cache = []
    for i, (g, p, truth) in enumerate(clips):
        times, emb, bn = analyze_audio(analyzer, p, label_idx)
        cache.append((g, truth, times, head.predict_proba(l2_normalize(emb, axis=1)), bn))
        print(f"  analysed {i + 1}/{len(clips)}", end="\r")
    print()

    # ---- sweep
    rows = []
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        for t_high in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            params = DetectParams(w_birdnet=w, t_high=t_high, t_low=t_high / 2)
            tot = {}
            for g, truth, times, ph, bn in cache:
                found = set(detect_events(times, fuse(ph, bn, has_bn, w), params))
                t = tot.setdefault(g, [0, 0, 0, 0, 0])  # tp fp fn clips clips_with_detection
                t[0] += len(found & truth); t[1] += len(found - truth); t[2] += len(truth - found)
                t[3] += 1; t[4] += bool(found)
            row = {"w_birdnet": w, "t_high": t_high}
            for g, (tp, fp, fn, n, nd) in tot.items():
                if g == "noise":
                    row["noise_false_pos_clips"] = f"{nd}/{n}"
                else:
                    pr, rc, f1 = prf(tp, fp, fn)
                    row.update({f"{g}_P": pr, f"{g}_R": rc, f"{g}_F1": f1})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(fdir / "eval_sweep.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(df.to_string(index=False))
    f1cols = [c for c in df.columns if c.endswith("_F1")]
    best = df.assign(mean_F1=df[f1cols].mean(axis=1)).sort_values("mean_F1", ascending=False).iloc[0]
    print(f"\nbest mean F1 over {f1cols}: w_birdnet={best.w_birdnet}, t_high={best.t_high} "
          f"(t_low = t_high/2)  -> set BIRDEAR_W_BIRDNET / BIRDEAR_T_HIGH / BIRDEAR_T_LOW")
    print("Check the noise column before trusting that row: false detections on no-bird clips matter.")


if __name__ == "__main__":
    main()
