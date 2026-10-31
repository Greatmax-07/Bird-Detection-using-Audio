#!/usr/bin/env bash
# Run from the repo root (after committing the previous state):  bash apply_update4.sh
set -euo pipefail
[ -f birdear/core.py ] || { echo "run from the repo root"; exit 1; }
mkdir -p ref

git rm -qf --ignore-unmatch scripts/04_train_window_svm.py scripts/debug_windows.py
rm -f scripts/04_train_window_svm.py scripts/debug_windows.py

cat > birdear/core.py << 'EOF'
"""
Single source of truth for everything that must be IDENTICAL between
training (scripts/02-04) and serving (web/app/main.py):

    - constants (segment length, dimensions)
    - BirdNET loading + embedding extraction
    - L2-normalise -> mean|max|std pooling -> L2-normalise
    - BirdNET's native per-species scores for our classes
    - the shared recording-level train/validation split
    - model metadata (label-map hash) used to refuse mismatched models
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
SPECIES_FILE = ROOT / "config" / "species_maharashtra.txt"
DEFAULT_DATA_DIR = ROOT / "data" / "xc"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

SEGMENT_SEC = 3.0          # BirdNET chunk length (birdnetlib, overlap=0)
EMBED_DIM = 1024           # BirdNET v2.4 penultimate layer
POOLED_DIM = 3 * EMBED_DIM  # mean | max | std


# ----------------------------------------------------------------------------
# species / labels
# ----------------------------------------------------------------------------

def slugify(name: str) -> str:
    """'Tickell's Blue Flycatcher' -> 'Tickells_Blue_Flycatcher' (folder/class name)."""
    return name.replace("'", "").strip().replace(" ", "_")


def load_species_entries(path: Path | str = SPECIES_FILE) -> list[list[str]]:
    """
    Species file: one species per line, '#' starts a comment/group line.
    'Rock Dove | Rock Pigeon' = same species under several xeno-canto spellings.
    The FIRST name is the canonical one (class/folder name); all are searched/matched.
    """
    entries: list[list[str]] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        names = [n.strip() for n in line.split("|") if n.strip()]
        if names[0] not in seen:
            seen.add(names[0])
            entries.append(names)
    return entries


def load_species(path: Path | str = SPECIES_FILE) -> list[str]:
    """Canonical species names only."""
    return [e[0] for e in load_species_entries(path)]


def name_key(name: str) -> str:
    """Spelling-insensitive key: 'Paradise-flycatcher' == 'paradise flycatcher'."""
    return "".join(ch for ch in name.lower().replace("grey", "gray") if ch.isalnum())


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_names_from_label_map(label_map: dict[str, int]) -> list[str]:
    """label_map {name: id} -> list of names indexed by id. ids must be 0..N-1."""
    ids = sorted(label_map.values())
    if ids != list(range(len(ids))):
        raise ValueError("label_map ids are not contiguous 0..N-1")
    names = [""] * len(ids)
    for name, i in label_map.items():
        names[i] = name
    return names


def load_label_map(features_dir: Path | str = FEATURES_DIR) -> dict[str, int]:
    with open(Path(features_dir) / "label_map.json", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# model metadata (guards against label_map <-> pickle mismatch)
# ----------------------------------------------------------------------------

def meta_path(features_dir: Path | str = FEATURES_DIR) -> Path:
    return Path(features_dir) / "model_meta.json"


def read_meta(features_dir: Path | str = FEATURES_DIR) -> dict:
    p = meta_path(features_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_meta(section: str, data: dict, features_dir: Path | str = FEATURES_DIR) -> None:
    """Merge data under meta[section]; always stamp the current label_map hash."""
    meta = read_meta(features_dir)
    meta[section] = {
        **data,
        "label_map_sha256": file_sha256(Path(features_dir) / "label_map.json"),
    }
    meta_path(features_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# BirdNET
# ----------------------------------------------------------------------------

_patched = False


def patch_tflite() -> None:
    """birdnetlib reads an intermediate tensor, so tensors must be preserved."""
    global _patched
    if _patched:
        return
    import tensorflow as tf

    orig = tf.lite.Interpreter.__init__

    def patched(self, *args, **kwargs):
        kwargs["experimental_preserve_all_tensors"] = True
        orig(self, *args, **kwargs)

    tf.lite.Interpreter.__init__ = patched
    _patched = True


def load_analyzer():
    """Load BirdNET once (downloads the model on first ever use)."""
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    patch_tflite()
    from birdnetlib.analyzer import Analyzer

    analyzer = Analyzer()
    analyzer.interpreter.allocate_tensors()
    return analyzer


def extract_segments(analyzer, path: Path | str) -> list[tuple[float, float, np.ndarray]]:
    """Audio file -> [(start_s, end_s, 1024-D float32 embedding), ...] in time order."""
    from birdnetlib import Recording

    rec = Recording(analyzer, str(path))
    rec.extract_embeddings()
    out = []
    for seg in rec.embeddings:
        vec = seg.get("embeddings")
        if vec is None:
            vec = seg.get("embedding")
        if vec is None or seg.get("start_time") is None or seg.get("end_time") is None:
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != EMBED_DIM:
            raise ValueError(f"expected {EMBED_DIM}-D embedding, got {arr.shape[0]}")
        out.append((float(seg["start_time"]), float(seg["end_time"]), arr))
    out.sort(key=lambda t: t[0])
    return out


def birdnet_label_index(analyzer, class_names: list[str]) -> np.ndarray:
    """
    Class names (folder slugs, e.g. 'Rock_Dove') -> index into BirdNET's label list,
    or -1 when BirdNET has no species with that (or an alias) common name.
    """
    by_common: dict[str, int] = {}
    for i, label in enumerate(analyzer.labels):          # "Genus species_Common Name"
        by_common.setdefault(name_key(label.split("_", 1)[1]), i)
    entries = {slugify(e[0]): e for e in load_species_entries()}
    out = []
    for cls in class_names:
        names = entries.get(cls, [cls.replace("_", " ")])
        out.append(next((by_common[name_key(n)] for n in names if name_key(n) in by_common), -1))
    return np.asarray(out, dtype=np.int64)


def analyze_audio(analyzer, path: Path | str, label_idx: np.ndarray, batch: int = 1):
    """
    ONE BirdNET pass over a file (one chunk per invoke: larger batches are slower and need
    several GB because all intermediate tensors are preserved). Returns
        times  (n, 2)        segment start/end seconds (3 s segments, no overlap)
        emb    (n, 1024)     penultimate-layer embeddings
        scores (n, C)        BirdNET's own sigmoid score for each of our C classes
                             (0 where BirdNET has no matching species)
    """
    from birdnetlib import Recording

    rec = Recording(analyzer, str(path))
    rec.read_audio_data()
    chunks = np.asarray(rec.chunks, dtype=np.float32)
    C = len(label_idx)
    if len(chunks) == 0:
        return np.zeros((0, 2)), np.zeros((0, EMBED_DIM), np.float32), np.zeros((0, C), np.float32)

    interp, in_idx, out_idx = analyzer.interpreter, analyzer.input_layer_index, analyzer.output_layer_index
    logits, embs = [], []
    for i in range(0, len(chunks), batch):
        data = chunks[i:i + batch]
        interp.resize_tensor_input(in_idx, [len(data), *data[0].shape])
        interp.allocate_tensors()
        interp.set_tensor(in_idx, data)
        interp.invoke()
        logits.append(interp.get_tensor(out_idx).copy())
        embs.append(interp.get_tensor(out_idx - 1).copy())
    logits = np.concatenate(logits)
    emb = np.concatenate(embs).astype(np.float32)
    sig = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))

    scores = np.zeros((len(chunks), C), dtype=np.float32)
    ok = label_idx >= 0
    scores[:, ok] = sig[:, label_idx[ok]]
    starts = np.arange(len(chunks)) * SEGMENT_SEC
    return np.stack([starts, starts + SEGMENT_SEC], axis=1), emb, scores


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def pool_segments(segments: np.ndarray) -> np.ndarray:
    """(n, 1024) segment embeddings -> (3072,) L2-normalised [mean | max | std]."""
    seg = l2_normalize(np.asarray(segments, dtype=np.float32), axis=1)
    vec = np.concatenate([seg.mean(axis=0), seg.max(axis=0), seg.std(axis=0)])
    return l2_normalize(vec).astype(np.float32)


# ----------------------------------------------------------------------------
# shared split (used by BOTH 03 and 04 so their validation sets are identical)
# ----------------------------------------------------------------------------

def recording_split(paths, labels, test_frac: float = 0.2, seed: int = 42):
    """
    paths: one entry per recording, labels: class id per recording.
    Returns (train_paths:set, val_paths:set). Deterministic: input is sorted first.
    Classes with <2 recordings cannot be stratified: they go to train only.
    """
    from sklearn.model_selection import train_test_split

    order = np.argsort(np.asarray(paths, dtype=object).astype(str))
    paths = np.asarray(paths, dtype=object)[order]
    labels = np.asarray(labels)[order]

    counts = {c: int((labels == c).sum()) for c in np.unique(labels)}
    rare = {c for c, n in counts.items() if n < 2}
    if rare:
        warnings.warn(f"{len(rare)} class id(s) have <2 recordings; kept in train only: {sorted(rare)}")
    eligible = np.array([c not in rare for c in labels])

    try:
        tr, va = train_test_split(
            paths[eligible], test_size=test_frac, random_state=seed, stratify=labels[eligible]
        )
    except ValueError:
        warnings.warn("stratified split impossible; falling back to a plain random split")
        tr, va = train_test_split(paths[eligible], test_size=test_frac, random_state=seed)

    train = set(tr) | set(paths[~eligible])
    val = set(va)
    assert not (train & val), "recording leakage"
    return train, val
EOF

cat > birdear/detect.py << 'EOF'
"""
Multi-species detection from per-segment scores. Pure numpy, no models, so the
server and the evaluation script share exactly the same logic.

Per 3 s segment and species we have two independent scores:
    head     our MLP head trained on Maharashtra Xeno-canto clips (sigmoid outputs)
    birdnet  BirdNET's own multi-label score for that species
They are blended (w_birdnet), then thresholded with hysteresis:
    * a species is reported only if it reaches t_high somewhere in the file
    * its events then extend over neighbouring segments above t_low
    * segments closer than merge_gap seconds are merged into one event
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DetectParams:
    w_birdnet: float = 0.5
    t_high: float = 0.40
    t_low: float = 0.20
    merge_gap: float = 3.0

    @classmethod
    def from_env(cls) -> "DetectParams":
        d = cls()
        return cls(
            w_birdnet=float(os.environ.get("BIRDEAR_W_BIRDNET", d.w_birdnet)),
            t_high=float(os.environ.get("BIRDEAR_T_HIGH", d.t_high)),
            t_low=float(os.environ.get("BIRDEAR_T_LOW", d.t_low)),
            merge_gap=float(os.environ.get("BIRDEAR_MERGE_GAP", d.merge_gap)),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def fuse(p_head: np.ndarray, p_birdnet: np.ndarray, has_birdnet: np.ndarray, w: float) -> np.ndarray:
    """(n, C) scores. Species BirdNET does not know use the head alone."""
    return np.where(has_birdnet[None, :], w * p_birdnet + (1.0 - w) * p_head, p_head)


def detect_events(times: np.ndarray, fused: np.ndarray, params: DetectParams) -> dict[int, list[dict]]:
    """{class_id: [event, ...]}; event = start_time, end_time, confidence (max), segments."""
    out: dict[int, list[dict]] = {}
    for k in range(fused.shape[1]):
        col = fused[:, k]
        if col.size == 0 or col.max() < params.t_high:
            continue
        events: list[dict] = []
        for s in np.flatnonzero(col >= params.t_low):
            start, end, conf = float(times[s, 0]), float(times[s, 1]), float(col[s])
            if events and start - events[-1]["end_time"] <= params.merge_gap:
                e = events[-1]
                e["end_time"] = end
                e["confidence"] = max(e["confidence"], conf)
                e["segments"] += 1
            else:
                events.append({"start_time": start, "end_time": end, "confidence": conf, "segments": 1})
        events = [e for e in events if e["confidence"] >= params.t_high]
        if events:
            out[k] = events
    return out
EOF

cat > scripts/04_train_segment_head.py << 'EOF'
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
EOF

cat > scripts/05_evaluate.py << 'EOF'
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
EOF

cat > scripts/inspect_clip.py << 'EOF'
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
EOF

cat > web/app/main.py << 'EOF'
"""
BirdEar API.

    GET  /          single-page UI
    GET  /health    {status, n_classes, n_with_birdnet_scores, params}
    POST /predict   multipart field "file" (.wav .mp3 .flac .ogg .m4a, <=30 MiB)

Run from the repo root:
    uvicorn main:app --app-dir web/app --reload
"""
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from birdnetlib.exceptions import AudioFormatError
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from birdear.core import (
    AUDIO_EXTS, EMBED_DIM, FEATURES_DIR, analyze_audio, birdnet_label_index,
    class_names_from_label_map, file_sha256, l2_normalize, load_analyzer, load_label_map, read_meta,
)
from birdear.detect import DetectParams, detect_events, fuse

log = logging.getLogger("birdear")
logging.basicConfig(level=os.environ.get("BIRDEAR_LOG", "INFO"))

STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TOP_K = 5                                   # strongest species scores returned in "predictions"
COUNT_NOTE = ("Approximate acoustic occurrence count: separate stretches of calling. "
              "It is not a count of individual birds.")

state: dict = {}
_birdnet_lock = threading.Lock()            # the TFLite interpreter is not thread-safe


# ----------------------------------------------------------------------------
# startup
# ----------------------------------------------------------------------------

def _load_state() -> None:
    head_path = FEATURES_DIR / "segment_head.pkl"
    lm_path = FEATURES_DIR / "label_map.json"
    missing = [p for p in (head_path, lm_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {[str(p) for p in missing]}. Train first: scripts/02 and 04 (see README).")

    meta = read_meta()
    if meta.get("segment", {}).get("label_map_sha256") != file_sha256(lm_path):
        raise RuntimeError("segment head was not trained with the current features/label_map.json "
                           "(species/ids differ). Re-run scripts/04 after script 02.")

    names = class_names_from_label_map(load_label_map())
    head = joblib.load(head_path)
    if head.n_features_in_ != EMBED_DIM or head.n_outputs_ != len(names):
        raise RuntimeError(f"segment head shape ({head.n_features_in_} in, {head.n_outputs_} out) does not "
                           f"match {EMBED_DIM} features / {len(names)} classes")

    analyzer = load_analyzer()
    label_idx = birdnet_label_index(analyzer, names)
    state.update(names=names, head=head, analyzer=analyzer, label_idx=label_idx,
                 has_bn=label_idx >= 0, params=DetectParams.from_env())
    log.info("loaded %d classes (%d with BirdNET scores), params %s",
             len(names), int(state["has_bn"].sum()), state["params"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield
    state.clear()


app = FastAPI(title="BirdEar", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------

def _species(class_id: int) -> str:
    return state["names"][class_id].replace("_", " ")


def analyze_file(path: Path) -> dict:
    try:
        with _birdnet_lock:
            times, emb, bn = analyze_audio(state["analyzer"], path, state["label_idx"])
    except AudioFormatError as e:
        raise ValueError(f"Could not decode the audio file ({e}).") from e
    if len(times) == 0:
        raise ValueError("The audio is too short to analyse (need at least 3 seconds).")

    params = state["params"]
    p_head = state["head"].predict_proba(l2_normalize(emb, axis=1))
    fused = fuse(p_head, bn, state["has_bn"], params.w_birdnet)
    events = detect_events(times, fused, params)

    detected = [{
        "species": _species(k), "event_count": len(ev), "estimated_count": len(ev),
        "count_note": COUNT_NOTE, "confidence": max(e["confidence"] for e in ev),
        "events": [{**e, "windows": e["segments"]} for e in ev],   # "windows" kept for older UIs
    } for k, ev in events.items()]
    detected.sort(key=lambda d: d["confidence"], reverse=True)

    peak = fused.max(axis=0)
    predictions = [{"species": _species(int(k)), "confidence": float(peak[k])}
                   for k in np.argsort(peak)[::-1][:TOP_K]]
    return {
        "predictions": predictions,
        "species_detected": detected,
        "no_confident_detection": not detected,
        "total_approximate_count": sum(d["estimated_count"] for d in detected),
        "count_note": COUNT_NOTE,
        "duration_seconds": float(times[-1, 1]),
    }


# ----------------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "n_classes": len(state["names"]),
            "n_with_birdnet_scores": int(state["has_bn"].sum()), "params": state["params"].as_dict()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTS:
        return JSONResponse(status_code=422, content={
            "error": f"Unsupported file type. Use one of: {', '.join(sorted(AUDIO_EXTS))}."})

    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=422, content={"error": "Uploaded audio file is empty."})
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=422, content={"error": "Audio file is too large (max 30 MiB)."})

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    try:
        tmp.write(raw)
        tmp.close()
        return await run_in_threadpool(analyze_file, tmp_path)
    except ValueError as e:                      # bad/undecodable audio
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception as e:                       # genuine server problem
        log.exception("prediction failed")
        return JSONResponse(status_code=500, content={"error": f"Failed to process audio: {e}"})
    finally:
        tmp_path.unlink(missing_ok=True)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
EOF

cat > tests/test_core.py << 'EOF'
import numpy as np

from birdear.core import (POOLED_DIM, class_names_from_label_map, load_species, name_key, pool_segments,
                          recording_split, slugify)
from birdear.detect import DetectParams, detect_events, fuse


def test_pool_shape_and_norm():
    v = pool_segments(np.random.default_rng(0).normal(size=(7, 1024)))
    assert v.shape == (POOLED_DIM,)
    assert abs(np.linalg.norm(v) - 1) < 1e-5


def test_split_is_deterministic_and_disjoint():
    paths = [f"c{c}/{i}.wav" for c in range(4) for i in range(10)]
    labels = [c for c in range(4) for _ in range(10)]
    t1, v1 = recording_split(paths, labels)
    t2, v2 = recording_split(paths[::-1], labels[::-1])
    assert t1 == t2 and v1 == v2 and not (t1 & v1)


def test_species_file_aliases_and_slug():
    from birdear.core import load_species_entries
    names = load_species()
    assert len(names) == len(set(names)) > 50
    assert slugify("Tickell's Blue Flycatcher") == "Tickells_Blue_Flycatcher"
    assert next(e for e in load_species_entries() if e[0] == "Rock Dove") == ["Rock Dove", "Rock Pigeon"]
    assert name_key("Paradise-flycatcher") == name_key("paradise Flycatcher")
    assert name_key("Indian Grey Hornbill") == name_key("Indian Gray Hornbill")


def test_label_map_contiguous():
    assert class_names_from_label_map({"b": 1, "a": 0}) == ["a", "b"]


def _times(n):
    s = np.arange(n) * 3.0
    return np.stack([s, s + 3], axis=1)


def test_detect_two_species_hysteresis_and_gating():
    f = np.zeros((10, 3))
    f[2:7, 0] = [0.25, 0.3, 0.8, 0.3, 0.25]      # species 0: strong core with weak shoulders
    f[8, 1] = 0.3                                # species 1: never reaches t_high -> ignored
    f[0:2, 2] = 0.9                              # species 2: separate strong event
    f[8:10, 2] = 0.9                             # species 2: a second event (gap of 6 s)
    ev = detect_events(_times(10), f, DetectParams(t_high=0.4, t_low=0.2, merge_gap=3.0))
    assert set(ev) == {0, 2}
    assert len(ev[0]) == 1 and ev[0][0]["start_time"] == 6.0 and ev[0][0]["end_time"] == 21.0
    assert len(ev[2]) == 2


def test_isolated_weak_segment_is_dropped():
    f = np.zeros((12, 1))
    f[1] = 0.9
    f[10] = 0.25                                 # far from the strong part, below t_high
    ev = detect_events(_times(12), f, DetectParams())
    assert len(ev[0]) == 1


def test_fuse_uses_head_only_without_birdnet():
    ph, bn = np.full((2, 2), 0.2), np.full((2, 2), 0.8)
    out = fuse(ph, bn, np.array([True, False]), 0.5)
    assert np.allclose(out[:, 0], 0.5) and np.allclose(out[:, 1], 0.2)
EOF

cat > README.md << 'EOF'
# BirdEar

Detect **every bird species** heard in an audio file, restricted to species commonly found in
**Maharashtra, India**, with timestamps.

## How it works

1. **BirdNET v2.4** (frozen) is run once per 3-second segment and yields two things:
   a 1024-D embedding and its own multi-label species scores.
2. **Segment head** (`04`): a small MLP on the embeddings, trained on Xeno-canto clips of the species in
   `config/species_maharashtra.txt`. Independent sigmoid outputs, so several species can score in one segment.
3. **Fusion** (`birdear/detect.py`): `w * BirdNET score + (1 - w) * head score` per species and segment, then
   hysteresis thresholds: a species is reported if it reaches `t_high` somewhere; its events extend over
   neighbouring segments above `t_low`. If nothing qualifies, the UI says "No confident bird detected".

Defaults `w=0.5, t_high=0.40, t_low=0.20` are untuned; measure and pick them with `05_evaluate.py`.

> Limits: the head is trained on single-species clips, and `estimated_count` counts separate stretches
> of calling, not individual birds.

## Setup (Fedora, fish)

```fish
sudo dnf install -y uv ffmpeg
uv venv --python 3.12 .venv
source .venv/bin/activate.fish
uv pip install -r requirements.txt
uv pip install -e .
python -m pytest -q tests
```

TensorFlow needs Python 3.11/3.12 (not 3.13/3.14). BirdNET's model downloads on first use.

## Pipeline

```fish
set -x XC_API_KEY your_key                          # https://xeno-canto.org/account
python scripts/01_download_weak_species.py --preview   # check every name matches
python scripts/01_download_weak_species.py             # -> data/xc/<Species>/xc_*.mp3
python scripts/02_extract_birdnet_embeddings.py        # -> features/{X,y}.npy, manifest.csv, label_map.json
python scripts/03_train_classifier_head.py             # optional baseline: whole-recording SVM
python scripts/04_train_segment_head.py                # -> features/segment_head.pkl (used by the server)
python scripts/05_evaluate.py                          # single / 2-species mixtures / your own labelled clips
python scripts/inspect_clip.py some_clip.mp3           # per-segment scores for one file
```

Rerun 04 after every run of 02. All scripts share one recording-level validation split.
Metrics are stored in `features/model_meta.json`.

**Your own test clips** for `05_evaluate.py --real_dir DIR`: folders named after classes
(`DIR/Rose-ringed_Parakeet/x.mp3`, several species with `+`: `DIR/Rose-ringed_Parakeet+House_Crow/y.wav`,
and `DIR/_none/` for clips with no bird).

## Serve

```fish
uvicorn main:app --app-dir web/app --reload    # http://127.0.0.1:8000
```

Thresholds can be changed without code: `BIRDEAR_W_BIRDNET`, `BIRDEAR_T_HIGH`, `BIRDEAR_T_LOW`,
`BIRDEAR_MERGE_GAP`. The server refuses to start if the head was trained against a different `label_map.json`.

| Endpoint | |
|---|---|
| `GET /health` | class count, detector parameters |
| `POST /predict` | multipart `file` (.wav .mp3 .flac .ogg .m4a, max 30 MiB) |

`/predict` returns `species_detected` (events per species), `no_confident_detection`, `predictions`
(strongest per-species scores), `total_approximate_count`, `duration_seconds`.

## Layout

```
birdear/core.py      constants, BirdNET loading + single-pass analysis, pooling, split, metadata
birdear/detect.py    score fusion and event detection (shared by server and evaluation)
config/              species_maharashtra.txt
scripts/             01 download, 02 embed, 03 recording SVM (baseline), 04 segment head, 05 evaluate, inspect_clip
web/app/             FastAPI backend + static/index.html UI
tests/               unit tests
data/  features/     generated, gitignored
```
EOF

python3 - << 'PYEOF'
p = "web/app/static/index.html"
s = open(p, encoding="utf-8").read()
if "noConfident" in s:
    print("index.html already patched"); raise SystemExit

a = s.index("function renderResults(predictions, speciesDetected) {")
b = s.index("    /* -----------------------------------------------------\n       OTHER RECORDING-LEVEL PREDICTIONS")
new_top = """function renderResults(predictions, speciesDetected, noConfident) {
    const top = predictions && predictions[0];
    const found = speciesDetected || [];
    const multi = found.length > 1;

    if (noConfident || found.length === 0) {
        topSpecies.textContent = 'No confident bird detected';
        topConfidence.textContent = top
            ? `Closest weak match: ${top.species} (${Math.round(top.confidence * 100)}%)`
            : '';
        topDescription.textContent =
            'No species reached the detection threshold. The clip may be noise, silence, or a species the model does not know.';
    } else if (multi) {
        topSpecies.textContent = `${found.length} species detected`;
        topConfidence.textContent = found.map((d) => d.species).join(' \\u00b7 ');
        topDescription.textContent =
            'Species found anywhere in the recording. See the timeline below for when each was heard.';
    } else {
        topSpecies.textContent = found[0].species;
        topConfidence.textContent =
            `${Math.round(found[0].confidence * 100)}% detection score`;
        topDescription.textContent =
            'The only species detected in this recording.';
    }


"""
s = s[:a] + new_top + s[b:]

def sub(old, new):
    global s
    assert old in s, "pattern not found: " + old[:50]
    s = s.replace(old, new, 1)

sub("    if (predictions && predictions.length > 1) {",
    "    const startAt = (multi || noConfident || found.length === 0) ? 0 : 1;\n"
    "    const showOthers = !(noConfident || found.length === 0) && predictions && predictions.length > startAt;\n\n"
    "    if (showOthers) {")
sub("            'Other recording-level predictions';",
    "            multi ? 'Strongest scores across the recording' : 'Other strong scores across the recording';")
sub("    (predictions || []).slice(1, 5).forEach((p) => {",
    "    (showOthers ? predictions.slice(startAt, 5) : []).forEach((p) => {")
sub("""        renderResults(
            data.predictions,
            data.species_detected || []
        );""", """        renderResults(
            data.predictions,
            data.species_detected || [],
            data.no_confident_detection === true
        );""")
open(p, "w", encoding="utf-8").write(s)
print("patched web/app/static/index.html")
PYEOF

if [ -f ref/context.md ] && ! grep -q '^### 14.15' ref/context.md; then
cat >> ref/context.md << 'EOF'

### 14.15 Architecture v2: fused per-segment multi-species detector (2026-10-31)
**Why:** real-clip tests showed the old 15 s window SVM (softmax, one label per window) hid quiet calls and cannot separate simultaneous species; traffic and wind clips produced no detections (top scores 6-8%). The goal is *every* bird in a file, so detection was rebuilt at 3 s resolution with independent per-species scores.

**Pipeline now:** one BirdNET pass per file (`birdear.core.analyze_audio`) returns segment times, 1024-D embeddings and BirdNET's own sigmoid scores for our classes (`birdnet_label_index` maps class names to BirdNET labels via common name, ignoring case/hyphens/spaces and grey=gray; every class except the dropped Indian Wren-Babbler matched). The **segment head** (`scripts/04_train_segment_head.py`, `sklearn MLPClassifier`, 512 hidden units, multilabel indicator targets => independent sigmoid outputs, <=40 segments per training recording, same split as 03) scores each segment on L2-normalised embeddings. `birdear/detect.py`: `fused = w*birdnet + (1-w)*head` (species without a BirdNET label use the head alone), then **hysteresis**: a species is reported only if some segment reaches `t_high`; events are runs of segments >= `t_low` merged when closer than `merge_gap` seconds; events whose peak is below `t_high` are dropped. Defaults (untuned): `w_birdnet=0.5, t_high=0.40, t_low=0.20, merge_gap=3.0 s`; env `BIRDEAR_W_BIRDNET / _T_HIGH / _T_LOW / _MERGE_GAP`.

**Removed from serving:** the window SVM (`scripts/04_train_window_svm.py` deleted; supersedes 14.5 detection algorithm, 14.14 two-threshold rule, `scripts/debug_windows.py`) and the recording SVM (script 03 stays as an optional whole-recording baseline). `window_ranges` and the window constants were removed from `birdear.core`. Server guard now checks `model_meta.json["segment"]`.

**API (`POST /predict`):** `species_detected` as before (events carry `segments`; `windows` kept as an alias), plus `no_confident_detection` (true when nothing qualifies), `duration_seconds`; `predictions` = top-5 species by peak fused score; `estimated_count` = number of events (separate stretches of calling, not birds). `GET /health` returns `n_classes`, `n_with_birdnet_scores`, `params`. UI: "No confident bird detected" state (shows closest weak match), "N species detected" summary when several, same timeline.

**New tools:** `scripts/05_evaluate.py` measures set-based precision/recall/F1 on (a) single validation recordings, (b) synthetic 2-species mixtures of validation recordings (random -6..+6 dB), (c) optional real labelled clips (`--real_dir`, folders = class names, `+` for several species, `_none` for no-bird), sweeps blend weight and thresholds, writes `features/eval_sweep.csv`. `scripts/inspect_clip.py` prints per-segment head/BirdNET/fused scores for one file.

**Performance finding:** BirdNET runs with `experimental_preserve_all_tensors=True`, so memory grows with batch size; on a 120 s clip batch 1 took 4.2 s / 1.1 GB, batch 4 7.3 s / 1.7 GB, batch 8 11.6 s / 2.7 GB. `analyze_audio` therefore uses one chunk per invoke (batch=1).

**Verified:** unit tests (7); `analyze_audio` embeddings and scores are bit-identical to birdnetlib's own `extract_embeddings` / `predict`; on a real multi-bird soundscape (BirdNET-Analyzer example, 120 s, North-American species) the detector produced per-species events consistent with birdnetlib's `analyze()`; server end-to-end and `05_evaluate.py` run on synthetic data. **Not verified:** head quality, fusion defaults and false-positive rate on Maharashtra audio (run 04 then 05). **Idea not yet done:** BirdNET also outputs non-bird classes (e.g. "Engine", wind/rain/human labels) that could be used as a background signal to suppress detections.
EOF
fi
echo "done. next: python -m pytest -q tests; python scripts/04_train_segment_head.py; python scripts/05_evaluate.py" 
