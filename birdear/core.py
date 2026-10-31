"""
Single source of truth for everything that must be IDENTICAL between
training (scripts/02-04) and serving (web/app/main.py):

    - constants (segment length, window size/step, dimensions)
    - BirdNET loading + embedding extraction
    - L2-normalise -> mean|max|std pooling -> L2-normalise
    - sliding-window ranges
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
WINDOW_SIZE = 5            # segments per window (15 s)
WINDOW_STEP = 2            # segments between window starts (6 s)


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
    return "".join(ch for ch in name.lower() if ch.isalnum())


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


def window_ranges(n: int, size: int = WINDOW_SIZE, step: int = WINDOW_STEP) -> list[tuple[int, int]]:
    """[start, end) segment index ranges. Always covers the final segment."""
    if n <= 0:
        return []
    if n <= size:
        return [(0, n)]
    ranges = [(s, s + size) for s in range(0, n - size + 1, step)]
    if ranges[-1][1] < n:
        ranges.append((n - size, n))
    return ranges


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
