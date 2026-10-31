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

Defaults `w=0.7, t_high=0.50, t_low=0.25` were chosen with `05_evaluate.py` on a small test set; re-measure as you add clips.

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
