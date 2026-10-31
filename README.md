# BirdEar

Identify birds in an audio file, restricted to species commonly found in **Maharashtra, India**.

Frozen **BirdNET v2.4** embeddings (1024-D per 3 s segment) are pooled and fed to two RBF SVMs:

| Model | Input | Output |
|---|---|---|
| recording SVM (`03`) | whole file | top-5 species for the file |
| window SVM (`04`) | 15 s windows (5 segments), step 6 s | per-species time events, merged |

Training audio comes from Xeno-canto; the species list is `config/species_maharashtra.txt`.

> Status: single-label models trained on single-species clips. They do not yet separate
> overlapping birds, have no "no bird" class, and `estimated_count` counts acoustic
> occurrences, not individual birds. See `ref/context.md` for the pre-cleanup analysis.

## Setup (Fedora, fish)

```fish
sudo dnf install -y uv ffmpeg
uv venv --python 3.12 .venv
source .venv/bin/activate.fish
uv pip install -r requirements.txt
uv pip install -e .
python -m pytest -q tests
```

TensorFlow needs Python 3.11/3.12 (not 3.13/3.14). BirdNET's model is downloaded on first use.

## Pipeline

```fish
set -x XC_API_KEY your_key            # https://xeno-canto.org/account
python scripts/01_download_weak_species.py --preview   # check every name matches
python scripts/01_download_weak_species.py             # -> data/xc/<Species>/xc_*.mp3
python scripts/02_extract_birdnet_embeddings.py        # -> features/{X,y}.npy, manifest.csv, label_map.json
python scripts/03_train_classifier_head.py             # -> features/recording_svm.pkl
python scripts/04_train_window_svm.py                  # -> features/window_svm.pkl
```

Rerun 03 **and** 04 after every run of 02. Metrics land in `features/model_meta.json`.
Both models share one recording-level split (`birdear.core.recording_split`).

## Serve

```fish
uvicorn main:app --app-dir web/app --reload    # http://127.0.0.1:8000
```

The server refuses to start if the models were trained against a different `label_map.json`.

| Endpoint | |
|---|---|
| `GET /health` | class count, window parameters |
| `POST /predict` | multipart `file` (.wav .mp3 .flac .ogg .m4a, max 30 MiB) |

## Layout

```
birdear/core.py      shared constants, BirdNET loading, pooling, windows, split, metadata
config/              species_maharashtra.txt
scripts/             01 download, 02 embed, 03 recording SVM, 04 window SVM
web/app/             FastAPI backend + static/index.html UI
tests/               unit tests for birdear.core
data/  features/     generated, gitignored
ref/context.md       snapshot analysis of the project before this cleanup
```

Feature definition: L2-normalise each segment embedding, then `[mean | max | std]` over
segments (3072-D), L2-normalise again. Defined once in `birdear.core.pool_segments`.
