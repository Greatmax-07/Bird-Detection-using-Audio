# BirdEar — Complete Reference

> Indian bird sound identifier. 61 species. BirdNET embeddings + focal-loss MLP head. FastAPI + ONNX Runtime serving.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Pipeline Scripts](#pipeline-scripts)
5. [Model](#model)
6. [Web App](#web-app)
7. [Inference Logic](#inference-logic)
8. [Tuning Knobs](#tuning-knobs)
9. [Species List](#species-list)
10. [Known Issues & Decisions](#known-issues--decisions)
11. [Quickstart](#quickstart)

---

## Project Structure

```
Bird-Detection-using-Audio/
├── ibc53/                          # Primary training data (48 Indian species, common-name folders)
├── xc_downloads/                   # Xeno-canto top-up for rare iBC53 species (21 species)
├── xc_urban/                       # Xeno-canto downloads for common urban species (13 new species)
├── features/
│   ├── X.npy                       # (N, 1024) BirdNET embedding vectors
│   ├── y.npy                       # (N,) integer labels
│   ├── label_map.json              # {class_name: integer_id}
│   ├── manifest.csv                # source file + time offset per embedding row
│   └── classifier_head_best.pt    # best MLP checkpoint
├── scripts/
│   ├── 00_prepare_ibc53.py         # rename scientific→common name, drop excluded classes
│   ├── 01_download_weak_species.py # Xeno-canto downloader for rare/urban species
│   ├── 02_extract_birdnet_embeddings.py  # frozen BirdNET → 1024-dim embeddings
│   ├── 03_train_classifier_head.py # MLP head training (focal loss + class weights)
│   └── 04_export_head_onnx.py      # export trained head to ONNX
├── web/app/
│   ├── main.py                     # FastAPI backend
│   └── static/index.html          # upload/record UI
├── bird_classifier.onnx            # deployed model (symlinked into web/)
├── bird_classifier.classes.json    # {index: class_name} for inference
├── docs/
│   └── REFERENCE.md                # this file
└── requirements.txt
```

---

## Architecture

### Training pipeline

```
iBC53 (raw) → 00_prepare → common-name folders
                                    ↓
              xc_downloads + xc_urban (Xeno-canto)
                                    ↓
                    02_extract → BirdNET embeddings (1024-dim, frozen)
                                    ↓
                    03_train  → MLP head (focal loss + inverse-freq weights)
                                    ↓
                    04_export → bird_classifier.onnx
```

### Inference pipeline (per request)

```
Browser uploads audio
        ↓
FastAPI (main.py)
        ↓
birdnetlib.Recording.extract_embeddings()
  → splits audio into ~3s segments
  → runs frozen BirdNET TFLite model
  → returns 1024-dim vector per segment
        ↓
For each segment:
  ONNX classifier head → logits → softmax → per-class probabilities
  Any class ≥ DETECTION_THRESHOLD gets recorded
        ↓
Aggregate: species detected in ≥ MIN_DETECTIONS segments pass the filter
Sort by (detection_count DESC, avg_confidence DESC)
        ↓
Return JSON: {total_segments, species_found, detections: [{species, detections, confidence, peak_confidence}]}
```

---

## Dataset

### Sources

| Source | Purpose | Classes |
|---|---|---|
| iBC53 | Primary — Northeast India endemic/rare species | 48 (after cleanup) |
| Xeno-canto (`xc_downloads`) | Top-up for rare iBC53 tail species | 21 species |
| Xeno-canto (`xc_urban`) | Common Indian urban/commensal species | 13 new species |

**Total: 61 classes, 23,650 embedding vectors**

### Class imbalance

Raw iBC53 ratio is ~90:1 (Puff-throated Babbler: 1300+ clips vs Black-bellied Plover: <10 clips). Addressed by:
1. Xeno-canto top-up for the 21 weakest species
2. Focal loss (γ=2.0) during head training
3. Inverse-frequency class weights

### Dropped classes

These were in the raw iBC53 but excluded:
- `Mystery` — unlabeled
- `Black-browed_Reed_Warbler` — only 1 recording
- `Asian_Emerald_Cuckoo` — only 2 recordings
- `Grey-throated_Babbler` — identity ambiguous (moved to `_excluded/`)

---

## Pipeline Scripts

### `00_prepare_ibc53.py`

Renames scientific-name folders to common-name convention, drops excluded classes.

```bash
python scripts/00_prepare_ibc53.py --data_dir ./ibc53 --dry-run   # preview
python scripts/00_prepare_ibc53.py --data_dir ./ibc53 --delete-dropped
```

Uses `birdnetlib.Analyzer` to load the official BirdNET label list for name mapping — guaranteed to match the installed model version.

---

### `01_download_weak_species.py`

Downloads recordings from Xeno-canto API v3. Needs a free API key from https://xeno-canto.org/account.

```bash
export XC_API_KEY="your-key"
python scripts/01_download_weak_species.py --out_dir ./xc_downloads --api_key $XC_API_KEY
python scripts/01_download_weak_species.py --out_dir ./xc_urban --api_key $XC_API_KEY
```

- Searches India first; auto-expands to Bangladesh, Nepal, Sri Lanka, Myanmar, Bhutan, Pakistan if India returns fewer than `--min_per_species` (default 15)
- Skips already-downloaded files (checked by Xeno-canto ID)
- Writes `metadata.csv` per species with provenance, license, quality

To add species: edit `TARGET_SPECIES` list in the script.

---

### `02_extract_birdnet_embeddings.py`

Runs every audio file through frozen BirdNET, saves 1024-dim embeddings.

```bash
python scripts/02_extract_birdnet_embeddings.py \
    --data_dirs ./ibc53 ./xc_downloads ./xc_urban \
    --out_dir ./features
```

Outputs: `features/X.npy`, `y.npy`, `label_map.json`, `manifest.csv`

**Requires ffmpeg on PATH** (birdnetlib dependency).

Supported audio formats: `.wav .mp3 .flac .ogg .m4a`

---

### `03_train_classifier_head.py`

Trains a 2-layer MLP on the frozen embeddings.

```bash
python scripts/03_train_classifier_head.py --features_dir ./features
```

Architecture:
```
Linear(1024 → 256) → ReLU → Dropout(0.3) → Linear(256 → 61)
```

Key flags: `--epochs 40`, `--batch_size 64`, `--lr 1e-3`, `--val_frac 0.2`

Loss: focal loss (γ=2.0) + inverse-frequency class weights. Saves best checkpoint by val loss to `features/classifier_head_best.pt`.

Prints per-class precision/recall/F1 at end. Current macro F1: **0.96**.

---

### `04_export_head_onnx.py`

Exports the trained head to ONNX for serving.

```bash
python scripts/04_export_head_onnx.py \
    --checkpoint ./features/classifier_head_best.pt \
    --label_map  ./features/label_map.json \
    --output     ./bird_classifier.onnx
```

- Input: `embedding` — float32 (batch, 1024)
- Output: `logits` — float32 (batch, 61)
- Also writes `bird_classifier.classes.json` — `{index: class_name}`

**Note:** requires `protobuf>=4.21` and `onnx>=1.15`. If you hit a protobuf `runtime_version` ImportError, run `pip install --upgrade protobuf`.

---

## Model

| Property | Value |
|---|---|
| Backbone | BirdNET-Analyzer (frozen TFLite, via birdnetlib) |
| Backbone output | 1024-dim embedding per ~3s audio segment |
| Head | 2-layer MLP |
| Head size | ~a few hundred KB (vs 31 MB EfficientNet-B2) |
| Loss | Focal loss γ=2.0 + inverse-freq class weights |
| Classes | 61 |
| Macro F1 | 0.96 |
| ONNX opset | 17 |

---

## Web App

### Backend — `web/app/main.py`

FastAPI app. Loads model and BirdNET analyzer once at startup (slow, ~seconds).

**Endpoints:**
- `GET /` — serves `static/index.html`
- `POST /predict` — accepts `.wav` or `.mp3` up to 100 MB, returns detection JSON

**Run:**
```bash
cd web/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Startup log to expect:**
```
Labels loaded.
load model True
Model loaded.
Meta model loaded.
Application startup complete.
```

### Frontend — `web/app/static/index.html`

Single-file vanilla JS UI. Features:
- Drag-and-drop or click-to-upload (`.wav`/`.mp3`)
- In-browser microphone recording with live timer
- Species cards sorted by detection count, top-3 colour-coded (gold/silver/bronze)
- Wikipedia extract for top species (fetched client-side, fails silently)

---

## Inference Logic

### Detection threshold

Each ~3s BirdNET segment independently produces a probability vector. A species is **detected in a segment** if its probability ≥ `DETECTION_THRESHOLD`.

### Minimum detections filter

A species is **included in the response** only if it was detected in ≥ `MIN_DETECTIONS` segments. This prevents single-segment false positives.

### Response shape

```json
{
  "total_segments": 12,
  "species_found": 3,
  "detections": [
    {
      "species": "House Sparrow",
      "detections": 9,
      "confidence": 0.847,
      "peak_confidence": 0.923
    }
  ]
}
```

---

## Tuning Knobs

All in `web/app/main.py`:

| Constant | Default | Effect |
|---|---|---|
| `DETECTION_THRESHOLD` | `0.4` | Raise to reduce false positives; lower to catch quieter/distant calls |
| `MIN_DETECTIONS` | `2` | Raise for longer recordings (e.g. 3–4 for >2 min audio) to require more evidence |
| `MAX_UPLOAD_BYTES` | `100 MB` | Max file size accepted |

---

## Species List

61 classes currently in the model:

| # | Species |
|---|---|
| 0 | Alexandrine Parakeet |
| 1 | Andaman Coucal |
| 2 | Andaman Drongo |
| 3 | Asian Barred Owlet |
| 4 | Asian Palm Swift |
| 5 | Baikal Bush Warbler |
| 6 | Black-bellied Plover |
| 7 | Blue-winged Leafbird |
| 8 | Cachar Bulbul |
| 9 | Chestnut-capped Babbler |
| 10 | Chestnut-tailed Starling |
| 11 | Chinspot Wren-Babbler |
| 12 | Cinnamon Bittern |
| 13 | Citrine Wagtail |
| 14 | Collared Kingfisher |
| 15 | Grey-cheeked Tit |
| 16 | Grey-throated Martin |
| 17 | Grey Peacock-Pheasant |
| 18 | Hill Partridge |
| 19 | Hume's Bar-tailed Scimitar Babbler |
| 20 | Indian Cuckoo |
| 21 | Jerdon's Leafbird |
| 22 | Jungle Myna |
| 23 | Large-billed Blue Flycatcher |
| 24 | Lineated Barbet |
| 25 | Long-billed Wren-Babbler |
| 26 | Long-tailed Shrike |
| 27 | Mrs Gould's Sunbird |
| 28 | Oriental Dollarbird |
| 29 | Pale-chinned Blue Flycatcher |
| 30 | Pale Blue Flycatcher |
| 31 | Plain Flowerpecker |
| 32 | Puff-throated Babbler |
| 33 | Pygmy Cupwing |
| 34 | Red-faced Liocichla |
| 35 | Ruddy Kingfisher |
| 36 | Scarlet-backed Flowerpecker |
| 37 | Slender-billed Babbler |
| 38 | Spot-breasted Parrotbill |
| 39 | Streak-breasted Scimitar Babbler |
| 40 | Streaked Spiderhunter |
| 41 | Tickell's Leaf Warbler |
| 42 | Vernal Hanging Parrot |
| 43 | White-tailed Flycatcher |
| 44 | Yellow-bellied Fantail |
| 45 | Yellow-browed Warbler |
| 46 | Yellow-throated Leaf Warbler |
| 47 | Yellow-vented Flowerpecker |
| 48 | Asian Koel *(urban)* |
| 49 | Bank Myna *(urban)* |
| 50 | Barn Swallow *(urban)* |
| 51 | Common Myna *(urban)* |
| 52 | Eurasian Collared Dove *(urban)* |
| 53 | House Crow *(urban)* |
| 54 | House Sparrow *(urban)* |
| 55 | Large-billed Crow *(urban)* |
| 56 | Oriental Magpie-Robin *(urban)* |
| 57 | Purple Sunbird *(urban)* |
| 58 | Red-vented Bulbul *(urban)* |
| 59 | Red-whiskered Bulbul *(urban)* |
| 60 | Rose-ringed Parakeet *(urban)* |

*Note: indices 48–60 are approximate — actual indices depend on alphabetical sort order in `label_map.json`. Check `bird_classifier.classes.json` for ground truth.*

---

## Known Issues & Decisions

**`Grey-throated_Babbler` excluded** — BirdNET species identity check (`check_species_identity.py`) found the recordings were misidentified; moved to `_excluded/`.

**Recording format caveat** — browser `MediaRecorder` doesn't guarantee `.wav` output on all browsers (Chrome uses WebM/Opus). The backend accepts `.mp3` and `.wav` by extension check; if a browser recording fails, it's likely a MIME mismatch. Fix: either accept `audio/webm` server-side or convert client-side with Web Audio API before upload.

**`bird_classifier.onnx.data`** — the ONNX export produces an external data file alongside the `.onnx` for large models. Both files must be present and co-located. The symlinks in `web/` cover this.

**protobuf conflict** — `onnx>=1.15` requires `protobuf>=4.21`. If `onnx` import fails with `cannot import name 'runtime_version'`, run `pip install --upgrade protobuf`.

**`MIN_DETECTIONS=2` is aggressive for short clips** — a 5s clip produces only 1–2 BirdNET segments, so almost nothing passes the filter. Lower to `1` for clips under 10s, or skip the filter entirely and fall back to top-K per-segment.

---

## Quickstart

### First-time setup

```bash
git clone https://github.com/Greatmax-07/Bird-Detection-using-Audio
cd Bird-Detection-using-Audio
git checkout framework-based-approach

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# also needs ffmpeg: sudo apt install ffmpeg
```

### Re-train from scratch (if dataset changes)

```bash
# 1. clean dataset
python scripts/00_prepare_ibc53.py --data_dir ./ibc53 --delete-dropped

# 2. download more data for weak species
python scripts/01_download_weak_species.py --out_dir ./xc_downloads --api_key $XC_API_KEY
python scripts/01_download_weak_species.py --out_dir ./xc_urban --api_key $XC_API_KEY

# 3. extract embeddings
python scripts/02_extract_birdnet_embeddings.py \
    --data_dirs ./ibc53 ./xc_downloads ./xc_urban \
    --out_dir ./features

# 4. train head
python scripts/03_train_classifier_head.py --features_dir ./features

# 5. export
python scripts/04_export_head_onnx.py \
    --checkpoint ./features/classifier_head_best.pt \
    --label_map  ./features/label_map.json \
    --output     ./bird_classifier.onnx
```

### Run the web app

```bash
cd web/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000
```

### Add a new species

1. Add the common name to `TARGET_SPECIES` in `01_download_weak_species.py`
2. Run steps 1 → 5 above
3. Update the species count in `web/app/static/index.html` header
