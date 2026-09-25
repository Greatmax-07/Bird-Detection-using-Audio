# BirdEar 🐦

**Identify Indian bird species from audio — 48 species, transfer-learned from BirdNET, served over a web UI.**

<p>
  <img alt="species" src="https://img.shields.io/badge/species-48-4f9cf9">
  <img alt="macro f1" src="https://img.shields.io/badge/macro%20F1-0.96-4f9cf9">
  <img alt="backbone" src="https://img.shields.io/badge/backbone-BirdNET%20(frozen)-4f9cf9">
  <img alt="serving" src="https://img.shields.io/badge/serving-FastAPI%20%2B%20ONNX%20Runtime-4f9cf9">
</p>

Upload a clip (or record one in-browser) → get the top-5 species with confidence scores.

---

## Why this exists

The original dataset (iBC53, 48 Indian bird species after cleanup) is heavily long-tailed:

| | |
|---|---|
| Largest class | 1,300+ clips (Puff-throated Babbler) |
| Smallest classes | under 20 clips (Black-bellied Plover, Cinnamon Bittern, Chinspot Wren-Babbler...) |
| Ratio | ~90:1 |

A CNN trained from scratch on this simply never learns the tail. The fix here is two-pronged: **top up the rarest classes with more real recordings**, and **stop training a spectrogram CNN from scratch** in favor of a frozen, globally-pretrained bioacoustic backbone.

```mermaid
flowchart LR
    A["Long-tailed dataset\n8–1300+ clips/class"] --> B["Xeno-canto top-up\nfor 21 weak/moderate species"]
    B --> C["BirdNET embeddings\n(frozen, pretrained)"]
    C --> D["Small MLP head\nfocal loss + class weights"]
    D --> E["0.96 macro F1\nacross 48 species"]
    style A fill:#1e2130,stroke:#f9b04f,color:#fff
    style B fill:#1e2130,stroke:#f9b04f,color:#fff
    style C fill:#1e2130,stroke:#4f9cf9,color:#fff
    style D fill:#1e2130,stroke:#4f9cf9,color:#fff
    style E fill:#1e2130,stroke:#4f9cf9,color:#fff
```

## Results

| Metric | Before (from-scratch CNN, imbalanced) | After (BirdNET embeddings + focal loss) |
|---|---|---|
| Weak-tier species usable? | No — effectively ignored | Yes — 0.82–1.00 F1 |
| Macro F1 (48 classes) | not meaningfully measurable | **0.96** |
| Model size (classifier head) | ~31 MB (EfficientNet-B2) | ~a few hundred KB (MLP) |

Full per-class precision/recall/F1 is in [`features/`](./features) after running the training script — see below.

## Pipeline

<img src="./diagrams/pipeline.svg" alt="Data and training pipeline: raw iBC53 dataset is cleaned and relabeled, augmented with Xeno-canto recordings for weak species, converted into BirdNET embeddings, and used to train a small focal-loss MLP classifier head that is exported to ONNX." width="100%">

```mermaid
flowchart TD
    A[Raw iBC53\nscientific-name folders] -->|00_prepare_ibc53.py| B[Common-name folders\n48 clean classes]
    B --> C{Weak-tier\nspecies?}
    C -->|yes, 21 species| D[01_download_weak_species.py\nXeno-canto API v3]
    C -->|no| E
    D --> E[02_extract_birdnet_embeddings.py\nfrozen BirdNET backbone]
    E --> F[features/\nX.npy, y.npy, label_map.json]
    F -->|03_train_classifier_head.py| G[Focal loss + class-weighted MLP]
    G -->|04_export_head_onnx.py| H[bird_classifier.onnx\n+ classes.json]
    H --> I[web/app\nFastAPI + ONNX Runtime]
```

## Architecture (inference)

<img src="./diagrams/architecture.svg" alt="Inference architecture: browser uploads audio to FastAPI, which extracts BirdNET embeddings, runs them through the ONNX classifier head, and returns top-5 predictions." width="100%">

```mermaid
sequenceDiagram
    participant U as Browser
    participant S as FastAPI (main.py)
    participant B as BirdNET (birdnetlib)
    participant O as ONNX classifier head

    U->>S: POST /predict (audio file)
    S->>B: extract_embeddings(tmp file)
    B-->>S: 1024-dim vector per ~3s segment
    loop each segment
        S->>O: run(embedding)
        O-->>S: logits → softmax
    end
    S->>S: average probabilities across segments
    S-->>U: top-5 species + confidence
```

## Project structure

```
.
├── ibc53/                      # cleaned, common-name-labeled training audio (48 classes)
├── xc_downloads/                # extra Xeno-canto recordings for weak-tier species
├── features/                    # BirdNET embeddings + trained classifier head
├── scripts/
│   ├── 00_prepare_ibc53.py       # scientific → common name, dedupe, drop excluded classes
│   ├── 01_download_weak_species.py   # top up rare species from Xeno-canto
│   ├── 02_extract_birdnet_embeddings.py  # frozen BirdNET → 1024-dim embeddings
│   ├── 03_train_classifier_head.py   # focal loss + class-weighted MLP training
│   └── 04_export_head_onnx.py        # export trained head to ONNX for serving
├── web/
│   └── app/
│       ├── main.py               # FastAPI backend (BirdNET + ONNX inference)
│       └── static/index.html     # upload / record UI
├── bird_classifier.onnx          # deployed classifier head
└── bird_classifier.classes.json  # index → species name
```

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. clean + relabel the raw dataset
python scripts/00_prepare_ibc53.py --data_dir ./ibc53 --delete-dropped

# 3. (optional) top up weak-tier species with more Xeno-canto recordings
python scripts/01_download_weak_species.py --api_key $XC_API_KEY

# 4. extract BirdNET embeddings for the full dataset
python scripts/02_extract_birdnet_embeddings.py \
    --data_dirs ./ibc53 ./xc_downloads \
    --out_dir ./features

# 5. train the classifier head
python scripts/03_train_classifier_head.py --features_dir ./features

# 6. export to ONNX for serving
python scripts/04_export_head_onnx.py \
    --checkpoint ./features/classifier_head_best.pt \
    --label_map  ./features/label_map.json \
    --output     ./bird_classifier.onnx

# 7. run the web app
cd web/app && uvicorn main:app --reload
```

## Species covered

48 Indian bird species — see [`bird_classifier.classes.json`](./bird_classifier.classes.json) for the full list.

## Credits

- [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) (K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology) — pretrained bioacoustic backbone, used via [birdnetlib](https://github.com/joeweiss/birdnetlib)
- [Xeno-canto](https://xeno-canto.org) — supplementary recordings for underrepresented species
- iBC53 — base training dataset
