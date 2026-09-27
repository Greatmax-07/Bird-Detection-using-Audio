# BirdEar — Improvement Plan

> Tracks what has been done, what is in progress, and exactly what needs to be built.
> No interpretation required — every item has a specific file, function, and expected output.

---

## Status Legend

- `[DONE]` — completed, in repo
- `[IN PROGRESS]` — partially done
- `[TODO]` — not started, spec below

---

## A. Better Model Quality

### A1. Deeper MLP head `[TODO]`

**File:** `scripts/03_train_classifier_head.py`

**Current architecture:**
```
Linear(1024 → 256) → ReLU → Dropout(0.3) → Linear(256 → N)
```

**New architecture:**
```
Linear(1024 → 512) → BatchNorm1d(512) → ReLU → Dropout(0.3)
→ Linear(512 → 256) → BatchNorm1d(256) → ReLU → Dropout(0.2)
→ Linear(256 → N)
```

**Changes in `03_train_classifier_head.py`:**
- Replace `ClassifierHead.__init__` `self.net` sequential block with the above
- Add `nn.BatchNorm1d` after each linear (before ReLU)
- Keep focal loss and class weights unchanged
- Add `--hidden1 512 --hidden2 256` CLI args so architecture is configurable without editing the file
- Keep `--epochs 40` default but raise to `60` as the deeper net needs more steps to converge

**Same change must be mirrored in `scripts/04_export_head_onnx.py`:**
- `ClassifierHead` class in that file must match exactly — currently it is a copy-paste of the training one
- Add same `--hidden1` and `--hidden2` args; default must match training defaults

**Expected output:** val loss should drop, weak class F1 should improve. Re-run and compare classification report.

---

### A2. Per-species detection threshold `[TODO]`

**File:** `web/app/main.py`

**Problem:** Global `DETECTION_THRESHOLD = 0.4` and `MIN_DETECTIONS = 2` treats Indian Peafowl (6065 embeddings, very distinct call) the same as Cinnamon Bittern (40 embeddings, easily confused). Result: common species over-trigger, rare species under-trigger.

**Solution:** A threshold config dict keyed by species name. Three tiers:

```python
# In web/app/main.py, replace the two global constants with:

THRESHOLD_HIGH = 0.55    # rare/easily confused species
THRESHOLD_MED  = 0.40    # default
THRESHOLD_LOW  = 0.30    # very distinctive, high-data species

# Species with very distinctive calls + high embedding counts → lower threshold
LOW_THRESHOLD_SPECIES = {
    "Asian Koel", "Indian Peafowl", "Common Hawk-Cuckoo",
    "Malabar Whistling Thrush", "Greater Coucal", "Indian Cuckoo",
    "Common Iora", "Black Drongo", "Red-vented Bulbul",
    "Red-whiskered Bulbul", "Purple Sunbird",
}

# Rare/low-data/easily confused species → higher threshold
HIGH_THRESHOLD_SPECIES = {
    "Cinnamon Bittern", "Black-bellied Plover", "Little Egret",
    "Indian Pond Heron", "White-eyed Buzzard", "Changeable Hawk-Eagle",
    "Malabar Trogon", "Stork-billed Kingfisher", "Blue-tailed Bee-eater",
    "Oriental Dollarbird", "Grey-throated Martin",
}

def get_threshold(species_name: str) -> float:
    if species_name in LOW_THRESHOLD_SPECIES:
        return THRESHOLD_LOW
    if species_name in HIGH_THRESHOLD_SPECIES:
        return THRESHOLD_HIGH
    return THRESHOLD_MED
```

**In the `/predict` endpoint:** replace `if p >= DETECTION_THRESHOLD` with `if p >= get_threshold(species)`.

**MIN_DETECTIONS** stays global at `2` for now — revisit after field testing.

---

### A3. Confidence calibration `[TODO]`

**File:** new script `scripts/05_calibrate.py`

**Problem:** Raw softmax outputs from a focal-loss model are not calibrated probabilities. A score of 0.82 does not mean 82% likely. This makes the UI confidence numbers misleading.

**Method:** Temperature scaling — single scalar T divides logits before softmax. Fit T on the val set after training, save it, apply at inference time.

**Script `scripts/05_calibrate.py` must:**
1. Load `features/X.npy`, `features/y.npy`, `features/label_map.json`
2. Load `features/classifier_head_best.pt`
3. Do the same stratified split as `03_train_classifier_head.py` (same `random_state=42`, same `val_frac=0.2`) to get the val set
4. Optimise T using `torch.optim.LBFGS` to minimise NLL on val logits
5. Save T to `features/temperature.json` as `{"temperature": <float>}`
6. Print ECE (expected calibration error) before and after

**Changes in `web/app/main.py`:**
- At startup, load `features/temperature.json` if present; default T=1.0 if missing
- In softmax: divide logits by T before exp: `x = logits / T`

**Changes in `scripts/04_export_head_onnx.py`:**
- The ONNX export does NOT bake T in — calibration happens in Python at inference time, not inside ONNX, because T may be re-fit without re-exporting the model

---

### A4. Xeno-canto quality filtering `[TODO]`

**File:** `scripts/01_download_weak_species.py`

**Problem:** Currently downloads all quality grades (A/B/C/D/E). Grade D/E recordings are often background noise, wind, or misidentified species, which corrupts the training data.

**Change:** Add `--min_quality` arg (default `B`). In `download_recording`, before writing the file, check `rec.get("q")` and skip if below threshold.

```python
QUALITY_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "no score": 0}

def quality_ok(rec: dict, min_quality: str) -> bool:
    return QUALITY_RANK.get(rec.get("q", "no score"), 0) >= QUALITY_RANK[min_quality]
```

Skip the file and still write its row to `metadata.csv` with a `skipped_quality` flag so you can audit later.

**Re-download command after this change:**
```bash
python scripts/01_download_weak_species.py \
    --out_dir ./xc_maharashtra \
    --min_quality B \
    --api_key $XC_API_KEY
```

Note: existing downloaded files are skipped by XC ID check regardless, so this only affects new downloads unless you delete and re-download.

---

## B. Field Recording Mode

### B1. Long audio streaming processing `[TODO]`

**File:** `web/app/main.py`

**Problem:** Current backend loads the entire file into a tempfile, runs `extract_embeddings` on it all at once. Works for short clips. For 30min+ field recordings this is fine memory-wise (birdnetlib chunks internally) but the response has no time information.

**Change to `/predict` endpoint:**

Currently returns:
```json
{"total_segments": 12, "species_found": 3, "detections": [...]}
```

New response shape:
```json
{
  "total_segments": 120,
  "duration_seconds": 1843.2,
  "species_found": 7,
  "detections": [
    {
      "species": "Asian Koel",
      "detections": 14,
      "confidence": 0.82,
      "peak_confidence": 0.94,
      "first_heard_at": 12.0,
      "last_heard_at": 1204.5,
      "timestamps": [12.0, 45.0, 67.5, ...]
    }
  ]
}
```

**Implementation:** `recording.embeddings` already returns `start_time` and `end_time` per segment. Currently ignored. Collect them per species alongside the confidence.

```python
# In the segment loop, change species_detections value type from list[float] to list[dict]:
species_detections[species].append({
    "confidence": float(p),
    "start_time": float(seg.get("start_time", 0)),
})

# When building results:
timestamps = [d["start_time"] for d in detections]
result = {
    "species": ...,
    "detections": len(detections),
    "confidence": mean of confidences,
    "peak_confidence": max confidence,
    "first_heard_at": min(timestamps),
    "last_heard_at": max(timestamps),
    "timestamps": sorted(timestamps),
}
```

**MAX_UPLOAD_BYTES:** raise from 100 MB to 500 MB to support ~60min WAV files.

---

### B2. Field report export `[TODO]`

**File:** `web/app/static/index.html` (frontend only, no backend change)

**After results render**, add a "Download Report" button that generates and downloads a CSV client-side using a Blob:

```
CSV columns: species, detections, avg_confidence, peak_confidence, first_heard_at, last_heard_at
```

Also generate a plain-text summary:
```
BirdEar Field Report
Date: <today's date from JS Date()>
Duration analysed: <X min Y sec>
Species detected: N

1. Asian Koel — 14 detections, first at 0:12, last at 20:04, avg confidence 82%
2. ...
```

Both CSV and TXT downloadable via separate buttons. No backend involvement — pure JS Blob + `URL.createObjectURL`.

---

### B3. Upload UI changes for long recordings `[TODO]`

**File:** `web/app/static/index.html`

- Remove the duration display line that just shows seconds — for a 45min file this is useless
- Add a progress indicator during processing: since fetch has no progress event for the response body, show a pulsing "Analysing X segments..." message that updates every 2 seconds using a JS interval that increments a fake counter (honest: say "Analysing audio, this may take a minute for long recordings")
- Increase the file size hint text from "up to 100 MB" to "up to 500 MB"
- After results, show total duration analysed in the subtitle: "7 species across 43 minutes of audio"

---

## E. UX Improvements

### E1. Spectrogram view with detection markers `[TODO]`

**Files:** `web/app/main.py` (no change needed), `web/app/static/index.html`

**What to build:** After results render, show a horizontal timeline bar (pure canvas/SVG, no library) representing the full audio duration. Each detection timestamp gets a coloured tick mark, colour-coded by species (up to 10 colours, cycle after that). Hovering a tick shows species name + confidence in a tooltip.

**Implementation:**
- Draw on an HTML5 `<canvas>` element, width = container width, height = 60px
- X axis = time (0 to duration_seconds)
- For each species, pick a colour from a fixed palette
- For each timestamp in `detections[i].timestamps`, draw a vertical line at `(t / duration_seconds) * canvas.width`
- Draw a legend below: coloured dot + species name

**No audio waveform** — that would require Web Audio API decoding of the whole file client-side which is slow. Just the detection timeline.

---

### E2. Bird info cards `[TODO]`

**File:** `web/app/static/index.html`

**Current behaviour:** clicking a species card does nothing.

**New behaviour:** clicking expands an inline info panel below the card showing:
- Wikipedia extract (already fetched for top species — extend to all species on click, lazy fetch)
- eBird species page link: `https://ebird.org/species/<ebird_code>` — need a static mapping of species name → eBird code. Build this as a JSON object in the HTML for the current species list.
- Xeno-canto page link: `https://xeno-canto.org/explore?query=en%3A"<species name>"` — can be constructed dynamically from species name, no static map needed

**Expand/collapse:** clicking the card again collapses it. Only one card open at a time.

**No backend change required** — all client-side.

---

### E3. "Report a mistake" button `[TODO]`

**Files:** `web/app/static/index.html`, `web/app/main.py`

**Frontend:** Add a small flag icon button on each species card. Clicking opens a small inline form:
```
[ ] Wrong species — what did you hear? [text input]
[ ] Not present at all
[ ] Correct but low confidence
[Submit]
```

**Backend:** New endpoint `POST /feedback`:
```python
@app.post("/feedback")
async def feedback(payload: dict):
    # append to feedback_log.jsonl in BASE_DIR
    # format: {"ts": ISO timestamp, "species": str, "issue": str, "correct_species": str|null}
```

Save to `feedback_log.jsonl` (one JSON object per line, append mode). This file becomes a correction dataset for the next retrain — wrong predictions with correct labels are gold.

**No UI for viewing feedback yet** — just collection. Review manually with `cat feedback_log.jsonl | python -m json.tool`.

---

## Implementation Order

Do these in this exact order — each builds on the previous:

1. **A4** — quality filter in download script. Re-download, re-extract, retrain. Clean data first.
2. **A1** — deeper MLP. Retrain after A4's data is ready.
3. **A3** — calibration script. Run after A1's new checkpoint is saved.
4. **A2** — per-species threshold in `main.py`. Uses species names, no model change needed.
5. **B1** — timestamp output from `/predict`. Backend only, no retrain.
6. **B2 + B3** — field report UI + long recording UX. Frontend only.
7. **E1** — spectrogram timeline. Frontend only, depends on B1's timestamp data.
8. **E2** — bird info cards. Frontend only, independent.
9. **E3** — feedback endpoint + UI. Backend + frontend, independent.

---

## Files Changed Per Item

| Item | Files Modified | Files Created |
|---|---|---|
| A1 | `scripts/03_train_classifier_head.py`, `scripts/04_export_head_onnx.py` | — |
| A2 | `web/app/main.py` | — |
| A3 | `web/app/main.py` | `scripts/05_calibrate.py`, `features/temperature.json` |
| A4 | `scripts/01_download_weak_species.py` | — |
| B1 | `web/app/main.py` | — |
| B2 | `web/app/static/index.html` | — |
| B3 | `web/app/static/index.html` | — |
| E1 | `web/app/static/index.html` | — |
| E2 | `web/app/static/index.html` | — |
| E3 | `web/app/main.py`, `web/app/static/index.html` | `feedback_log.jsonl` |

---

## Current State Snapshot

- Classes: 93 (post-NE/Andaman cleanup, pre-non-MH drop — actual MH-relevant count TBD after next retrain)
- Embeddings: 67,513 vectors across 93 classes
- Backbone: BirdNET-Analyzer (frozen), 1024-dim
- Head: Linear(1024→256)→ReLU→Dropout(0.3)→Linear(256→93)
- Loss: Focal loss γ=2.0 + inverse-freq class weights
- ONNX opset: 17
- Model size: ~1.1 MB
- Serving: FastAPI + uvicorn, ONNX Runtime CPU
- Detection: per-segment threshold 0.4, min 2 segments
- UI: single-page vanilla JS, upload + record, species cards, no timeline
