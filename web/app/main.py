"""
BirdEar API.

    GET  /          single-page UI
    GET  /health    {status, n_classes, window_size, step_size}
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
    AUDIO_EXTS, EMBED_DIM, FEATURES_DIR, POOLED_DIM, WINDOW_SIZE, WINDOW_STEP,
    class_names_from_label_map, extract_segments, file_sha256, load_analyzer,
    load_label_map, pool_segments, read_meta, window_ranges,
)

log = logging.getLogger("birdear")
logging.basicConfig(level=os.environ.get("BIRDEAR_LOG", "INFO"))

STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TOP_K = 5                                   # recording-level predictions returned
TOP_PER_WINDOW = 5                          # classes considered per window
WINDOW_THRESHOLD = float(os.environ.get("BIRDEAR_WINDOW_THRESHOLD", 0.20))      # a species must reach this somewhere
WINDOW_THRESHOLD_LOW = float(os.environ.get("BIRDEAR_WINDOW_THRESHOLD_LOW", 0.10))  # then windows above this are kept
MERGE_GAP_SECONDS = 2.0                     # merge detections closer than this
OCCURRENCE_GAP_SECONDS = 15.0               # events further apart count as new occurrence
COUNT_NOTE = ("Approximate acoustic occurrence count. "
              "The same bird may produce multiple occurrences.")

state: dict = {}
_birdnet_lock = threading.Lock()            # the TFLite interpreter is not thread-safe


# ----------------------------------------------------------------------------
# startup
# ----------------------------------------------------------------------------

def _load_state() -> None:
    rec_path = FEATURES_DIR / "recording_svm.pkl"
    win_path = FEATURES_DIR / "window_svm.pkl"
    lm_path = FEATURES_DIR / "label_map.json"
    missing = [p for p in (rec_path, win_path, lm_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {[str(p) for p in missing]}. Train first: scripts/02, 03 and 04 (see README).")

    meta = read_meta()
    sha = file_sha256(lm_path)
    for section in ("recording", "window"):
        if meta.get(section, {}).get("label_map_sha256") != sha:
            raise RuntimeError(
                f"{section} model was not trained with the current features/label_map.json "
                f"(species/ids differ). Re-run scripts/03 and 04 after script 02.")

    names = class_names_from_label_map(load_label_map())
    recording_model = joblib.load(rec_path)
    window_model = joblib.load(win_path)
    for label, m in (("recording", recording_model), ("window", window_model)):
        if m.n_features_in_ != POOLED_DIM:
            raise RuntimeError(f"{label} SVM expects {m.n_features_in_} features, need {POOLED_DIM}")
        if len(m.classes_) > len(names):
            raise RuntimeError(f"{label} SVM has more classes than label_map.json")

    w = meta["window"]
    state.update(
        names=names, recording_model=recording_model, window_model=window_model,
        window_size=int(w.get("window_size", WINDOW_SIZE)),
        step_size=int(w.get("step_size", WINDOW_STEP)),
        analyzer=load_analyzer(),
    )
    log.info("loaded %d classes, window %d/%d", len(names), state["window_size"], state["step_size"])


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
    names = state["names"]
    return (names[class_id] if 0 <= class_id < len(names) else f"Unknown_{class_id}").replace("_", " ")


def detect_species_events(segments: list) -> list[dict]:
    """Sliding-window classification -> thresholded, merged per-species events."""
    model = state["window_model"]
    emb = np.stack([s[2] for s in segments])
    ranges = window_ranges(len(emb), state["window_size"], state["step_size"])
    pooled = np.stack([pool_segments(emb[a:b]) for a, b in ranges])
    probs = model.predict_proba(pooled)

    detections: dict[str, list[dict]] = {}
    for (a, b), p in zip(ranges, probs):
        for i in np.argsort(p)[::-1][:TOP_PER_WINDOW]:
            if p[i] < WINDOW_THRESHOLD_LOW:
                continue
            detections.setdefault(_species(int(model.classes_[i])), []).append(
                {"start_time": segments[a][0], "end_time": segments[b - 1][1], "confidence": float(p[i])})

    results = []
    for species, dets in detections.items():
        if max(d["confidence"] for d in dets) < WINDOW_THRESHOLD:
            continue  # never confident anywhere in the file: ignore weak-only species
        dets.sort(key=lambda d: d["start_time"])
        events: list[dict] = []
        for d in dets:
            if events and d["start_time"] <= events[-1]["end_time"] + MERGE_GAP_SECONDS:
                cur = events[-1]
                cur["end_time"] = max(cur["end_time"], d["end_time"])
                cur["confidence"] = max(cur["confidence"], d["confidence"])
                cur["windows"] += 1
            else:
                events.append({**d, "windows": 1})
        occurrences = 1 + sum(
            1 for prev, nxt in zip(events, events[1:])
            if nxt["start_time"] - prev["end_time"] > OCCURRENCE_GAP_SECONDS)
        results.append({
            "species": species, "event_count": len(events), "estimated_count": occurrences,
            "count_note": COUNT_NOTE, "confidence": max(e["confidence"] for e in events),
            "events": events,
        })
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


def analyze_file(path: Path) -> dict:
    try:
        with _birdnet_lock:
            segments = extract_segments(state["analyzer"], path)
    except AudioFormatError as e:
        raise ValueError(f"Could not decode the audio file ({e}).") from e
    if not segments:
        raise ValueError("No BirdNET embeddings could be extracted from the audio.")

    model = state["recording_model"]
    probs = model.predict_proba(pool_segments(np.stack([s[2] for s in segments])).reshape(1, -1))[0]
    predictions = [
        {"species": _species(int(model.classes_[i])), "confidence": float(probs[i])}
        for i in np.argsort(probs)[::-1][:TOP_K]
    ]
    detected = detect_species_events(segments)
    return {
        "predictions": predictions,
        "species_detected": detected,
        "total_approximate_count": sum(d["estimated_count"] for d in detected),
        "count_note": "Approximate acoustic-event count; it does not guarantee the number of individual birds.",
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
            "window_size": state["window_size"], "step_size": state["step_size"]}


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
