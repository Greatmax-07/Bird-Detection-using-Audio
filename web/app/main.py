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
