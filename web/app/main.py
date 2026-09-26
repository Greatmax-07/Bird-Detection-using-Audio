import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / ".." / "bird_classifier.onnx"
CLASSES_PATH = BASE_DIR / ".." / "bird_classifier.classes.json"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB for longer recordings
DETECTION_THRESHOLD = 0.4  # minimum per-segment confidence to count as a detection
MIN_DETECTIONS = 2          # species must appear in at least this many segments

app = FastAPI(title="BirdEar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session: Optional[ort.InferenceSession] = None
input_name: Optional[str] = None
classes: dict[str, str] = {}
analyzer: Optional[Analyzer] = None


@app.on_event("startup")
def load_model() -> None:
    global session, input_name, classes, analyzer
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
    analyzer = Analyzer()


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not (filename.endswith(".wav") or filename.endswith(".mp3")):
        return JSONResponse(status_code=422, content={"error": "Could not process audio"})

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=422, content={"error": "File too large (max 100 MB)"})

    suffix = ".wav" if filename.endswith(".wav") else ".mp3"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()

            recording = Recording(analyzer, tmp.name)
            recording.extract_embeddings()

            if not recording.embeddings:
                return JSONResponse(status_code=422, content={"error": "Could not process audio"})

            # Per-species: list of per-segment peak confidences
            from collections import defaultdict
            species_detections: dict[str, list[float]] = defaultdict(list)
            total_segments = 0

            for seg in recording.embeddings:
                vec = seg.get("embeddings") or seg.get("embedding")
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)[np.newaxis, :]
                outputs = session.run(None, {input_name: arr})
                probs = softmax(np.asarray(outputs[0]))[0]
                total_segments += 1

                # Record every species that crosses threshold in this segment
                for idx, p in enumerate(probs):
                    if p >= DETECTION_THRESHOLD:
                        species = classes.get(str(int(idx)), f"Unknown_{idx}")
                        species_detections[species].append(float(p))

            if total_segments == 0:
                return JSONResponse(status_code=422, content={"error": "Could not process audio"})

    except Exception:
        return JSONResponse(status_code=422, content={"error": "Could not process audio"})

    # Filter to species detected in enough segments, sort by detection count then avg confidence
    results = []
    for species, confs in species_detections.items():
        if len(confs) >= MIN_DETECTIONS:
            results.append({
                "species": species.replace("_", " "),
                "detections": len(confs),
                "confidence": round(sum(confs) / len(confs), 4),
                "peak_confidence": round(max(confs), 4),
            })

    results.sort(key=lambda x: (x["detections"], x["confidence"]), reverse=True)

    return {
        "total_segments": total_segments,
        "species_found": len(results),
        "detections": results,
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
