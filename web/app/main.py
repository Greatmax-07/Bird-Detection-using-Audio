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

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TOP_K = 5

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
    # Loaded once at startup — BirdNET's model load is the slow part (~seconds),
    # so this must NOT happen per-request.
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
        return JSONResponse(status_code=422, content={"error": "Could not process audio"})

    suffix = ".wav" if filename.endswith(".wav") else ".mp3"

    try:
        # birdnetlib's Recording needs a real file path, not an in-memory buffer,
        # so the upload is written to a scratch file for the duration of the request.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()

            recording = Recording(analyzer, tmp.name)
            recording.extract_embeddings()

            if not recording.embeddings:
                return JSONResponse(status_code=422, content={"error": "Could not process audio"})

            all_probs = []
            for seg in recording.embeddings:
                vec = seg.get("embeddings") or seg.get("embedding")
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)[np.newaxis, :]  # (1, 1024)
                outputs = session.run(None, {input_name: arr})
                logits = np.asarray(outputs[0])  # (1, n_classes)
                probs = softmax(logits)[0]
                all_probs.append(probs)

            if not all_probs:
                return JSONResponse(status_code=422, content={"error": "Could not process audio"})

            avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    except Exception:
        return JSONResponse(status_code=422, content={"error": "Could not process audio"})

    top_indices = np.argsort(avg_probs)[::-1][:TOP_K]
    predictions = []
    for idx in top_indices:
        species = classes.get(str(int(idx)), f"Unknown_{idx}")
        predictions.append(
            {
                "species": species.replace("_", " "),
                "confidence": float(avg_probs[idx]),
            }
        )

    return {"predictions": predictions}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
