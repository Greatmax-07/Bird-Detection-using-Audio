import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / ".." / "bird_classifier.onnx"
CLASSES_PATH = BASE_DIR / ".." / "bird_classifier.classes.json"

SAMPLE_RATE = 22050
CHUNK_SECONDS = 5
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 50
FMAX = 11025
TARGET_FRAMES = 216
NORM_MEAN = -46.8267
NORM_STD = 16.7479
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


@app.on_event("startup")
def load_model() -> None:
    global session, input_name, classes
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)


def compute_spectrogram(chunk: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )
    db = librosa.power_to_db(mel)
    norm = (db - NORM_MEAN) / NORM_STD

    if norm.shape[1] > TARGET_FRAMES:
        norm = norm[:, :TARGET_FRAMES]
    elif norm.shape[1] < TARGET_FRAMES:
        norm = np.pad(norm, ((0, 0), (0, TARGET_FRAMES - norm.shape[1])), mode="constant")

    return norm.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def chunk_audio(y: np.ndarray) -> list[np.ndarray]:
    if len(y) < CHUNK_SAMPLES:
        y = np.pad(y, (0, CHUNK_SAMPLES - len(y)))

    n_chunks = int(np.ceil(len(y) / CHUNK_SAMPLES))
    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        chunk = y[start:end]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)
    return chunks


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

    try:
        y, _ = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return JSONResponse(status_code=422, content={"error": "Could not process audio"})

    if y is None or len(y) == 0:
        return JSONResponse(status_code=422, content={"error": "Could not process audio"})

    try:
        chunks = chunk_audio(y)
        all_probs = []
        for chunk in chunks:
            spec = compute_spectrogram(chunk)  # (128, 216)
            arr = np.repeat(spec[np.newaxis, np.newaxis, :, :], 3, axis=1)  # (1, 3, 128, 216)
            arr = arr.astype(np.float32)
            outputs = session.run(None, {input_name: arr})
            logits = np.asarray(outputs[0])  # (1, n_classes)
            probs = softmax(logits)[0]
            all_probs.append(probs)
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
