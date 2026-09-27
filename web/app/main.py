import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import tensorflow as tf


# ============================================================
# TensorFlow Lite compatibility patch
# ============================================================

orig_init = tf.lite.Interpreter.__init__


def patched_init(self, *args, **kwargs):
    kwargs["experimental_preserve_all_tensors"] = True
    orig_init(self, *args, **kwargs)


tf.lite.Interpreter.__init__ = patched_init


# ============================================================
# BirdNET
# ============================================================

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


# ============================================================
# FastAPI
# ============================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# web/app/main.py
#        ↑
# BASE_DIR = web/app
#
# Therefore:
# BASE_DIR / ".." / ".." = project root
#
# If bird_classifier.onnx is in the project root, use this.
PROJECT_ROOT = BASE_DIR.parent.parent

MODEL_PATH = PROJECT_ROOT / "bird_classifier.onnx"
CLASSES_PATH = PROJECT_ROOT / "bird_classifier.classes.json"


# ============================================================
# Configuration
# ============================================================

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TOP_K = 5


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(title="BirdEar")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global model objects
# ============================================================

session: Optional[ort.InferenceSession] = None
input_name: Optional[str] = None

classes: dict[str, str] = {}

analyzer: Optional[Analyzer] = None


# ============================================================
# Startup — load models ONCE
# ============================================================

@app.on_event("startup")
def load_model() -> None:
    global session
    global input_name
    global classes
    global analyzer

    print()
    print("=" * 60)
    print("BirdEar startup")
    print("=" * 60)

    # --------------------------------------------------------
    # Check classifier model
    # --------------------------------------------------------

    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Classifier   : {MODEL_PATH}")
    print(f"[INFO] Classes      : {CLASSES_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found:\n{MODEL_PATH}"
        )

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class map not found:\n{CLASSES_PATH}"
        )

    print(f"[INFO] Model size   : {MODEL_PATH.stat().st_size / 1024:.2f} KB")

    # --------------------------------------------------------
    # Load ONNX classifier
    # --------------------------------------------------------

    print("[INFO] Loading ONNX classifier...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if not inputs:
        raise RuntimeError("ONNX model has no inputs.")

    if not outputs:
        raise RuntimeError("ONNX model has no outputs.")

    input_name = inputs[0].name

    print("[INFO] ONNX classifier loaded successfully.")
    print(f"[INFO] Input name   : {input_name}")
    print(f"[INFO] Input shape  : {inputs[0].shape}")
    print(f"[INFO] Input type   : {inputs[0].type}")
    print(f"[INFO] Output name  : {outputs[0].name}")
    print(f"[INFO] Output shape : {outputs[0].shape}")

    # --------------------------------------------------------
    # Load class mapping
    # --------------------------------------------------------

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    print(f"[INFO] Classes loaded: {len(classes)}")

    if len(classes) != 48:
        print(
            f"[WARN] Expected 48 classes, "
            f"but class map contains {len(classes)}."
        )

    # --------------------------------------------------------
    # Load BirdNET ONCE
    # --------------------------------------------------------

    print("[INFO] Loading BirdNET...")

    analyzer = Analyzer()

    # This is important for the BirdNET embedding extraction
    # used by this project.
    analyzer.interpreter.allocate_tensors()

    print("[INFO] BirdNET loaded successfully.")

    print("=" * 60)
    print("BirdEar startup complete")
    print("=" * 60)
    print()


# ============================================================
# Softmax
# ============================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Convert classifier logits into probabilities.
    """

    x = x - np.max(x, axis=-1, keepdims=True)

    e = np.exp(x)

    return e / np.sum(e, axis=-1, keepdims=True)


# ============================================================
# Frontend
# ============================================================

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    if analyzer is None:
        return JSONResponse(
            status_code=500,
            content={"error": "BirdNET model is not loaded."},
        )

    if session is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Classifier model is not loaded."},
        )

    if input_name is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Classifier input is not initialized."},
        )

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    filename = (file.filename or "").lower()

    if not (
        filename.endswith(".wav")
        or filename.endswith(".mp3")
        or filename.endswith(".webm")
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Could not process audio. "
                         "Only WAV, MP3, and WebM files are supported."
            },
        )

    # --------------------------------------------------------
    # Read uploaded audio
    # --------------------------------------------------------

    raw = await file.read()

    print()
    print("-" * 60)
    print("[PREDICT] New audio received")
    print(f"[PREDICT] Filename : {file.filename}")
    print(f"[PREDICT] Size     : {len(raw) / 1024:.2f} KB")

    if len(raw) == 0:
        return JSONResponse(
            status_code=422,
            content={"error": "Uploaded audio file is empty."},
        )

    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=422,
            content={"error": "Audio file is too large."},
        )

    # --------------------------------------------------------
    # Determine temporary file extension
    # --------------------------------------------------------

    if filename.endswith(".wav"):
        suffix = ".wav"
    elif filename.endswith(".mp3"):
        suffix = ".mp3"
    else:
        suffix = ".webm"

    tmp_path = None
    converted_path = None
    audio_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded audio temporarily
        # ----------------------------------------------------

        tmp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        )

        tmp_path = Path(tmp_file.name)

        tmp_file.write(raw)
        tmp_file.close()

        print(f"[PREDICT] Temporary file: {tmp_path}")

        # ----------------------------------------------------
        # Convert browser recording: WebM/Opus -> WAV
        # ----------------------------------------------------

        audio_path = tmp_path

        if filename.endswith(".webm"):

            print("[PREDICT] Browser recording detected.")
            print("[PREDICT] Converting WebM/Opus -> WAV with FFmpeg...")

            if shutil.which("ffmpeg") is None:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": (
                            "FFmpeg is not installed or not available "
                            "in PATH. Install FFmpeg and restart the server."
                        )
                    },
                )

            converted_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            )
            converted_path = Path(converted_file.name)
            converted_file.close()

            try:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(tmp_path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "48000",
                        "-c:a",
                        "pcm_s16le",
                        str(converted_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                print("[PREDICT] WebM -> WAV conversion successful.")

            except FileNotFoundError:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": (
                            "FFmpeg was not found. Install FFmpeg and "
                            "make sure it is available in PATH."
                        )
                    },
                )

            except subprocess.CalledProcessError as ffmpeg_error:
                print("[ERROR] FFmpeg conversion failed:")
                print(ffmpeg_error.stderr)

                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "Could not convert the recorded audio."
                    },
                )

            audio_path = converted_path

        # ----------------------------------------------------
        # BirdNET
        # ----------------------------------------------------

        print("[PREDICT] Extracting BirdNET embeddings...")

        recording = Recording(
            analyzer,
            str(audio_path),
        )

        recording.extract_embeddings()

        print(
            f"[PREDICT] BirdNET segments: "
            f"{len(recording.embeddings)}"
        )

        if not recording.embeddings:

            print("[WARN] No embeddings extracted.")

            return JSONResponse(
                status_code=422,
                content={
                    "error": "No BirdNET embeddings could be "
                             "extracted from the audio."
                },
            )

        # ----------------------------------------------------
        # Classify every BirdNET segment
        # ----------------------------------------------------

        all_probs = []

        for segment_number, seg in enumerate(
            recording.embeddings,
            start=1,
        ):

            # IMPORTANT:
            # Do NOT use:
            #
            # vec = seg.get("embeddings") or seg.get("embedding")
            #
            # because NumPy arrays cannot safely be evaluated
            # as booleans.

            vec = seg.get("embeddings")

            if vec is None:
                vec = seg.get("embedding")

            if vec is None:
                print(
                    f"[WARN] Segment {segment_number}: "
                    f"no embedding found."
                )
                continue

            # ------------------------------------------------
            # Convert to NumPy
            # ------------------------------------------------

            arr = np.asarray(
                vec,
                dtype=np.float32,
            )

            # Make sure it is exactly one 1024-D vector.
            arr = arr.reshape(1, -1)

            print(
                f"[PREDICT] Segment {segment_number}: "
                f"embedding shape = {arr.shape}"
            )

            # ------------------------------------------------
            # Validate embedding dimension
            # ------------------------------------------------

            if arr.shape[1] != 1024:

                print(
                    f"[ERROR] Expected 1024-dimensional "
                    f"embedding, got {arr.shape[1]}"
                )

                return JSONResponse(
                    status_code=500,
                    content={
                        "error": (
                            "BirdNET embedding dimension mismatch. "
                            f"Expected 1024, got {arr.shape[1]}."
                        )
                    },
                )

            # ------------------------------------------------
            # Run trained classifier
            # ------------------------------------------------

            outputs = session.run(
                None,
                {
                    input_name: arr
                },
            )

            logits = np.asarray(
                outputs[0],
                dtype=np.float32,
            )

            print(
                f"[PREDICT] Segment {segment_number}: "
                f"classifier output shape = {logits.shape}"
            )

            # ------------------------------------------------
            # Convert logits → probabilities
            # ------------------------------------------------

            probs = softmax(logits)[0]

            all_probs.append(probs)

        # ----------------------------------------------------
        # Make sure at least one segment worked
        # ----------------------------------------------------

        if not all_probs:

            print(
                "[WARN] No valid classifier predictions "
                "were generated."
            )

            return JSONResponse(
                status_code=422,
                content={
                    "error": "Could not compute predictions."
                },
            )

        # ----------------------------------------------------
        # Average predictions across audio segments
        # ----------------------------------------------------

        avg_probs = np.mean(
            np.stack(all_probs, axis=0),
            axis=0,
        )

        print(
            f"[PREDICT] Averaged {len(all_probs)} "
            f"segment predictions."
        )

        # ----------------------------------------------------
        # Top-K predictions
        # ----------------------------------------------------

        top_indices = np.argsort(
            avg_probs
        )[::-1][:TOP_K]

        predictions = []

        for idx in top_indices:

            idx = int(idx)

            species = classes.get(
                str(idx),
                f"Unknown_{idx}",
            )

            predictions.append(
                {
                    "species": species.replace(
                        "_",
                        " ",
                    ),
                    "confidence": float(
                        avg_probs[idx]
                    ),
                }
            )

        # ----------------------------------------------------
        # Print final prediction to terminal
        # ----------------------------------------------------

        print("[PREDICT] Final predictions:")

        for rank, prediction in enumerate(
            predictions,
            start=1,
        ):

            print(
                f"  {rank}. "
                f"{prediction['species']} "
                f"({prediction['confidence'] * 100:.2f}%)"
            )

        print("-" * 60)
        print()

        # ----------------------------------------------------
        # Return JSON to frontend
        # ----------------------------------------------------

        return {
            "predictions": predictions
        }

    except Exception as e:

        import traceback

        print()
        print("[ERROR] Prediction failed:")
        traceback.print_exc()
        print()

        return JSONResponse(
            status_code=422,
            content={
                "error": f"Failed to process audio: {str(e)}"
            },
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary audio file
        # ----------------------------------------------------

        for temporary_path in (tmp_path, converted_path):

            if temporary_path is not None:

                try:

                    if temporary_path.exists():
                        temporary_path.unlink()

                except Exception as cleanup_error:

                    print(
                        f"[WARN] Could not delete temporary file: "
                        f"{cleanup_error}"
                    )


# ============================================================
# Static files
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static",
)