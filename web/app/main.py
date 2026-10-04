import json
import tempfile
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
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

# main.py is:
# project_root/web/app/main.py
#
# Therefore:
# BASE_DIR.parent.parent = project root

PROJECT_ROOT = BASE_DIR.parent.parent

MODEL_PATH = PROJECT_ROOT / "features" / "recording_svm.pkl"
WINDOW_MODEL_PATH = PROJECT_ROOT / "features" / "window_svm.pkl"
LABEL_MAP_PATH = PROJECT_ROOT / "features" / "label_map.json"

# ============================================================
# Configuration
# ============================================================

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TOP_K = 5

BIRDNET_EMBEDDING_DIM = 1024
RECORDING_EMBEDDING_DIM = 3072

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

model = None
window_model = None
classes: dict[int, str] = {}
analyzer: Optional[Analyzer] = None

# ============================================================
# Startup — load models ONCE
# ============================================================

@app.on_event("startup")
def load_model() -> None:
    global model
    global window_model
    global classes
    global analyzer

    print()
    print("=" * 60)
    print("BirdEar startup")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model files
    # --------------------------------------------------------

    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Classifier   : {MODEL_PATH}")
    print(f"[INFO] Label map    : {LABEL_MAP_PATH}")
    print(f"[INFO] Window model : {WINDOW_MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found:\n{MODEL_PATH}"
        )

    if not WINDOW_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Window model not found:\n{WINDOW_MODEL_PATH}"
        )

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Label map not found:\n{LABEL_MAP_PATH}"
        )

    print(
        f"[INFO] Model size   : "
        f"{MODEL_PATH.stat().st_size / 1024:.2f} KB"
    )

    # --------------------------------------------------------
    # Load trained SVMs
    # --------------------------------------------------------

    print("[INFO] Loading recording-level SVM...")

    model = joblib.load(MODEL_PATH)

    print("[INFO] Recording-level SVM loaded successfully.")

    print("[INFO] Loading window-level SVM...")

    window_model = joblib.load(WINDOW_MODEL_PATH)

    print("[INFO] Window-level SVM loaded successfully.")

    # --------------------------------------------------------
    # Validate recording SVM input dimension
    # --------------------------------------------------------

    if hasattr(model, "n_features_in_"):
        print(
            f"[INFO] Model input  : "
            f"{model.n_features_in_} dimensions"
        )

        if model.n_features_in_ != RECORDING_EMBEDDING_DIM:
            raise RuntimeError(
                "SVM input dimension mismatch. "
                f"Expected {RECORDING_EMBEDDING_DIM}, "
                f"got {model.n_features_in_}."
            )

    # --------------------------------------------------------
    # Load label map
    #
    # Stored format:
    #
    # {
    #     "Ashy_Drongo": 0,
    #     "Ashy_Prinia": 1,
    #     ...
    # }
    #
    # Convert to:
    #
    # {
    #     0: "Ashy_Drongo",
    #     1: "Ashy_Prinia",
    #     ...
    # }
    # --------------------------------------------------------

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    classes = {
        int(class_id): class_name
        for class_name, class_id in label_map.items()
    }

    print(f"[INFO] Classes loaded: {len(classes)}")

    if len(classes) != 83:
        print(
            f"[WARN] Expected 83 classes, "
            f"but label map contains {len(classes)}."
        )

    # --------------------------------------------------------
    # Load BirdNET ONCE
    # --------------------------------------------------------

    print("[INFO] Loading BirdNET...")

    analyzer = Analyzer()

    # Required for embedding extraction used by this project.
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
    Convert SVM decision scores into relative confidence scores.

    NOTE:
    These are NOT calibrated probabilities.
    They are normalized scores used for ranking predictions.
    """

    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)

# ============================================================
# Normalize rows
# ============================================================

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row independently.
    """

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    return x / np.maximum(norms, 1e-12)

# ============================================================
# Frontend
# ============================================================

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

def estimate_approximate_count(events):
    """
    Estimate the number of distinct acoustic events for one species.

    This is an acoustic-event count, not a guaranteed count of individual
    birds. Each merged temporal event is treated as one occurrence.
    """
    if not events:
        return 0

    return len(events)

def detect_species_events(
    segment_data,
    window_svm,
    classes,
    window_size=5,
    step_size=2,
    confidence_threshold=0.05,
    top_n=1,
    merge_gap_seconds=2.0,
    occurrence_gap_seconds=15.0
):
    if not segment_data:
        return []

    embeddings = np.asarray(
        [item["embedding"] for item in segment_data],
        dtype=np.float32
    )

    # Normalize each BirdNET segment embedding
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
    embeddings = embeddings / np.maximum(norms, 1e-12)

    # Create temporal windows
    if len(segment_data) < window_size:
        window_ranges = [(0, len(segment_data))]
    else:
        window_ranges = [
            (start, start + window_size)
            for start in range(
                0,
                len(segment_data) - window_size + 1,
                step_size
            )
        ]

    detections_by_species = {}
    max_window_confidence = 0.0

    for start_idx, end_idx in window_ranges:
        window_embeddings = embeddings[start_idx:end_idx]

        # Same 3072-D pooling used during window-SVM training
        mean_embedding = np.mean(
            window_embeddings,
            axis=0
        )

        max_embedding = np.max(
            window_embeddings,
            axis=0
        )

        std_embedding = np.std(
            window_embeddings,
            axis=0
        )

        pooled_embedding = np.concatenate(
            [
                mean_embedding,
                max_embedding,
                std_embedding
            ]
        )

        pooled_norm = np.linalg.norm(pooled_embedding)

        if pooled_norm > 0:
            pooled_embedding = (
                pooled_embedding / pooled_norm
            )

        pooled_embedding = pooled_embedding.reshape(1, -1)

        # Window-level classifier
        probabilities = window_svm.predict_proba(
            pooled_embedding
        )[0]

        top_indices = np.argsort(probabilities)[::-1][:5]

        print(
            "[DEBUG] Top-5:",
            [
                (
                    classes.get(
                        int(window_svm.classes_[i]),
                        f"Unknown_{window_svm.classes_[i]}"
                    ).replace("_", " "),
                    f"{probabilities[i]:.2%}"
                )
                for i in top_indices
            ]
        )

        window_start = float(
            segment_data[start_idx]["start_time"]
        )

        window_end = float(
            segment_data[end_idx - 1]["end_time"]
        )

        for index in top_indices:
            confidence = float(
                probabilities[index]
            )

            max_window_confidence = max(
                max_window_confidence,
                confidence
            )

            if confidence < confidence_threshold:
                print(
                f"[DEBUG] Window {window_start:.2f}-{window_end:.2f}s "
                f"top confidence={confidence:.2%} -> rejected"
            )
                continue

            class_id = int(
                window_svm.classes_[index]
            )

            species = classes.get(
                class_id,
                f"Unknown_{class_id}"
            )

            species = species.replace("_", " ")

            if species not in detections_by_species:
                detections_by_species[species] = []

            detections_by_species[species].append(
                {
                    "start_time": window_start,
                    "end_time": window_end,
                    "confidence": confidence
                }
            )

    # Merge overlapping/nearby detections
    print(
        f"[DEBUG] Maximum window confidence: "
        f"{max_window_confidence:.2%}"
    )
    species_events = []

    for species, detections in detections_by_species.items():
        detections.sort(
            key=lambda x: x["start_time"]
        )

        merged_events = []

        for detection in detections:
            if not merged_events:
                merged_events.append(
                    {
                        "start_time": detection["start_time"],
                        "end_time": detection["end_time"],
                        "confidence": detection["confidence"],
                        "windows": 1
                    }
                )
                continue

            current = merged_events[-1]

            if (
                detection["start_time"]
                <= current["end_time"] + merge_gap_seconds
            ):
                current["end_time"] = max(
                    current["end_time"],
                    detection["end_time"]
                )

                current["confidence"] = max(
                    current["confidence"],
                    detection["confidence"]
                )

                current["windows"] += 1

            else:
                merged_events.append(
                    {
                        "start_time": detection["start_time"],
                        "end_time": detection["end_time"],
                        "confidence": detection["confidence"],
                        "windows": 1
                    }
                )

        # --------------------------------------------------------
        # Approximate temporal occurrence counting
        # --------------------------------------------------------
        #
        # event_count:
        #     Number of acoustically distinct events after
        #     overlapping/nearby classifier windows are merged.
        #
        # estimated_count:
        #     Number of temporally separated acoustic occurrences.
        #
        # IMPORTANT:
        #     This is NOT an exact individual-bird count.
        #     The same bird may produce multiple occurrences
        #     separated by a long silence.
        # --------------------------------------------------------

        occurrence_count = 0

        if merged_events:
            occurrence_count = 1

            for previous_event, current_event in zip(
                merged_events,
                merged_events[1:]
            ):
                gap = (
                    current_event["start_time"]
                    - previous_event["end_time"]
                )

                if gap > occurrence_gap_seconds:
                    occurrence_count += 1

        species_events.append(
            {
                "species": species,

                # Raw number of temporally distinct acoustic events
                "event_count": len(merged_events),

                # Approximate number of separated acoustic occurrences
                "estimated_count": occurrence_count,

                # Expose the uncertainty explicitly
                "count_note": (
                    "Approximate acoustic occurrence count. "
                    "The same bird may produce multiple occurrences."
                ),

                "confidence": max(
                    event["confidence"]
                    for event in merged_events
                ),

                "events": merged_events
            }
        )

    # Highest-confidence species first
    species_events.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return species_events

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
            content={
                "error": "BirdNET model is not loaded."
            },
        )

    if model is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Classifier model is not loaded."
            },
        )

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    filename = (file.filename or "").lower()

    if not (
        filename.endswith(".wav")
        or filename.endswith(".mp3")
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": (
                    "Could not process audio. "
                    "Only WAV and MP3 files are supported."
                )
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
    print(
        f"[PREDICT] Size     : "
        f"{len(raw) / 1024:.2f} KB"
    )

    if len(raw) == 0:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Uploaded audio file is empty."
            },
        )

    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Audio file is too large."
            },
        )

    # --------------------------------------------------------
    # Temporary file extension
    # --------------------------------------------------------

    suffix = (
        ".wav"
        if filename.endswith(".wav")
        else ".mp3"
    )

    tmp_path = None

    try:
        # ====================================================
        # Save uploaded audio temporarily
        # ====================================================

        tmp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        )

        tmp_path = Path(tmp_file.name)

        tmp_file.write(raw)
        tmp_file.close()

        print(
            f"[PREDICT] Temporary file: {tmp_path}"
        )

        # ====================================================
        # BirdNET embedding extraction
        # ====================================================

        print(
            "[PREDICT] Extracting BirdNET embeddings..."
        )

        recording = Recording(
            analyzer,
            str(tmp_path),
        )

        recording.extract_embeddings()
        print(
            f"[PREDICT] BirdNET segments: "
            f"{len(recording.embeddings)}"
        )

        if not recording.embeddings:
            print(
                "[WARN] No embeddings extracted."
            )

            return JSONResponse(
                status_code=422,
                content={
                    "error": (
                        "No BirdNET embeddings could "
                        "be extracted from the audio."
                    )
                },
            )

        # ====================================================
        # Collect all 1024-D segment embeddings
        # ====================================================

        segment_vectors = []
        segment_data = []

        for segment_number, seg in enumerate(
            recording.embeddings,
            start=1,
        ):
            # IMPORTANT:
            # Do NOT use:
            #
            # vec = seg.get("embeddings") or seg.get("embedding")
            #
            # because NumPy arrays cannot safely be
            # evaluated as booleans.

            vec = seg.get("embeddings")

            if vec is None:
                vec = seg.get("embedding")

            if vec is None:
                print(
                    f"[WARN] Segment {segment_number}: "
                    f"no embedding found."
                )
                continue

            arr = np.asarray(
                vec,
                dtype=np.float32,
            ).reshape(-1)

            if arr.shape[0] != BIRDNET_EMBEDDING_DIM:
                print(
                    f"[ERROR] Segment {segment_number}: "
                    f"expected {BIRDNET_EMBEDDING_DIM}-D "
                    f"embedding, got {arr.shape[0]}."
                )

                return JSONResponse(
                    status_code=500,
                    content={
                        "error": (
                            "BirdNET embedding dimension "
                            "mismatch. "
                            f"Expected {BIRDNET_EMBEDDING_DIM}, "
                            f"got {arr.shape[0]}."
                        )
                    },
                )

            start_time = seg.get("start_time")
            end_time = seg.get("end_time")

            if start_time is None or end_time is None:
                print(
                    f"Segment {segment_number}: "
                    "missing timestamp"
                )
                continue

            segment_vectors.append(arr)

            segment_data.append(
                {
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "embedding": arr
                }
            )

        # ====================================================
        # Make sure valid embeddings exist
        # ====================================================

        if not segment_vectors:
            print(
                "[WARN] No valid BirdNET embeddings "
                "were found."
            )

            return JSONResponse(
                status_code=422,
                content={
                    "error": (
                        "Could not obtain valid BirdNET "
                        "embeddings from the audio."
                    )
                },
            )

        # ====================================================
        # Build segment matrix
        #
        # Shape:
        #     (number_of_segments, 1024)
        # ====================================================

        segment_matrix = np.vstack(
            segment_vectors
        ).astype(np.float32)

        print(
            f"[PREDICT] Valid embeddings: "
            f"{segment_matrix.shape[0]}"
        )

        print(
            f"[PREDICT] Segment embedding shape: "
            f"{segment_matrix.shape}"
        )

        # ====================================================
        # EXACT SAME PREPROCESSING USED DURING TRAINING
        #
        # 1. L2-normalize every segment
        # 2. Mean pooling
        # 3. Max pooling
        # 4. Std pooling
        # 5. Concatenate
        # 6. L2-normalize final recording vector
        # ====================================================

        # ----------------------------------------------------
        # 1. L2-normalize every segment
        # ----------------------------------------------------

        segment_matrix = l2_normalize_rows(
            segment_matrix
        )

        # ----------------------------------------------------
        # 2. Mean pooling
        # ----------------------------------------------------

        mean_embedding = np.mean(
            segment_matrix,
            axis=0,
        )

        # ----------------------------------------------------
        # 3. Max pooling
        # ----------------------------------------------------

        max_embedding = np.max(
            segment_matrix,
            axis=0,
        )

        # ----------------------------------------------------
        # 4. Standard deviation pooling
        # ----------------------------------------------------

        std_embedding = np.std(
            segment_matrix,
            axis=0,
        )

        # ----------------------------------------------------
        # 5. Concatenate
        #
        # 1024 + 1024 + 1024 = 3072
        # ----------------------------------------------------

        recording_embedding = np.concatenate(
            [
                mean_embedding,
                max_embedding,
                std_embedding,
            ]
        ).astype(np.float32)

        if (
            recording_embedding.shape[0]
            != RECORDING_EMBEDDING_DIM
        ):
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Recording embedding dimension "
                        "mismatch. "
                        f"Expected {RECORDING_EMBEDDING_DIM}, "
                        f"got {recording_embedding.shape[0]}."
                    )
                },
            )

        # ----------------------------------------------------
        # 6. L2-normalize final recording embedding
        # ----------------------------------------------------

        recording_norm = np.linalg.norm(
            recording_embedding
        )

        recording_embedding = (
            recording_embedding
            / max(recording_norm, 1e-12)
        )

        print(
            "[PREDICT] Recording embedding shape: "
            f"{recording_embedding.shape}"
        )

        # ====================================================
        # SVM prediction
        # ====================================================

        model_input = recording_embedding.reshape(
            1,
            -1,
        )

        # Get probability estimates from the trained SVC.
        confidence_scores = model.predict_proba(
            model_input
        )[0]

        confidence_scores = np.asarray(
            confidence_scores,
            dtype=np.float64,
        )

        # For the current 83-class SVM this should be
        # (83,).
        if confidence_scores.ndim != 1:
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Unexpected SVM probability shape: "
                        f"{confidence_scores.shape}"
                    )
                },
            )

        # ====================================================
        # Top-K predictions
        # ====================================================

        top_indices = np.argsort(
            confidence_scores
        )[::-1][:TOP_K]

        predictions = []

        for idx in top_indices:
            idx = int(idx)

            # SVC class ordering is represented by
            # model.classes_. Since training labels are
            # 0..82, the class ID maps directly here.
            class_id = int(model.classes_[idx])

            species = classes.get(
                class_id,
                f"Unknown_{class_id}",
            )

            predictions.append(
                {
                    "species": species.replace(
                        "_",
                        " ",
                    ),
                    "confidence": float(
                        confidence_scores[idx]
                    ),
                }
            )

        # ====================================================
        # Print final prediction
        # ====================================================

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

        print("[PREDICT] Detecting temporal species events...")

        species_detected = detect_species_events(
            segment_data=segment_data,
            window_svm=window_model,
            classes=classes,
            window_size=5,
            step_size=2,
            confidence_threshold=0.20,
            top_n=1,
            merge_gap_seconds=2.0,
            occurrence_gap_seconds=15.0
        )

        print("[PREDICT] Detected species events:")

        for species in species_detected:
            print(
                f"  - {species['species']}: "
                f"{species['event_count']} event(s), "
                f"confidence={species['confidence']:.2%}"
            )

        print("-" * 60)
        print()

        # ====================================================
        # Return JSON to frontend
        # ====================================================

        total_approximate_count = sum(
            item["estimated_count"]
            for item in species_detected
        )

        return {
            "predictions": predictions,
            "species_detected": species_detected,
            "total_approximate_count": total_approximate_count,
            "count_note": (
                "Approximate acoustic-event count; it does not guarantee "
                "the number of individual birds."
            )
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
                "error": (
                    f"Failed to process audio: {str(e)}"
                )
            },
        )

    finally:
        # ====================================================
        # Delete temporary audio file
        # ====================================================

        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()

            except Exception as cleanup_error:
                print(
                    "[WARN] Could not delete temporary "
                    f"file: {cleanup_error}"
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
