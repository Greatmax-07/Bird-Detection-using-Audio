"""
04_train_window_svm.py
===================

Train a temporal window-level bird species classifier using existing
1024-D BirdNET segment embeddings.

Input:
    features/X.npy
    features/y.npy
    features/label_map.json
    features/manifest.csv

Window representation:
    L2-normalized segment embeddings
    -> mean pooling
    -> max pooling
    -> std pooling
    -> 3072-D concatenation
    -> L2 normalization

Important:
    Windows inherit the recording-level species label.
    This is weakly supervised: some windows may contain silence,
    background noise, or other sounds.

To reduce unnecessary RBF-SVM computation, only a limited number
of representative windows are used PER RECORDING for training.

All validation windows are retained.

Output:
    features/window_svm.pkl
    features/window_validation_predictions.csv
"""

import os
import json
import argparse
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)
from sklearn.svm import SVC


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_dir",
        type=str,
        default="features"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--gamma",
        type=str,
        default="scale"
    )

    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=5
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--max_train_windows_per_recording",
        type=int,
        default=2
    )

    return parser.parse_args()


def create_windows(
    X,
    y,
    manifest,
    window_size,
    step_size
):
    required_columns = [
        "filepath",
        "class",
        "start_time",
        "end_time"
    ]

    for column in required_columns:
        if column not in manifest.columns:
            raise ValueError(
                f"manifest.csv is missing required column: {column}"
            )

    if len(X) != len(y) or len(X) != len(manifest):
        raise ValueError(
            "Length mismatch:\n"
            f"X: {len(X)}\n"
            f"y: {len(y)}\n"
            f"manifest: {len(manifest)}"
        )

    window_features = []
    window_labels = []
    window_recordings = []
    window_start_times = []
    window_end_times = []

    grouped = manifest.groupby(
        "filepath",
        sort=False
    )

    for recording_number, (filepath, indices) in enumerate(
        grouped.groups.items(),
        start=1
    ):
        indices = np.asarray(list(indices))

        recording_X = X[indices]
        recording_y = y[indices]
        recording_manifest = manifest.iloc[indices]

        unique_labels = np.unique(recording_y)

        if len(unique_labels) != 1:
            raise ValueError(
                f"Recording has multiple labels: {filepath}"
            )

        recording_label = int(unique_labels[0])

        # Explicitly restore chronological order.
        order = np.argsort(
            recording_manifest["start_time"].to_numpy(
                dtype=float
            )
        )

        recording_X = recording_X[order]
        recording_manifest = recording_manifest.iloc[order]

        # Normalize each BirdNET segment embedding.
        recording_X = normalize(
            recording_X,
            norm="l2",
            axis=1
        )

        number_of_segments = len(recording_X)

        if number_of_segments < window_size:
            starts = [0]
        else:
            starts = list(
                range(
                    0,
                    number_of_segments - window_size + 1,
                    step_size
                )
            )

        for start in starts:
            end = min(
                start + window_size,
                number_of_segments
            )

            window = recording_X[start:end]

            mean_embedding = np.mean(
                window,
                axis=0
            )

            max_embedding = np.max(
                window,
                axis=0
            )

            std_embedding = np.std(
                window,
                axis=0
            )

            window_embedding = np.concatenate(
                [
                    mean_embedding,
                    max_embedding,
                    std_embedding
                ]
            )

            window_embedding = normalize(
                window_embedding.reshape(1, -1),
                norm="l2"
            )[0]

            window_start = float(
                recording_manifest.iloc[start]["start_time"]
            )

            window_end = float(
                recording_manifest.iloc[end - 1]["end_time"]
            )

            window_features.append(
                window_embedding
            )

            window_labels.append(
                recording_label
            )

            window_recordings.append(
                filepath
            )

            window_start_times.append(
                window_start
            )

            window_end_times.append(
                window_end
            )

        if recording_number % 500 == 0:
            print(
                f"Processed recordings: {recording_number}"
            )

    return (
        np.asarray(
            window_features,
            dtype=np.float32
        ),
        np.asarray(
            window_labels,
            dtype=np.int64
        ),
        np.asarray(
            window_recordings,
            dtype=object
        ),
        np.asarray(
            window_start_times,
            dtype=np.float32
        ),
        np.asarray(
            window_end_times,
            dtype=np.float32
        )
    )


def select_training_windows(
    window_recordings,
    train_recordings,
    max_windows_per_recording
):
    """
    Select evenly distributed windows from each training recording.

    This reduces highly overlapping training samples while preserving
    temporal coverage across the recording.
    """

    selected_indices = []

    for filepath in train_recordings:

        recording_indices = np.flatnonzero(
            window_recordings == filepath
        )

        if len(recording_indices) <= max_windows_per_recording:
            selected_indices.extend(
                recording_indices.tolist()
            )
            continue

        positions = np.linspace(
            0,
            len(recording_indices) - 1,
            num=max_windows_per_recording,
            dtype=int
        )

        selected = recording_indices[
            np.unique(positions)
        ]

        selected_indices.extend(
            selected.tolist()
        )

    return np.asarray(
        selected_indices,
        dtype=np.int64
    )


def main():

    args = parse_args()

    features_dir = args.features_dir

    X_path = os.path.join(
        features_dir,
        "X.npy"
    )

    y_path = os.path.join(
        features_dir,
        "y.npy"
    )

    label_map_path = os.path.join(
        features_dir,
        "label_map.json"
    )

    manifest_path = os.path.join(
        features_dir,
        "manifest.csv"
    )

    print("=" * 70)
    print("WINDOW-LEVEL BIRD SPECIES CLASSIFIER")
    print("=" * 70)

    print("\nLoading segment embeddings...")

    X = np.load(X_path)
    y = np.load(y_path)

    with open(
        label_map_path,
        "r",
        encoding="utf-8"
    ) as f:
        label_map = json.load(f)

    manifest = pd.read_csv(
        manifest_path
    )

    print(
        f"Segment embeddings: {len(X)}"
    )

    print(
        f"Embedding dimension: {X.shape[1]}"
    )

    print(
        f"Original classes: {len(label_map)}"
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )

    # ---------------------------------------------------------------
    # CREATE ALL TEMPORAL WINDOWS
    # ---------------------------------------------------------------

    print("\nCreating temporal windows...")

    print(
        f"Window size: {args.window_size} segments"
    )

    print(
        f"Step size:   {args.step_size} segments"
    )

    (
        X_windows,
        y_windows,
        window_recordings,
        window_start_times,
        window_end_times
    ) = create_windows(
        X,
        y,
        manifest,
        args.window_size,
        args.step_size
    )

    print(
        f"\nWindow embeddings: "
        f"{len(X_windows)}"
    )

    print(
        f"Window embedding dimension: "
        f"{X_windows.shape[1]}"
    )

    # ---------------------------------------------------------------
    # RECORDING-LEVEL SPLIT
    # ---------------------------------------------------------------

    print(
        "\nCreating recording-level "
        "train/validation split..."
    )

    unique_recordings = np.unique(
        window_recordings
    )

    recording_labels = []

    for filepath in unique_recordings:

        first_index = np.flatnonzero(
            window_recordings == filepath
        )[0]

        recording_labels.append(
            y_windows[first_index]
        )

    recording_labels = np.asarray(
        recording_labels
    )

    (
        train_recordings,
        val_recordings
    ) = train_test_split(
        unique_recordings,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=recording_labels
    )

    train_recordings = set(
        train_recordings
    )

    val_recordings = set(
        val_recordings
    )

    overlap = (
        train_recordings
        .intersection(val_recordings)
    )

    print(
        f"Training recordings:   "
        f"{len(train_recordings)}"
    )

    print(
        f"Validation recordings: "
        f"{len(val_recordings)}"
    )

    print(
        f"Recording overlap:     "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Recording leakage detected!"
        )

    # ---------------------------------------------------------------
    # SELECT REPRESENTATIVE TRAINING WINDOWS
    # ---------------------------------------------------------------

    print(
        "\nSelecting representative "
        "training windows..."
    )

    train_window_indices = select_training_windows(
        window_recordings=window_recordings,
        train_recordings=train_recordings,
        max_windows_per_recording=(
            args.max_train_windows_per_recording
        )
    )

    X_train = X_windows[
        train_window_indices
    ]

    y_train = y_windows[
        train_window_indices
    ]

    # Validation retains ALL windows.
    val_mask = np.array([
        filepath in val_recordings
        for filepath in window_recordings
    ])

    X_val = X_windows[
        val_mask
    ]

    y_val = y_windows[
        val_mask
    ]

    print(
        f"\nMaximum training windows per recording: "
        f"{args.max_train_windows_per_recording}"
    )

    print(
        f"Training windows actually used: "
        f"{len(X_train)}"
    )

    print(
        f"Validation windows: "
        f"{len(X_val)}"
    )

    # ---------------------------------------------------------------
    # TRAIN SVM
    # ---------------------------------------------------------------

    print(
        "\nTraining window-level SVM..."
    )

    print(
        f"C = {args.C}"
    )

    print(
        f"gamma = {args.gamma}"
    )

    model = SVC(
        C=args.C,
        gamma=args.gamma,
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=args.seed
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    print(
        "\nRunning window-level validation..."
    )

    y_pred = model.predict(
        X_val
    )

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("WINDOW-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    # ---------------------------------------------------------------
    # CLASS NAMES
    # ---------------------------------------------------------------

    if all(
        str(i) in label_map
        for i in range(len(label_map))
    ):

        class_names = [
            label_map[str(i)]
            for i in range(len(label_map))
        ]

    else:

        class_names = [
            name
            for name, _ in sorted(
                label_map.items(),
                key=lambda item: item[1]
            )
        ]

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_val,
            y_pred,
            labels=np.arange(
                len(class_names)
            ),
            target_names=class_names,
            zero_division=0
        )
    )

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    model_path = os.path.join(
        features_dir,
        "window_svm.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nWindow model saved to:"
    )

    print(
        model_path
    )

    # ---------------------------------------------------------------
    # SAVE VALIDATION RESULTS
    # ---------------------------------------------------------------

    results = pd.DataFrame({
        "filepath":
            window_recordings[val_mask],

        "start_time":
            window_start_times[val_mask],

        "end_time":
            window_end_times[val_mask],

        "true_label":
            y_val,

        "predicted_label":
            y_pred
    })

    results["true_class"] = results[
        "true_label"
    ].map(
        lambda x: class_names[int(x)]
    )

    results["predicted_class"] = results[
        "predicted_label"
    ].map(
        lambda x: class_names[int(x)]
    )

    results_path = os.path.join(
        features_dir,
        "window_validation_predictions.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print(
        "\nValidation predictions saved to:"
    )

    print(
        results_path
    )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Window size:             "
        f"{args.window_size}"
    )

    print(
        f"Step size:               "
        f"{args.step_size}"
    )

    print(
        f"Total windows:           "
        f"{len(X_windows)}"
    )

    print(
        f"Training windows used:   "
        f"{len(X_train)}"
    )

    print(
        f"Validation windows:      "
        f"{len(X_val)}"
    )

    print(
        f"Training recordings:     "
        f"{len(train_recordings)}"
    )

    print(
        f"Validation recordings:   "
        f"{len(val_recordings)}"
    )

    print(
        f"Classes:                 "
        f"{len(class_names)}"
    )

    print(
        f"Accuracy:                "
        f"{accuracy:.2%}"
    )

    print(
        f"Macro F1:                "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1:             "
        f"{weighted_f1:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()