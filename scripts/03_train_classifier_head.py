import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.svm import SVC


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_dir", type=str, default="features")

    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", type=str, default="scale")

    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def aggregate_recordings(X, y, manifest):
    """
    Convert segment-level BirdNET embeddings into
    one embedding per original recording.

    For each recording:
        1. collect all segment embeddings
        2. L2-normalize each embedding
        3. average the normalized embeddings
        4. L2-normalize the final recording embedding
    """

    manifest = manifest.copy()

    required_columns = [
        "filepath",
        "class"
    ]

    for column in required_columns:
        if column not in manifest.columns:
            raise ValueError(
                f"manifest.csv is missing required column: {column}"
            )

    if len(X) != len(y) or len(X) != len(manifest):
        raise ValueError(
            f"Length mismatch:\n"
            f"X: {len(X)}\n"
            f"y: {len(y)}\n"
            f"manifest: {len(manifest)}"
        )

    recording_embeddings = []
    recording_labels = []
    recording_paths = []

    grouped = manifest.groupby("filepath", sort=False)

    for filepath, indices in grouped.groups.items():

        indices = np.asarray(list(indices))

        recording_X = X[indices]

        # Normalize each segment embedding
        recording_X = normalize(
            recording_X,
            norm="l2",
            axis=1
        )

        # Mean + max + standard deviation pooling
        mean_embedding = np.mean(recording_X, axis=0)
        max_embedding = np.max(recording_X, axis=0)
        std_embedding = np.std(recording_X, axis=0)

        recording_embedding = np.concatenate(
            [mean_embedding, max_embedding, std_embedding]
        )

        # Normalize final recording embedding
        recording_embedding = normalize(
            recording_embedding.reshape(1, -1),
            norm="l2"
        )[0]

        recording_embeddings.append(recording_embedding)

        # Every segment from one recording must have the same label
        recording_labels.append(y[indices[0]])
        recording_paths.append(filepath)

    X_recordings = np.asarray(
        recording_embeddings,
        dtype=np.float32
    )

    y_recordings = np.asarray(
        recording_labels,
        dtype=np.int64
    )

    recording_paths = np.asarray(
        recording_paths,
        dtype=object
    )

    return X_recordings, y_recordings, recording_paths


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
    print("BIRD SPECIES CLASSIFICATION")
    print("Recording-Level BirdNET Classifier")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print("\nLoading data...")

    X = np.load(X_path)
    y = np.load(y_path)

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    manifest = pd.read_csv(manifest_path)

    print(f"Segment embeddings: {len(X)}")
    print(f"Embedding dimension: {X.shape[1]}")
    print(f"Original classes: {len(label_map)}")
    print(f"Manifest rows: {len(manifest)}")

    # ---------------------------------------------------------
    # AGGREGATE SEGMENTS INTO RECORDINGS
    # ---------------------------------------------------------

    print("\nAggregating embeddings by recording...")

    X_rec, y_rec, paths = aggregate_recordings(
        X,
        y,
        manifest
    )

    print(f"Unique recordings: {len(X_rec)}")
    print(f"Recording embedding dimension: {X_rec.shape[1]}")

    # ---------------------------------------------------------
    # VERIFY LABEL CONSISTENCY
    # ---------------------------------------------------------

    print("\nChecking recording labels...")

    inconsistent_recordings = 0

    for filepath, indices in manifest.groupby("filepath").groups.items():

        indices = np.asarray(list(indices))

        labels = np.unique(y[indices])

        if len(labels) != 1:
            inconsistent_recordings += 1
            print(
                f"[WARNING] Multiple labels found for: {filepath}"
            )

    if inconsistent_recordings == 0:
        print("All recordings have consistent labels.")
    else:
        raise ValueError(
            f"Found {inconsistent_recordings} recordings "
            f"with inconsistent labels."
        )

    # ---------------------------------------------------------
    # RECORDING-LEVEL TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------

    print("\nCreating recording-level split...")

    (
        X_train,
        X_val,
        y_train,
        y_val,
        paths_train,
        paths_val
    ) = train_test_split(
        X_rec,
        y_rec,
        paths,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=y_rec
    )

    # ---------------------------------------------------------
    # VERIFY NO RECORDING LEAKAGE
    # ---------------------------------------------------------

    train_paths = set(paths_train)
    val_paths = set(paths_val)

    overlap = train_paths.intersection(val_paths)

    print(f"\nTraining recordings:   {len(train_paths)}")
    print(f"Validation recordings: {len(val_paths)}")
    print(f"Recording overlap:     {len(overlap)}")

    if len(overlap) != 0:
        raise RuntimeError(
            "Recording leakage detected!"
        )

    # ---------------------------------------------------------
    # TRAIN SVM
    # ---------------------------------------------------------

    print("\nTraining SVM...")
    print(f"C = {args.C}")
    print(f"gamma = {args.gamma}")

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

    print("Training complete.")

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\nRunning validation...")

    y_pred = model.predict(X_val)

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
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # ---------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------

    if isinstance(label_map, dict):
    # Handle either {"0": "species"} or {"species": 0}
        if all(str(i) in label_map for i in range(len(label_map))):
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
    else:
        raise ValueError("Unexpected label_map format.")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    import joblib

    model_path = os.path.join(
        features_dir,
        "recording_svm.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"\nModel saved to:")
    print(model_path)

    # ---------------------------------------------------------
    # SAVE VALIDATION PREDICTIONS
    # ---------------------------------------------------------

    results = pd.DataFrame({
        "filepath": paths_val,
        "true_label": y_val,
        "predicted_label": y_pred
    })

    class_names = [
    name
    for name, class_id in sorted(
        label_map.items(),
        key=lambda item: item[1]
    )
    ]

    results["true_class"] = results["true_label"].map(
        lambda x: class_names[int(x)]
    )

    results["predicted_class"] = results["predicted_label"].map(
        lambda x: class_names[int(x)]
    )

    results_path = os.path.join(
        features_dir,
        "recording_validation_predictions.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print(f"Validation predictions saved to:")
    print(results_path)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Recordings:       {len(X_rec)}")
    print(f"Classes:          {len(class_names)}")
    print(f"Train recordings: {len(X_train)}")
    print(f"Val recordings:   {len(X_val)}")
    print(f"Accuracy:         {accuracy:.2%}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Weighted F1:      {weighted_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()