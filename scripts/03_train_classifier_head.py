"""
03_train_classifier_head.py
=============================
Trains a small MLP classifier head on top of the frozen BirdNET embeddings
produced by script 02. Since the backbone (BirdNET) is already pretrained,
this head is tiny and fast to train, and far less prone to overfitting your
minority classes than training a CNN from scratch would be.

Uses focal loss instead of plain cross-entropy, since focal loss down-
weights easy majority-class examples and concentrates gradient on
hard/minority examples -- generally a bigger win than class weighting
alone on long-tailed datasets like this one.

Install:
    pip install torch scikit-learn numpy

Usage:
    python 03_train_classifier_head.py --features_dir ./features
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    """Standard multi-class focal loss (Lin et al., 2017)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_probs = (
            log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        )
        target_probs = (
            probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        )

        focal_term = (1 - target_probs) ** self.gamma
        loss = -focal_term * target_log_probs

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--features_dir",
        default="./features",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
    )

    args = ap.parse_args()

    feat_dir = Path(args.features_dir)

    # ------------------------------------------------------------------
    # Load BirdNET embeddings and labels
    # ------------------------------------------------------------------
    X = np.load(feat_dir / "X.npy")
    y = np.load(feat_dir / "y.npy")

    with open(feat_dir / "label_map.json") as f:
        label_map = json.load(f)

    # label_map is expected to be:
    #   {"Class_Name": integer_id, ...}
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [
        inv_label_map[i]
        for i in range(len(label_map))
    ]

    print(f"Loaded {len(X)} embeddings")
    print(f"Embedding dimension: {X.shape[1]}")
    print(f"Number of classes: {len(class_names)}")

    # ------------------------------------------------------------------
    # Stratified train/validation split
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_frac,
        stratify=y,
        random_state=42,
    )

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Inverse-frequency class weights
    # ------------------------------------------------------------------
    counts = np.bincount(
        y_train,
        minlength=len(class_names),
    )

    class_weights = torch.tensor(
        counts.sum() / (counts + 1e-6),
        dtype=torch.float32,
    ).to(device)

    class_weights = (
        class_weights / class_weights.mean()
    )

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ClassifierHead(
        in_dim=X.shape[1],
        n_classes=len(class_names),
    ).to(device)

    criterion = FocalLoss(
        gamma=2.0,
        weight=class_weights,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_ds)

        # --------------------------------------------------------------
        # Save best checkpoint
        # --------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                feat_dir / "classifier_head_best.pt",
            )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d}  "
                f"train_loss {train_loss:.4f}  "
                f"val_loss {val_loss:.4f}"
            )

    # ------------------------------------------------------------------
    # Final validation using best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict(
        torch.load(
            feat_dir / "classifier_head_best.pt",
            map_location=device,
        )
    )

    model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(
            X_val,
            dtype=torch.float32,
        ).to(device)

        preds = (
            model(val_tensor)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------
    print("\n=== Per-class validation report ===")

    print(
        classification_report(
            y_val,
            preds,
            target_names=class_names,
            zero_division=0,
        )
    )

    print(
        f"Best model saved to "
        f"{feat_dir / 'classifier_head_best.pt'}"
    )

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(
        y_val,
        preds,
        labels=np.arange(len(class_names)),
    )

    print("\n=== Top confusions for the weakest classes ===")

    for name in [
        "Humes_Bar-tailed_Scimitar_Babbler",
    ]:
        if name not in label_map:
            print(
                f"\n{name}: not present in label_map, skipping."
            )
            continue

        idx = label_map[name]

        row = cm[idx].copy()

        # Remove correct predictions.
        row[idx] = 0

        top = row.argsort()[::-1][:3]

        print(
            f"\n{name} (true label) most often predicted as:"
        )

        found_confusion = False

        for target_idx in top:
            if row[target_idx] > 0:
                found_confusion = True

                print(
                    f"  {class_names[target_idx]:35s} "
                    f"{row[target_idx]} times"
                )

        if not found_confusion:
            print("  No incorrect predictions.")


if __name__ == "__main__":
    main()
