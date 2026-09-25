"""
04_export_head_onnx.py
========================
Export the trained classifier head (a small MLP on top of frozen BirdNET
embeddings) to ONNX, for use in the FastAPI backend.

This replaces the OLD export_onnx.py / model.py pipeline, which exported a
full EfficientNet-B2 that took raw mel spectrograms as input. The new
pipeline splits inference into two stages instead of one ONNX graph:

    1. BirdNET (via birdnetlib) turns raw audio into 1024-dim embeddings
       — this step runs in Python at request time, NOT inside ONNX,
       since BirdNET's TFLite internals aren't part of this export.
    2. This exported ONNX model takes those embeddings and outputs
       per-species logits.

Usage:
    python 04_export_head_onnx.py \
        --checkpoint ./features/classifier_head_best.pt \
        --label_map  ./features/label_map.json \
        --output     ./bird_classifier.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Must match the architecture in 03_train_classifier_head.py exactly."""

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
    ap.add_argument("--checkpoint", default="./features/classifier_head_best.pt")
    ap.add_argument("--label_map", default="./features/label_map.json")
    ap.add_argument("--output", default="./bird_classifier.onnx")
    ap.add_argument("--embedding_dim", type=int, default=1024)
    args = ap.parse_args()

    with open(args.label_map) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    idx_to_class = {v: k for k, v in label_map.items()}

    model = ClassifierHead(in_dim=args.embedding_dim, n_classes=num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, args.embedding_dim)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["embedding"],
        output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model validated: {args.output}")

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"embedding": dummy.numpy()})
    logits = out[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    print(f"Runtime test passed — output shape: {logits.shape}, {num_classes} classes")

    classes_out = Path(args.output).with_suffix(".classes.json")
    with open(classes_out, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Class map saved: {classes_out}")

    size_kb = Path(args.output).stat().st_size / 1024
    print(f"\nExport complete. Model size: {size_kb:.1f} KB (vs. ~31 MB for the old EfficientNet-B2)")


if __name__ == "__main__":
    main()
