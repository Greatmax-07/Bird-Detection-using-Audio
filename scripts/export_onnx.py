"""
export_onnx.py
--------------
Export a trained BirdSoundClassifier checkpoint to ONNX for deployment.

Usage:
    python export_onnx.py \
        --checkpoint ./checkpoints/best_model.pt \
        --class_map  ./processed/class_map.json \
        --output     ./bird_classifier.onnx

The exported model:
  - Input:  float32 tensor of shape (1, 3, 128, 216)
  - Output: float32 logits of shape (1, num_classes)
  - After softmax → probability per species

For mobile (TFLite), convert further:
    pip install onnx-tf tensorflow
    onnx-tf convert -i bird_classifier.onnx -o bird_tf/
    # Then use tf.lite.TFLiteConverter
"""

import argparse
import json
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

from model import BirdSoundClassifier


def export(args):
    device = torch.device("cpu")   # export on CPU for compatibility

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt["num_classes"]

    model = BirdSoundClassifier(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded model: {num_classes} classes, from epoch {ckpt.get('epoch', '?')}")
    print(f"Val top-1 at export: {ckpt.get('val_top1', 'N/A'):.2f}%")

    # Dummy input (batch=1, 3 channels, 128 mel bins, 216 time frames)
    dummy = torch.randn(1, 3, 128, 216)

    output_path = args.output
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params   = True,
        opset_version   = 17,
        do_constant_folding = True,
        input_names     = ["mel_spectrogram"],
        output_names    = ["logits"],
        dynamic_axes    = {
            "mel_spectrogram": {0: "batch_size"},
            "logits":          {0: "batch_size"},
        },
    )

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model validated: {output_path}")

    # Quick runtime check
    sess = ort.InferenceSession(output_path)
    out  = sess.run(None, {"mel_spectrogram": dummy.numpy()})
    logits = out[0]
    probs  = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    print(f"✓ ONNX runtime test passed — output shape: {logits.shape}")
    print(f"  Top class probability: {probs.max():.4f}")

    # Save class map alongside for inference
    with open(args.class_map) as f:
        class_map = json.load(f)

    idx_to_class = {v: k for k, v in class_map.items()}
    top_idx = int(np.argmax(probs[0]))
    print(f"  (Dummy input predicted: '{idx_to_class.get(top_idx, '?')}' — expected random)")

    class_map_out = Path(output_path).with_suffix(".classes.json")
    with open(class_map_out, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"  Class map saved: {class_map_out}")

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"\nExport complete. Model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--class_map",  type=str, default="./processed/class_map.json")
    parser.add_argument("--output",     type=str, default="./bird_classifier.onnx")
    args = parser.parse_args()
    export(args)
