"""
model.py
--------
EfficientNet-B2 fine-tuned for mel spectrogram bird classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B2_Weights


class BirdSoundClassifier(nn.Module):
    """
    EfficientNet-B2 pretrained on ImageNet, head replaced for num_classes.

    Input:  (B, 3, 128, 216) — batch of 3-channel mel spectrograms
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Load pretrained backbone
        self.backbone = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the classifier head (useful for first few epochs)."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes: int, device: torch.device) -> BirdSoundClassifier:
    model = BirdSoundClassifier(num_classes=num_classes)
    model = model.to(device)
    return model
