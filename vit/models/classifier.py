from pathlib import Path

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class CNNClassifier(nn.Module):
    feature_size = 112

    def __init__(self, num_classes=10, dropout=0.1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, self.feature_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.out_head = nn.Linear(self.feature_size, num_classes)

    def forward_features(self, x):
        return self.classifier(self.features(x))

    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        logits = self.out_head(feats)

        if return_features:
            return logits, feats

        return logits


class CNNFeatureExtractor(nn.Module):
    """Expose only the frozen classifier's penultimate features."""

    def __init__(self, classifier: CNNClassifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        return self.classifier.forward_features(x)


def load_cnn_classifier(
    checkpoint_path: str | Path,
    num_classes: int = 10,
    map_location: str | torch.device = "cpu",
) -> CNNClassifier:
    """Initialize a CNN classifier from a training checkpoint or state dict."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    dropout = checkpoint.get("hyperparameters", {}).get("dropout", 0.1)

    # Checkpoints created before ``out_head`` was split from ``classifier``
    # stored the final linear layer under classifier.4.*.
    if "classifier.4.weight" in state_dict:
        state_dict = dict(state_dict)
        state_dict["out_head.weight"] = state_dict.pop("classifier.4.weight")
        state_dict["out_head.bias"] = state_dict.pop("classifier.4.bias")

    model = CNNClassifier(num_classes=num_classes, dropout=dropout)
    model.load_state_dict(state_dict)
    return model
