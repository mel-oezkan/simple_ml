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
            nn.Linear(128 * 7 * 7, 112),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(112, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))