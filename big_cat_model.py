"""
Big Cat Classifier CNN Model 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigCatCNN(nn.Module):
    """
    Improved CNN with double convolutions per block for better accuracy.

    Architecture:
        - 4 blocks with DOUBLE convolutions each
        - BatchNorm after each conv
        - MaxPooling after each block
        - Global Average Pooling
        - Dropout for regularization
        - Two FC layers

    Input: [batch_size, 3, 224, 224] RGB images
    Output: [batch_size, num_classes] class logits
    """

    def __init__(self, num_classes=10):
        super(BigCatCNN, self).__init__()

        # Block 1: 3 -> 64 -> 64 channels
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)

        # Block 2: 64 -> 128 -> 128 channels
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)

        # Block 3: 128 -> 256 -> 256 channels
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)

        # Block 4: 256 -> 512 -> 512 channels
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(512)

        # Pooling and regularization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """Forward pass with double convolutions per block."""

        # Block 1: Two convs -> Pool (224 -> 112)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        # Block 2: Two convs -> Pool (112 -> 56)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        # Block 3: Two convs -> Pool (56 -> 28)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        # Block 4: Two convs -> Pool (28 -> 14)
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)

        # Global Average Pooling: 512x14x14 -> 512
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier with BatchNorm
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def count_parameters(model):
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test the model
    model = BigCatCNN(num_classes=10)
    model.eval()  # Important for BatchNorm with batch_size=1

    total, trainable = count_parameters(model)

    print(f"Model: BigCatCNN (Double Conv)")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
