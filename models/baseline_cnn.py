"""
Baseline Convolutional Neural Network (CNN) Model for Kannada HCR.
Faithfully recreates the architecture specified in the 2019 IEEE WIECON-ECE paper:
'Offline Kannada Handwritten Character Recognition Using Convolutional Neural Networks'
by Ramesh G., Ganesh N. Sharma, J. Manoj Balaji, Champa H.N. (Fig. 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineKannadaCNN(nn.Module):
    """
    Baseline CNN Architecture from 2019 IEEE Paper:
    - 4 Convolutional layers (3x3 kernels)
    - 2 Max-pooling layers (2x2)
    - Flatten layer
    - 2 Dense layers with Dropout
    - Output softmax layer
    """
    def __init__(self, num_classes: int = 47, dropout_rate: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)  # 28 -> 26
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0) # 26 -> 24
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 24 -> 12
        
        # Block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0) # 12 -> 10
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0) # 10 -> 8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                # 8 -> 4
        
        # Classifier
        self.flatten_dim = 64 * 4 * 4  # 1024
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
