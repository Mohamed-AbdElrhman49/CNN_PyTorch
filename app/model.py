import torch
import torch.nn as nn


class PyTorchModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.linear1 = nn.Linear(256, 128)
        self.bnfc1 = nn.BatchNorm1d(128)

        self.linear2 = nn.Linear(128, 64)
        self.bnfc2 = nn.BatchNorm1d(64)

        self.linear3 = nn.Linear(64, 1)

        # Activations
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.pool(
            self.relu(self.bn1(self.conv1(x)))
        )

        x = self.pool(
            self.relu(self.bn2(self.conv2(x)))
        )

        x = self.pool(
            self.relu(self.bn3(self.conv3(x)))
        )

        x = self.pool(
            self.relu(self.bn4(self.conv4(x)))
        )

        x = self.global_pool(x)
        x = self.flatten(x)

        x = self.relu(self.bnfc1(self.linear1(x)))
        x = self.relu(self.bnfc2(self.linear2(x)))

        x = self.dropout(x)

        x = self.linear3(x)

        return x