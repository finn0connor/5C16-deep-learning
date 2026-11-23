import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout_conv = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout_conv(self.pool(F.relu(self.conv1(x))))
        x = self.dropout_conv(self.pool(F.relu(self.conv2(x))))
        x = self.dropout_conv(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x