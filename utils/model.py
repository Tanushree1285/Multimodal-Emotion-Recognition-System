# utils/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 48x48 -> 48x48
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 48x48 -> 48x48
        self.pool = nn.MaxPool2d(2, 2)                # 48x48 -> 24x24
        self.pool2 = nn.MaxPool2d(2, 2)               # 24x24 -> 12x12
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*12*12, 128)           # matches 12x12 feature map
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))   # first pooling
        x = self.pool2(x)                       # second pooling
        x = self.dropout(x)
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
