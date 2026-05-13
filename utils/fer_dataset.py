import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FERDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row['emotion'])
        pixels = np.array(row['pixels'].split(), dtype=np.float32)
        image = pixels.reshape(48, 48, 1) / 255.0  # Normalize
        image = torch.tensor(image).permute(2,0,1)  # C,H,W

        if self.transform:
            image = self.transform(image)

        return image, label
