import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import os

class TextEmotionDataset(Dataset):
    def __init__(self, tsv_path):
        # Paths
        label_map_path = os.path.join(os.path.dirname(__file__), "../data/text/goemotions/label_map.json")
        label_map_path = os.path.normpath(label_map_path)

        # Load label map
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        
        self.num_labels = len(self.label_map)  # ✅ Define before using

        # Load TSV
        self.data = pd.read_csv(tsv_path, sep="\t", header=None, names=["text", "labels", "id"])

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Parse labels
        self.data["label_ids"] = self.data["labels"].apply(self._parse_labels)

    def _parse_labels(self, label_str):
        try:
            indices = [int(i) for i in str(label_str).split(",") if i.strip().isdigit()]
            multi_hot = torch.zeros(self.num_labels)
            for i in indices:
                if i < self.num_labels:
                    multi_hot[i] = 1
            return multi_hot
        except Exception as e:
            print(f"⚠️ Label parsing error for '{label_str}': {e}")
            return torch.zeros(self.num_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        labels = row["label_ids"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        # Flatten tensors
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = labels.float()

        return item
