import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import TextEmotionDataset
from models import TextEmotionModel
from tqdm import tqdm
import os
import numpy as np

def train(tsv_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"🚀 Training text emotion model on {device}")

    # Load full dataset
    full_dataset = TextEmotionDataset(tsv_path)
    num_labels = full_dataset.num_labels

    # Split into train (80%) and validation (20%)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model setup
    model = TextEmotionModel(num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} average train loss: {avg_train_loss:.4f}")

        # --- Validation step ---
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(logits).cpu().numpy()
                labels = labels.cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Convert sigmoid outputs to binary (0 or 1)
        all_preds_binary = (all_preds > 0.5).astype(int)
        accuracy = (all_preds_binary == all_labels).mean()

        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Validation loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/text_emotion_model.pth")
    print("💾 Model saved to models/text_emotion_model.pth")

if __name__ == "__main__":
    tsv_path = "../data/text/goemotions/train.tsv"
    train(tsv_path)
