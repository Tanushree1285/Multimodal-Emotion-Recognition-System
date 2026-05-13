import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils.fer_dataset import FERDataset
from utils.model import SimpleCNN

# Hyperparameters
batch_size = 64
epochs = 25
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset & Dataloader
train_dataset = FERDataset("data/train.csv")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), "models/face_emotion_model.pth")
print("Model saved at models/face_emotion_model.pth")
