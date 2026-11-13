# Akkoordherkenning
# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from chroma_CNN_dataset import ChordSpectrogramDataset
from chroma_CNN_model import ChordCNN
import warnings

DATASET_DIR = "./audio_samples_live"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load dataset
    full_dataset = ChordSpectrogramDataset(DATASET_DIR)
    num_classes = len(full_dataset.class_to_idx)
    print(f"Found {len(full_dataset)} samples across {num_classes} classes.")

    # Train / Test split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Loss, Optimizer
    model = ChordCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, correct = 0.0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
        train_acc = correct / train_size
        
        # Validation Step
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                
        val_acc = val_correct / val_size
        print(f"Epoch {epoch} Summary: Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Save model and mapping
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': full_dataset.class_to_idx,
        'idx_to_class': full_dataset.idx_to_class,
        'num_classes': num_classes
    }, "cnn_root_chord_model.pth")
    print("CNN Model successfully saved to 'cnn_root_chord_model.pth'")

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore", 
        category=UserWarning, 
        module="librosa.core.constantq"
    )
    train()
