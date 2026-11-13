import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from spectr_CNN_dataset import ChordSpectrogramDataset
from spectr_CNN_model import ChordCNN

DATASET_DIR = "./audio_samples"
BATCH_SIZE = 16          # Verlaagd naar 16 vanwege de grotere tensor in de Linear layer
EPOCHS = 15              # Verhoogd voor gestage convergentie
LEARNING_RATE = 0.0003   # CRUCIAAL: Verlaagd van 0.001 voor Mel-spectrogrammen

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Gebruikte hardware: {device}")

    full_dataset = ChordSpectrogramDataset(DATASET_DIR)
    num_classes = len(full_dataset.class_to_idx)
    print(f"Dataset geladen: {len(full_dataset)} samples over {num_classes} klassen.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ChordCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Adam met L2-regularisatie (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

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
        
        # Validatie
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

    # Opslaan
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': full_dataset.class_to_idx,
        'idx_to_class': full_dataset.idx_to_class,
        'num_classes': num_classes
    }, "cnn_root_chord_model.pth")
    print("CNN Model succesvol opgeslagen in 'cnn_root_chord_model.pth'")

if __name__ == "__main__":
    train()