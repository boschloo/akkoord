import torch
import torch.nn as nn

class ChordCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChordCNN, self).__init__()
        
        # Invoer: (1, 128, 128) -> (1 channel, 128 Mel Bins, 128 Time Frames)
        self.features = nn.Sequential(
            # Block 1 -> Output: (32, 128, 64)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), # Pool ALLEEN tijd
            
            # Block 2 -> Output: (64, 128, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), # Pool ALLEEN tijd
            
            # Block 3 -> Output: (128, 128, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Pool ALLEEN tijd
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),  # Berekent automatisch 262.144
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x