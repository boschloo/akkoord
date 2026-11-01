import os
import re
from pathlib import Path
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

FILENAME_REGEX = re.compile(
    r"^([a-zA-Z0-9]+)_(\d+)_([a-zA-Z]+)_(\d+)_([A-G][\#b\_][0-9])_([a-zA-Z0-9]+)$"
)

def audio_to_spectrogram_tensor(file_path, target_shape=(128, 128)):
    """
    Loads audio, extracts log-Mel spectrogram, and pads/crops time axis to 128x128.
    Returns torch.Tensor of shape (1, 128, 128) representing 1 grayscale channel.
    """
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract 128 Mel frequency bins
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Normalize between 0.0 and 1.0
    min_val, max_val = S_dB.min(), S_dB.max()
    if max_val - min_val > 0:
        S_norm = (S_dB - min_val) / (max_val - min_val)
    else:
        S_norm = np.zeros_like(S_dB)

    # Pad or crop time frames (columns) to target_shape[1] (128)
    n_bins, n_frames = S_norm.shape
    if n_frames < target_shape[1]:
        pad_width = target_shape[1] - n_frames
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_norm = S_norm[:, :target_shape[1]]
        
    # Expand to 3D tensor: (Channels=1, Height=128, Width=128)
    tensor = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
    return tensor

def parse_filename(file_path):
    stem = Path(file_path).stem
    match = FILENAME_REGEX.match(stem)
    if not match:
        raise ValueError(f"Filename '{file_path}' does not match expected pattern.")
    
    chord_type = match.group(3)
    root_note = match.group(5)[:2]
    return f"{root_note} {chord_type}"  # Combined target (e.g. 'G_ maj')

class ChordSpectrogramDataset(Dataset):
    def __init__(self, dataset_dir, class_to_idx=None):
        self.dataset_dir = dataset_dir
        self.file_paths = []
        self.labels = []
        
        # Scan folder for .wav files
        raw_labels = []
        valid_paths = []
        for entry in os.scandir(dataset_dir):
            if entry.is_file() and entry.name.endswith('.wav'):
                try:
                    label = parse_filename(entry.path)
                    valid_paths.append(entry.path)
                    raw_labels.append(label)
                except Exception:
                    continue
                    
        self.file_paths = valid_paths
        
        # Generate or use provided class-to-index mapping
        if class_to_idx is None:
            unique_classes = sorted(list(set(raw_labels)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
            
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        self.labels = [self.class_to_idx[lbl] for lbl in raw_labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_tensor = audio_to_spectrogram_tensor(file_path)
        label_idx = self.labels[idx]
        return image_tensor, label_idx