# Akkoordherkenning 
# dataset.py

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

def audio_to_chroma_tensor(file_path, target_time_frames=128):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract 12 chroma pitch classes across time
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Normalize globally (0 to 1 per chroma frame)
    chroma = librosa.util.normalize(chroma, axis=0)
    
    # Pad or crop time axis (columns) to fixed width
    _, n_frames = chroma.shape
    if n_frames < target_time_frames:
        pad_width = target_time_frames - n_frames
        chroma = np.pad(chroma, ((0, 0), (0, pad_width)), mode='constant')
    else:
        chroma = chroma[:, :target_time_frames]
        
    # Shape: (1 channel, 12 pitch bins, 128 time frames)
    return torch.tensor(chroma, dtype=torch.float32).unsqueeze(0)

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
        image_tensor = audio_to_chroma_tensor(file_path)
        label_idx = self.labels[idx]
        return image_tensor, label_idx
