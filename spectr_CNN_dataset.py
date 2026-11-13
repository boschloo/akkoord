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
    y, sr = librosa.load(file_path, sr=22050)
    
    # 128 Mel frequency bins
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Stabiele normalisatie tussen 0.0 en 1.0 (voorkom delen door 0)
    min_val, max_val = S_dB.min(), S_dB.max()
    denom = max_val - min_val
    if denom > 1e-6:
        S_norm = (S_dB - min_val) / denom
    else:
        S_norm = np.zeros_like(S_dB)

    # Pad of crop de tijd-as naar target_shape[1] (128 frames)
    _, n_frames = S_norm.shape
    if n_frames < target_shape[1]:
        pad_width = target_shape[1] - n_frames
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_norm = S_norm[:, :target_shape[1]]
        
    # Vorm: (Channels=1, Mel_Bins=128, Time_Frames=128)
    return torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)

def parse_filename(file_path):
    stem = Path(file_path).stem
    match = FILENAME_REGEX.match(stem)
    if not match:
        raise ValueError(f"Bestandsnaam '{file_path}' voldoet niet aan het patroon.")
    
    chord_type = match.group(3)
    root_note = match.group(5)[:2]
    return f"{root_note} {chord_type}"

class ChordSpectrogramDataset(Dataset):
    def __init__(self, dataset_dir, class_to_idx=None):
        self.dataset_dir = dataset_dir
        self.file_paths = []
        
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