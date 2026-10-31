import os
import re
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
import joblib

FILENAME_REGEX = re.compile(
    r"^([a-zA-Z0-9]+)_(\d+)_([a-zA-Z]+)_(\d+)_([A-G][\#b\_][0-9])_([a-zA-Z0-9]+)$"
)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    return np.hstack([chroma_mean, chroma_std])

def parse_filename(file_path):
    """
    Parses filename format: sss_440_acc_123_PITCH_src.wav
    Example: asx_435_dim_123_A_4_drc.wav
    """
    stem = Path(file_path).stem
    match = FILENAME_REGEX.match(stem)
    
    if not match:
        raise ValueError(f"Filename '{file_path}' does not match expected pattern.")
    
    chord_type = match.group(3)        # e.g., 'maj', 'dim', 'min', 'aug'
    pitch_str = match.group(5)         # e.g., 'G_3', 'C#4', 'Bb2'
    
    # Extract note name (first 2 characters) and remove octave digit
    root_note = pitch_str[:2]          # e.g., 'G_', 'C#', 'Bb'
    
    # Create combined classification target
    root_chord = f"{root_note}{chord_type}"  # e.g., 'G_ maj', 'C# dim'
    
    return {
        'instrument': match.group(1),
        'tuning': int(match.group(2)),
        'chord_type': chord_type,
        'inversion': match.group(4),
        'root_octave': pitch_str,
        'root_note': root_note,
        'root_chord': root_chord,        # Target label
        'source': match.group(6)
    }

def load_dataset_from_filenames(dataset_dir, target_attribute='root_chord'):
    X, y = [], []
    
    # Get total file count for the progress bar
    all_entries = os.listdir(dataset_dir)
    wav_files = [f for f in all_entries if f.endswith('.wav')]
    
    print(f"Extracting features from {len(wav_files)} files...")
    
    # Wrap iteration with tqdm progress bar
    for file_name in tqdm(wav_files, desc="Processing Audio Files", unit="file"):
        file_path = os.path.join(dataset_dir, file_name)
        try:
            metadata = parse_filename(file_path)
            features = extract_features(file_path)
            
            X.append(features)
            y.append(metadata[target_attribute])
        except Exception as e:
            tqdm.write(f"Skipping {file_name}: {e}")  # Prevents progress bar distortion
            
    return np.array(X), np.array(y)

# Silence the specific internal Constant-Q filter size warning from librosa
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="librosa.core.constantq"
)

# 1. Train your model on the root_chord target
X, y = load_dataset_from_filenames("./audio_samples", target_attribute='root_chord')

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Fit classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Save the trained model to disk
joblib.dump(clf, "root_chord_model.pkl")
print("Model saved successfully as 'root_chord_model.pkl'")

# 5. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title("Chord Classification Confusion Matrix")
plt.show()