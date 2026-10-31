# Akkoordherkenning
# Gemini
# Logistic Regression, chromagram
# RandomForest Classifier



import os
import re
import warnings
from pathlib import Path
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Suppress the internal librosa Constant-Q filter window size warning
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="librosa.core.constantq"
)

FILENAME_REGEX = re.compile(
    r"^([a-zA-Z0-9]+)_(\d+)_([a-zA-Z]+)_(\d+)_([A-G][\#b\_][0-9])_([a-zA-Z0-9]+)$"
)

def extract_chroma_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract Chromagram (12 pitch classes)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Aggregate over time -> 12 means + 12 stds = 24 features
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    return np.hstack([chroma_mean, chroma_std])

def parse_filename(file_path):
    stem = Path(file_path).stem
    match = FILENAME_REGEX.match(stem)
    if not match:
        raise ValueError(f"Invalid format: {file_path}")
    
    chord_type = match.group(3)      # e.g., 'maj', 'dim'
    root_note = match.group(5)[:2]   # e.g., 'G_', 'C#'
    return f"{root_note} {chord_type}"

def load_dataset(dataset_dir):
    X, y = [], []
    
    # Get total .wav files for the progress indicator
    all_entries = os.listdir(dataset_dir)
    wav_files = [f for f in all_entries if f.endswith('.wav')]
    
    print(f"Extracting features from {len(wav_files)} files...")
    
    for file_name in tqdm(wav_files, desc="Processing Audio Files", unit="file"):
        file_path = os.path.join(dataset_dir, file_name)
        try:
            label = parse_filename(file_path)
            features = extract_chroma_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            tqdm.write(f"Skipping {file_name}: {e}")
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 2. Load Data with Progress Bar
    X, y = load_dataset("./audio_samples")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Logistic Regression
    clf = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test_scaled)
    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred))

    # 7. Save Model and Scaler
    joblib.dump({'model': clf, 'scaler': scaler}, "logistic_chord_model.pkl")
    print("Model and scaler saved to 'logistic_chord_model.pkl'")