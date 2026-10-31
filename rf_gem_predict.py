import os
import joblib
import numpy as np
import librosa

MODEL_PATH = "root_chord_model.pkl"
UNLABELED_DIR = "./unlabeled_samples"

def extract_features(file_path):
    """Must match the exact feature extraction logic used during training."""
    y, sr = librosa.load(file_path, sr=22050)
    
    n_samples = len(y)
    if n_samples < 128:
        return np.zeros(24)
        
    n_fft = min(2048, 2 ** int(np.log2(n_samples)))
    hop_length = max(32, n_fft // 4)
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    return np.hstack([chroma_mean, chroma_std])

def predict_folder(folder_path, model):
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' does not exist.")
        return

    # Find all .wav files and sort them alphabetically
    wav_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])
    
    if not wav_files:
        print(f"No .wav files found in '{folder_path}'.")
        return

    print(f"Found {len(wav_files)} files to classify:\n")

    for file_name in wav_files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            features = extract_features(file_path)
            features_2d = features.reshape(1, -1)
            
            # Predict chord label and get confidence score
            prediction = model.predict(features_2d)[0]
            confidence = np.max(model.predict_proba(features_2d)[0])
            
            print(f"{file_name}: Predicted Chord: {prediction} ({confidence * 100:.1f}%)")
        except Exception as e:
            print(f"{file_name}: Error processing file ({e})")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
        predict_folder(UNLABELED_DIR, clf)
    else:
        print(f"Model file '{MODEL_PATH}' not found. Train and save the model first.")