import argparse
import os
import sys
import warnings
import joblib
import librosa
import numpy as np

# 1. Suppress the internal librosa Constant-Q filter window size warning
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="librosa.core.constantq"
)

MODEL_PATH = "logistic_chord_model.pkl"

def extract_chroma_features(file_path):
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

def recognize_chord(file_path, model, scaler):
    """Extracts features, scales them using the saved scaler, and predicts the chord."""
    features = extract_chroma_features(file_path)
    features_2d = features.reshape(1, -1)
    
    # Scale features using the fitted StandardScaler from training
    features_scaled = scaler.transform(features_2d)
    
    prediction = model.predict(features_scaled)[0]
    confidence = np.max(model.predict_proba(features_scaled)[0])
    
    return prediction, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Herken akkoorden in een mappenstructuur met het Logistische Regressie model."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="unlabeled_samples",
        help="pad naar de map met audiobestanden (standaard: unlabeled_samples)"
    )
    args = parser.parse_args()

    # Verify model artifact exists
    if not os.path.isfile(MODEL_PATH):
        print(f"Modelbestand niet gevonden: {MODEL_PATH}. Run 'train_logistic.py' eerst.")
        sys.exit(1)

    # Verify target directory exists
    target_dir = args.directory
    if not os.path.isdir(target_dir):
        print(f"Map niet gevonden: {target_dir}")
        sys.exit(1)

    # Load dictionary containing both model and scaler
    saved_artifacts = joblib.load(MODEL_PATH)
    clf = saved_artifacts['model']
    scaler = saved_artifacts['scaler']

    # Collect and sort all .wav files in the directory
    wav_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.wav')])

    if not wav_files:
        print(f"Geen .wav bestanden gevonden in: {target_dir}")
        sys.exit(0)

    print(f"Verwerken van {len(wav_files)} bestanden uit '{target_dir}':\n")

    for file_name in wav_files:
        file_path = os.path.join(target_dir, file_name)
        try:
            chord, conf = recognize_chord(file_path, clf, scaler)
            print(f"{file_name}: Predicted Chord: {chord} ({conf * 100:.1f}%)")
        except Exception as e:
            print(f"{file_name}: Fout bij verwerken ({e})")