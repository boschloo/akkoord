import argparse
import os
import sys
import torch
import torch.nn.functional as F

from spectr_CNN_dataset import audio_to_spectrogram_tensor
from spectr_CNN_model import ChordCNN

MODEL_PATH = "cnn_root_chord_model.pth"

def load_trained_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    idx_to_class = checkpoint['idx_to_class']
    
    model = ChordCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, idx_to_class

def predict_audio_file(file_path, model, idx_to_class, device):
    image_tensor = audio_to_spectrogram_tensor(file_path).unsqueeze(0).to(device) # (1, 1, 128, 128)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        
    predicted_label = idx_to_class[pred_idx.item()]
    confidence_pct = conf.item() * 100
    return predicted_label, confidence_pct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herken akkoorden met een 2D CNN spectrogram model.")
    parser.add_argument("directory", nargs="?", default="unlabeled_samples", help="Pad naar audio bestanden (standaard: unlabeled_samples)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Modelbestand '{MODEL_PATH}' niet gevonden. Run 'python train.py' eerst.")
        sys.exit(1)

    if not os.path.isdir(args.directory):
        print(f"Map niet gevonden: {args.directory}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, idx_to_class = load_trained_model(MODEL_PATH, device)

    wav_files = sorted([f for f in os.listdir(args.directory) if f.endswith('.wav')])
    if not wav_files:
        print(f"Geen .wav bestanden gevonden in '{args.directory}'.")
        sys.exit(0)

    print(f"Verwerken van {len(wav_files)} bestanden uit '{args.directory}':\n")

    for file_name in wav_files:
        file_path = os.path.join(args.directory, file_name)
        try:
            chord, conf = predict_audio_file(file_path, model, idx_to_class, device)
            print(f"{file_name}: Predicted Chord: {chord} ({conf:.1f}%)")
        except Exception as e:
            print(f"{file_name}: Fout bij verwerken ({e})")