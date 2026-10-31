import librosa
import numpy as np
import sys
import argparse
import os
import sys

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]
NOTE = {name: i for i, name in enumerate(NOTE_NAMES)}

# Intervals in semitones above the root, per chord type.
QUALITIES = {
    "maj":    [0, 4, 7],   # major: root, major third, fifth
    "min":   [0, 3, 7],   # minor: root, minor third, fifth
    "dim": [0, 3, 6],   # diminished: root, minor third, diminished fifth
    "aug": [0, 4, 8],   # augmented: root, major third, augmented fifth
}

# Builds all 12 x 4 = 48 chords: "C", "Cm", "Cdim", "Caug", "C#", ...
CHORDS = {
    root + suffix: [(NOTE[root] + interval) % 12 for interval in intervals]
    for root in NOTE_NAMES
    for suffix, intervals in QUALITIES.items()
}

def chord_score(chroma, chord_notes):
    """
    Bereken hoe goed een chromagram overeenkomt
    met de noten van een akkoord.
    """

    score = 0

    # Positieve bijdrage voor noten die in het akkoord zitten
    for note in chord_notes:
        score += chroma[note]

    # Negatieve bijdrage voor sterke noten die niet
    # in het akkoord zitten.
    other_notes = [i for i in range(12) if i not in chord_notes]

    for note in other_notes:
        score -= 0.2 * chroma[note]

    return score

def recognize_chord(filename):
    """
    Lees een audio-opname in en probeer het akkoord
    te herkennen.
    """

    print(f"Audio laden: {filename}")

    # y = audio-signaal
    # sr = sample rate
    y, sr = librosa.load(filename, sr=None, mono=True)
    MIN_DURATION = 1.0  # seconds
    min_samples = int(MIN_DURATION * sr)

    if len(y) < min_samples:
        print(f"Waarschuwing: opname is kort, aangevuld met stilte tot {MIN_DURATION} s")
        y = np.pad(y, (0, min_samples - len(y)), mode="constant")
    

    print(f"Sample rate: {sr} Hz")
    print(f"Duur: {len(y) / sr:.2f} seconden")

    # Bereken chromagram.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Gemiddelde sterkte van iedere toonklasse.
    mean_chroma = np.mean(chroma, axis=1)

    # Normaliseren
    mean_chroma = mean_chroma / np.max(mean_chroma)

    print("\nToonsterktes:")
    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    for note, value in zip(note_names, mean_chroma):
        print(f"{note:2s}: {value:.3f}")

    # Bereken score voor ieder akkoord.
    scores = {}

    for chord, notes in CHORDS.items():
        scores[chord] = chord_score(mean_chroma, notes)

    # Sorteer van hoog naar laag.
    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # print("\nAkkoorden:")
    # for chord, score in sorted_scores:
    #     print(f"{chord:2s}: {score:.3f}")

    best_chord = sorted_scores[0][0]

    return best_chord

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Herken accoorden in een mappenstructuur van audiobestanden."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="unlabeled_samples",
        help="pad naar de map met audiobestanden (standaard: unlabeled_samples)"
    )
    args = parser.parse_args()

    target_dir = args.directory

    if not os.path.isdir(target_dir):
        print(f"Map niet gevonden: {target_dir}")
        sys.exit(1)

    # Collect and sort all .wav files in the directory
    wav_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.wav')])

    if not wav_files:
        print(f"Geen .wav bestanden gevonden in: {target_dir}")
        sys.exit(0)

    print(f"Verwerken van {len(wav_files)} bestanden uit '{target_dir}':\n")

    for file_name in wav_files:
        file_path = os.path.join(target_dir, file_name)
        try:
            chord = recognize_chord(file_path)
            print(f"{file_name}: Predicted Chord: {chord}\n")
        except Exception as e:
            print(f"{file_name}: Fout bij verwerken ({e})\n")