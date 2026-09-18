import librosa
import numpy as np


# De vijf akkoorden die we voorlopig herkennen.
CHORDS = {
    "C":  [0, 4, 7],       # C E G
    "Am": [9, 0, 4],       # A C E
    "F":  [5, 9, 0],       # F A C
    "G":  [7, 11, 2],      # G B D
    "Bdim": [11,2,5]       # B D F
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

    print("\nAkkoorden:")
    for chord, score in sorted_scores:
        print(f"{chord:2s}: {score:.3f}")

    best_chord = sorted_scores[0][0]

    return best_chord


if __name__ == "__main__":
    filename = "test_g_maj.wav"

    chord = recognize_chord(filename)

    print(f"\n>>> Herkend akkoord: {chord}")
