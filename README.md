# akkoord
Project voor het OU-vak Key Topics

**Investigated combinations of audio representation and model type**

|  # | Filename | Audio Representation | Model  |
|-------|-------|----------------------|--------| 
| 1|[chords_no_ML.py](https://github.com/boschloo/akkoord/blob/main/chords_no_ML.py) | chromagram | - |
| 2|[spectr_rf_train.py](https://github.com/boschloo/akkoord/blob/main/spectr_rf_train.py) | MEL spectrogram | Random Forest Classifier |
| 3|[spectr_logistic_train.py](https://github.com/boschloo/akkoord/blob/main/spectr_logistic_train.py)| MEL spectrogram | Logistic Regression |
| 4|[spectr_CNN_train.py](https://github.com/boschloo/akkoord/blob/main/spectr_CNN_train.py)| MEL spectrogram | 2D Convolutional Neural Network |
| 5|[chroma_rf_train.py](https://github.com/boschloo/akkoord/blob/main/chroma_rf_train.py)| chromagram | Random Forest Classifier | 
| 6|[chroma_logistic_train.py](https://github.com/boschloo/akkoord/blob/main/chroma_logistic_train.py) | chromagram | Logistic Regression |
| 7|[chroma_CNN_train.py](https://github.com/boschloo/akkoord/blob/main/chroma_CNN_train.py) | chromagram | 2D Convolutional Neural Network |


**Dataset**

The [audio_samples](https://github.com/boschloo/akkoord/tree/main/audio_samples) folder contains 9,102 chord samples. The chords were generated using the notation software [Dorico](https://www.steinberg.net/dorico/) by Steinberg. Various software instruments were used, all of which are from the included HALion Sonic SE3 library.

The file format is WAV, with a sample rate of 48,000 Hz, 16-bit signed PCM, and mono. The original recordings took up a significant amount of space, so they have been trimmed to 2 seconds per sample.

Naming convention for the chord samples:


| Code    |   Possible values         | Explanation|
|---------|--------------------------|------------|
| Abbreviation instrument   |<ul><li>asx: alto saxophone</li><li>blf: recorder</li><li>cel: violoncello</li><li>cbs: double bass</li><li>hbo: oboe</li><li>klt: clarinet</li><li>org: church organ</li><li>pno: piano</li><li>trb:trombone</li><li>tpt: trumpet</li></ul>|  |
| Pitch  |<ul><li>435 Hz</li><li>440 Hz</li><li>442 Hz</li></ul> | Pitch of A4 (the first A to the right of the center of the piano ;)   |
| Character of the chord   | <ul><li>maj: major</li><li> min: minor</li><li> dim: diminished</li><li>aug: augmented</li></ul>|
Inversion of the chord    |<ul><li>123: base</li><li>231: first inversion</li><li>312: second inversion</li></ul> |By definition, diminished and augmented chords only exist in root position.
| Source of sample| <ul><li>drc: Dorico</li><li>liv: live played and/ or composed</li></ul> | 

The number of samples varies per musical instrument, depending on the instrument's range (low to high). 
 
Directory [unlabeled_samples](https://github.com/boschloo/akkoord/tree/main/unlabeled_samples) is intended for .wav files containing unknown chords, which the model should be able to recognize following a training phase. The names of these files can be chosen arbitrarily.
 
Directory [unlabeled_samples_run04](https://github.com/boschloo/akkoord/tree/main/unlabeled_samples_run04) contains a set of 20 live played samples by different instruments.


**Instructions**

Python scripts ending in "-train" use data from the [audio_samples](https://github.com/boschloo/akkoord/tree/main/audio_samples) directory to train the model. These scripts use the naming convention of the .wav files to label the features. Specifically, the model learns 12 pitch classes (C, C#, etc.) and the chord quality (major, minor, diminished, or augmented), resulting in 48 possible chord type combinations.

Python scripts ending in "-predict" read .wav data from the [unlabeled_samples](https://github.com/boschloo/akkoord/tree/main/unlabeled_samples) directory and use the model to predict the chord types.
*Example:*

>`Projects/akkoord> python3 spectr_rf_train.py`


>`Projects/akkoord> python3 spectr_rf_predict.py`


A [requirements.txt](https://github.com/boschloo/akkoord/blob/main/requirements.txt) files can be used to set up the environment.

**Results**

The file [output.txt](https://github.com/boschloo/akkoord/blob/main/output.txt) shows a prelimiary result.

The file [output_run04.txt](https://github.com/boschloo/akkoord/blob/main/output_run04.txt) show results with the live played test set.

Generative AI - specifically Google (2026) Gemini 3.6 Flash - was used during the programming process. [https://gemini.google.com]()
