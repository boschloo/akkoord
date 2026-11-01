# akkoord
Project voor het OU-vak Key Topics

**Onderzochte combinaties van audiorepresentatie en modeltype**

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

De map [audio_samples](https://github.com/boschloo/akkoord/tree/main/audio_samples) bevat 9102 samples van akkoorden. De akkoorden zijn gegenereerd met behulp van het notatieprogramma [Dorico](https://www.steinberg.net/dorico/%5D(Dorico)) van Steinberg.  Er zijn verschillende software instrumenten gebruikt. De instrumenten zijn afkomstig uit de meegeleverde HALion Sonic SE3-bibliotheek. 

Het bestandsformaat is WAV, sample rate 48000 Hz, signed 16-bit PCM, mono. De oorspronkelijke opnames namen veel ruimte in en zijn daarom ingekort tot 2 seconden per sample.
 
Naamgeving van de akkoordsamples:


| Code    |   Mogelijke waarden         | Toelichting|
|---------|--------------------------|------------|
| Afkorting instrument   |<ul><li>asx: altsaxofoon</li><li>blf: blokfluit</li><li>cel: cello</li><li>cbs: contrabas</li><li>hbo: hobo</li><li>klt:klarinet</li><li>org: kerkorgel</li><li>pno: piano</li><li>trb:trombone</li><li>tpt: trompet</li></ul>|  |
| Stemming  |<ul><li>435 Hz</li><li>440 Hz</li><li>442 Hz</li></ul> | toonhoogte van A (de eerste A rechts van het midden van de piano ;)   |
| Karakter van het akkoord   | <ul><li>maj: majeur</li><li> min: mineur</li><li> dim: verminderd</li><li>aug: overmatig</li></ul>|
Ligging van het akkoord    |<ul><li>123: grondligging</li><li>231: eerste omkering</li><li>312: tweede omkering</li></ul> |Verminderde en overmatige akkoorden per definitie alleen in de grondligging
| Bron van de sample| <ul><li>drc: Dorico</li><li>liv: live ingespeeld en/of samengesteld</li></ul> | 

De aantallen samples verschillen per instrument, afhankelijk van het bereik (laag-hoog). 
 
De map [unlabeled_samples](https://github.com/boschloo/akkoord/tree/main/unlabeled_samples) is bedoeld voor .wav-bestanden met onbekende akkoorden die na het leren door het model herkend moeten kunnen worden. De naam van deze bestanden kan. willekeurig worden gekozen.
 
**Gebruiksaanwijzing**

De Python-scripts die op -train eindigen, gebruiken data uit de map [audio_samples](https://github.com/boschloo/akkoord/tree/main/audio_samples) om het model te leren. Deze scripts gebruiken de naamgeving van deze .wav-bestanden voor het labelen van features. Van de mogelijke features worden de 12 toonhoogtes (C_, C#,..,) en het karakter (alleen maj, min, dim, aug) geleerd. DIt geeft 48 mogelijke combinaties van akkoordtypes.

Python-scripts eindigend op -predict lezen .wav-data in de map [unlabeled_samples](https://github.com/boschloo/akkoord/tree/main/unlabeled_samples). Met behulp van het model worden de akkoordtypes voorspeld.

*Voorbeeld:*

>`Projects/akkoord> python3 spectr_rf_train.py`


>`Projects/akkoord> python3 spectr_rf_predict.py`


Er is een [requirements.txt](https://github.com/boschloo/akkoord/blob/main/requirements.txt) file om de omgeving in te stellen.

**Resultaten en conclusies**

Bestand output.txt toont een voorlopig resultaat. Uitwerking: -- TODO --

