# 🎙️ echomood

> Classifying human emotion from raw speech using deep learning on the RAVDESS dataset.

`Speech-ER` is a Speech Emotion Recognition (SER) system that extracts MFCC acoustic features from audio files and classifies them into one of eight emotions using a stacked Bidirectional LSTM neural network built with TensorFlow/Keras.

---

## Emotions Detected

| Code | Emotion   |
|------|-----------|
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

---

## How It Works

1. **Feature Extraction** — Each `.wav` file is loaded with `librosa`. 13 MFCC coefficients are extracted and averaged across time to produce a fixed-length feature vector per audio clip.

2. **Model Architecture** — A stacked Bidirectional LSTM with dropout regularization and a dense output layer with softmax activation.

```
BiLSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(8, Softmax)
```

3. **Training** — Trained for up to 200 epochs with early stopping, using the Adam optimizer and sparse categorical cross-entropy loss.

4. **Inference** — Given a new `.wav` file, the model extracts features and returns a predicted emotion label.

---

## Dataset

This project uses the [RAVDESS dataset](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song).

Download it and place it at:

```
ravdess_dataset/
└── audio_speech_actors_01-24/
    ├── Actor_01/
    ├── Actor_02/
    ...
    └── Actor_24/
```

Then update the path in the notebook:

```python
data_path = r"path/to/ravdess_dataset/audio_speech_actors_01-24"
```

---

## Requirements

- Python 3.11+
- TensorFlow 2.18+
- librosa
- soundfile
- numpy
- matplotlib
- scikit-learn

Install all dependencies:

```bash
pip install tensorflow librosa soundfile numpy matplotlib scikit-learn
```

---

## Usage

Run `SSR.ipynb` end-to-end to:

- Extract MFCC features from all actors and audio files
- Train the Bi-LSTM model
- Evaluate on the test set
- Run inference on a single audio file

To predict the emotion of a new file:

```python
audio_file = "path/to/your/audio.wav"
predicted_emotion = predict_emotion(model, audio_file)
print(f"Predicted Emotion: {predicted_emotion}")
```

---

## Project Structure

```
Speech-ER/
│
├── SSR.ipynb             # Main notebook: training, evaluation, inference
├── emotion_model.h5      # Saved trained model (generated after training)
├── README.md
└── .gitignore
```

---

## Notes

- The saved model file `emotion_model.h5` is included in the repo. You can skip training and load it directly:

```python
from tensorflow.keras.models import load_model
model = load_model('emotion_model.h5')
```

- RAVDESS audio files are not included due to size. Download them from the link above.
