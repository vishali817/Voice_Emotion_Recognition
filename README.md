# Voice Emotion Recognition (SER)

A real-time Voice Emotion Recognition system that identifies human emotions from speech. This project utilizes deep learning (PyTorch) and rule-based frequency analysis to classify emotions such as Happy, Sad, Angry, and Neutral. It also provides interactive feedback in Tamil using text-to-speech.

## 🚀 Features

- **Real-time Detection**: Captures live audio from the microphone and predicts emotions instantly.
- **Deep Learning Model**: Uses a PyTorch-based neural network trained on MFCC (Mel-frequency cepstral coefficients) features.
- **Rule-based Enhancement**: Combines model predictions with fundamental frequency and energy analysis for better accuracy.
- **Data Augmentation**: Includes pitch shifting, time stretching, and noise addition to improve model robustness.
- **Tamil Interaction**: Provides voice prompts and emotional feedback in Tamil using `gTTS`.

## 🛠️ Technologies Used

- **Python**
- **PyTorch**: Deep Learning framework for the emotion classification model.
- **Librosa**: Audio processing and feature extraction (MFCC, Pitch, RMS).
- **PyAudio**: Real-time audio stream capture.
- **gTTS (Google Text-to-Speech)**: Generating speech responses in Tamil.
- **Pygame**: Audio playback for speech responses.
- **Scikit-learn**: Feature normalization using `StandardScaler`.

## 📋 Prerequisites

Before running the project, ensure you have Python installed and the following dependencies:

```bash
pip install -r requirements.txt
```

*Note: You may need to install `portaudio` on your system for `pyaudio` to work.*

## 📂 Project Structure

```text
SER/
├── audio_data/          # Training dataset organized by emotion folders
│   ├── Happy/
│   ├── Sad/
│   ├── Angry/
│   └── Neutral/
├── main.py              # Main execution script
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## ⚙️ How It Works

1.  **Training Phase**: The script automatically loads audio files from the `audio_data` folder, performs data augmentation, extracts features, and trains a neural network.
2.  **Live Capture**: Once trained, it uses `pyaudio` to record 5 seconds of live audio.
3.  **Analysis**:
    *   Extracts MFCC features.
    *   Analyzes fundamental frequency (pitch) and RMS energy.
    *   Classifies the emotion using rule-based logic and the trained model.
4.  **Feedback**: The system announces the detected emotion and provides a suggestion in Tamil.

## 🏃 Running the Project

To start the application, run:

```bash
python main.py
```

Follow the Tamil voice prompts to speak when ready.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/vishali817/Voice_Emotion_Recognition/issues).

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
