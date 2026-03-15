# Voice Emotion Recognition (SER)

A real-time Voice Emotion Recognition system that identifies human emotions from speech. This project utilizes deep learning (PyTorch) to classify emotions such as **Happy, Sad, Angry, and Neutral**.

## 🚀 Features

- **Fast Inference**: Split training and inference logic for near-instant detection.
- **Microphone Capture**: Uses `PyAudio` for 5-second real-time voice recording.
- **Deep Learning Model**: A PyTorch neural network trained on 13 MFCC features.
- **Data Augmentation**: Includes pitch shifting, time stretching (slow/fast), and white noise to improve accuracy.
- **Feature Normalization**: Uses `StandardScaler` (saved as `scaler.pkl`) for consistent preprocessing.

## 🛠️ Technologies Used

- **Python**
- **PyTorch**: Model architecture and training.
- **Librosa**: Audio feature extraction and augmentation.
- **PyAudio**: Real-time microphone stream input.
- **Scikit-learn**: Feature scaling and normalization.
- **Pickle**: Storing the feature scaler object.

## 📋 Prerequisites

Install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

*Note: Windows users may need to install `PyAudio` via a wheel or ensuring C++ build tools are installed.*

## 📂 Project Structure

```text
SER/
├── audio_data/          # Training dataset categorized by emotion
│   ├── Angry/
│   ├── Happy/
│   ├── Neutral/
│   └── Sad/
├── train.py             # Script to train the model and save weights
├── inference.py         # Script for real-time emotion detection
├── emotion_model.pth    # Saved PyTorch model weights
├── scaler.pkl           # Saved feature scaler
├── requirements.txt     # List of hardware/software dependencies
└── README.md            # Documentation
```

## ⚙️ How It Works

1.  **Feature Extraction**: Audio is sampled at 44.1kHz and processed into 13 MFCC features.
2.  **Training (`train.py`)**: 
    - Loads `.wav` files and applies data augmentation.
    - Trains a 4-layer fully connected neural network.
    - Saves the model and the scaler for later use.
3.  **Inference (`inference.py`)**:
    - Loads the pre-trained model and scaler **once** at startup.
    - Records 5 seconds of audio.
    - Predicts the emotion using the trained weights.

## 🏃 Running the Project

### 1. Training (Optional)
If you want to retrain the model with your own data:
```bash
python train.py
```

### 2. Inference
To detect emotions in real-time from your microphone:
```bash
python inference.py
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/vishali817/Voice_Emotion_Recognition/issues).

## 📄 License

Distributed under the MIT License.
