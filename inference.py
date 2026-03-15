import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyaudio
import pickle

# Identical Model Architecture
class EmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(13, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Reusable Feature Extraction
def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Capture Microphone Audio
def capture_live_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    DURATION = 5

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Recording...")
    frames = []
    for _ in range(int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.hstack(frames)
    # int16 -> float32 normalization
    audio_data = audio_data.astype(np.float32) / 32768.0
    return audio_data, RATE

def main():
    # 10. Load model and scaler at the beginning (only once)
    model = EmotionRecognitionModel()
    try:
        model.load_state_dict(torch.load("emotion_model.pth"))
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("Error: Required files (emotion_model.pth/scaler.pkl) not found. Run train.py first.")
        return
        
    model.eval()
    emotion_list = ["Angry", "Sad", "Happy", "Neutral"]

    # Capture audio
    audio_data, sr = capture_live_audio()
    
    # Extract features
    feat = extract_features(audio_data, sr)
    
    # Normalize features using the saved scaler
    feat = scaler.transform([feat])
    
    # 6. Convert to PyTorch tensor
    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    
    # 7. Run prediction
    with torch.no_grad():
        output = model(feat_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    # 8-9. Print result
    print(f"Detected Emotion: {emotion_list[prediction]}")

if __name__ == "__main__":
    main()
