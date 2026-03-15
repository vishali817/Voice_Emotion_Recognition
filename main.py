import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyaudio
from gtts import gTTS
import pygame
import time
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Data Augmentation Function
def augment_audio(audio_data, sr):
    augmented = []

    # Pitch shift
    augmented.append(librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=4))

    # Slow stretch
    audio_slow = librosa.effects.time_stretch(audio_data, rate=0.8)  # Slow stretch
    augmented.append(audio_slow)

    # Fast stretch
    audio_fast = librosa.effects.time_stretch(audio_data, rate=1.2)  # Fast stretch
    augmented.append(audio_fast)

    # Add noise
    noise = np.random.normal(0, 0.02, audio_data.shape)  # Add noise
    augmented.append(audio_data + noise)

    return augmented

# Function to extract MFCC features
def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    print(f"MFCC Shape: {mfccs.shape}")  # Debugging
    if mfccs.size == 0:
        raise ValueError("No MFCC features could be extracted.")
    return np.mean(mfccs.T, axis=0)

# Normalize Features
def normalize_features(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Define a more complex neural network
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

# Load and augment data
def load_augmented_data():
    emotions = {
        'Happy': r'D:\Vishalii\SER\audio_data\Happy',
        'Sad': r'D:\Vishalii\SER\audio_data\Sad',
        'Angry': r'D:\Vishalii\SER\audio_data\Angry',
        'Neutral': r'D:\Vishalii\SER\audio_data\Neutral'
    }
    data, labels = [], []
    for emotion, folder in emotions.items():
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):
                print(f"Processing file: {file_path}")
                audio_data, sr = librosa.load(file_path)
                features = extract_features(audio_data, sr)
                data.append(features)
                labels.append(list(emotions.keys()).index(emotion))

                # Add augmented samples
                for augmented_audio in augment_audio(audio_data, sr):
                    augmented_features = extract_features(augmented_audio, sr)
                    data.append(augmented_features)
                    labels.append(list(emotions.keys()).index(emotion))
    return normalize_features(np.array(data)), np.array(labels)

# Train the model
def train_model():
    data, labels = load_augmented_data()
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = EmotionRecognitionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

    print("Model training complete!")
    return model

# Real-time emotion prediction with frequency and energy-based rules
def predict_emotion_with_frequency(audio_data, sr, model):
    # Extract fundamental frequency
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
    avg_frequency = np.mean(pitches[pitches > 0])  # Average pitch above 0

    # Calculate RMS energy
    rms_energy = librosa.feature.rms(y=audio_data)[0]
    avg_energy = np.mean(rms_energy)  # Average energy
    threshold_energy = 0.05  # Example threshold for energy

    # Rule-based classification based on frequency and energy
    if avg_frequency > 250 and avg_energy > threshold_energy:
        return "Angry"
    elif avg_frequency < 100 and avg_energy < threshold_energy:  # Slightly increase the range for "Sad"
        return "Sad"
    elif 150 < avg_frequency < 250 and threshold_energy * 0.5 < avg_energy < threshold_energy * 1.5:
        return "Happy"  # Adjusted thresholds for "Happy"
    elif 100 < avg_frequency < 150 and avg_energy < threshold_energy * 1.2:
        return "Neutral"  # New range for "Neutral"

    else:
        # Use the model for other emotions
        features = extract_features(audio_data, sr)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(features)
        emotion_label = torch.argmax(output, dim=1).item()
        emotions = ['Angry', 'Sad', 'Happy', 'Neutral']  # Ensure order matches training
        return emotions[emotion_label]

# Function to capture live audio
def capture_live_audio():
    CHUNK = 1024  # Buffer size
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1  # Mono
    RATE = 44100  # Sample rate
    DURATION = 5  # Duration in seconds

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")

    frames = []
    for _ in range(int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.hstack(frames)
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize audio data
    return audio_data, RATE

# Function to convert text to speech in Tamil
def speak(text):
    tts = gTTS(text=text, lang='ta')
    filename = "response.mp3"
    tts.save(filename)

    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Stop and unload the music file
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

    # Remove the audio file after playing
    os.remove(filename)

# Main loop for real-time emotion recognition and suggestions
def main():
    model = train_model()  # Train the model
    while True:
        speak("நீங்கள் பேச தயாரா? தயவுசெய்து பேசுங்கள்.")  # Tamil: "Are you ready to speak? Please speak."
        time.sleep(3)  # Provide time for the user to start speaking
        audio_data, sr = capture_live_audio()
        emotion = predict_emotion_with_frequency(audio_data, sr, model)
        print(f"கண்டறியப்பட்ட உணர்வு: {emotion}")  # Tamil: "Detected Emotion:"

        if emotion == "Angry":
            response = "உங்கள் குரல் மிகவும் கூர்மையாக உள்ளது. கவனமாக இருந்தால் உதவும்."  # Tamil: "Your voice is very sharp. Staying calm might help."
        elif emotion == "Sad":
            response = "நான் உங்களுக்காக இருக்கிறேன். இசையைக் கேட்பது உங்களை நன்றாகச் செய்யலாம்."  # Tamil: "Your voice indicates sadness. I'm here for you."
        elif emotion == "Happy":
            response = "நீங்கள் மகிழ்ச்சியாக இருக்கிறீர்கள்! 😊"  # Tamil: "You are happy! 😊"
        elif emotion == "Neutral":
            response = "உங்கள் உணர்வு நிலை சீராக உள்ளது."  # Tamil: "Your emotional state is stable."

        print(response)
        speak(response)

if __name__ == "__main__":
    main()
