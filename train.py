import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle

# 1. Identical Model Architecture
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

# 2. Reusable Feature Extraction Function
def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# 3. Data Augmentation Function
def augment_audio(audio_data, sr):
    augmented = []
    # Pitch shift
    augmented.append(librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=4))
    # Slow stretch
    augmented.append(librosa.effects.time_stretch(audio_data, rate=0.8))
    # Fast stretch
    augmented.append(librosa.effects.time_stretch(audio_data, rate=1.2))
    # Add gaussian noise
    noise = np.random.normal(0, 0.02, audio_data.shape)
    augmented.append(audio_data + noise)
    return augmented

def train():
    # Exact label order as requested
    emotion_list = ["Angry", "Sad", "Happy", "Neutral"]
    base_path = "audio_data"
    
    data, labels = [], []
    
    print("Loading and augmenting data...")
    for idx, emotion in enumerate(emotion_list):
        folder_path = os.path.join(base_path, emotion)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                # sr=None preserves the native sampling rate
                audio, sr = librosa.load(file_path, sr=None)
                
                # Original Sample
                feat = extract_features(audio, sr)
                data.append(feat)
                labels.append(idx)
                
                # Augmented Samples
                for aug_audio in augment_audio(audio, sr):
                    aug_feat = extract_features(aug_audio, sr)
                    data.append(aug_feat)
                    labels.append(idx)

    # 4. Normalize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data))
    y = np.array(labels)
    
    # Save scaler to ensure consistent preprocessing during inference
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 5. Training Configuration
    model = EmotionRecognitionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
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
        
        # 7. Print progress
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss/len(dataloader):.4f}")
        
    # 6. Save Model
    torch.save(model.state_dict(), "emotion_model.pth")
    print("\nTraining complete! Saved 'emotion_model.pth' and 'scaler.pkl'.")

if __name__ == "__main__":
    train()
