import os
import random
import librosa
import numpy as np
from tqdm import tqdm
# dataset downloade link: https://www.kaggle.com/datasets/sdeogade/voice-emotion-classification
input_dir = "/Users/yuezhiluo/Downloads/Voice Emotion Dataset"
output_dir = "/Users/yuezhiluo/Downloads/processed_data"
os.makedirs(output_dir, exist_ok=True)

# Parameter settings
sample_rate = 22050
n_mels = 128
duration = 5
max_length = sample_rate * duration
sample_size = 500 # change if possible


def extract_samples(folder_path, sample_size):
    #Randomly select file paths from the specified folder.
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    sampled_files = random.sample(files, min(sample_size, len(files)))
    return sampled_files


def process_and_save_samples(folder_name):
    folder_path = os.path.join(input_dir, folder_name)
    save_path = os.path.join(output_dir, f"{folder_name}.npy")
    sampled_files = extract_samples(folder_path, sample_size)

    all_audio_data = []
    labels = []
    print(f"Processing category: {folder_name}, extracting {len(sampled_files)} files")

    for file_name in tqdm(sampled_files):
        file_path = os.path.join(folder_path, file_name)
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate)

            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=sr // 2)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            all_audio_data.append(mel_spectrogram_db)
            labels.append(folder_name)
        except Exception as e:
            print(f"Failed to process file: {file_path}, Error: {e}")

    np.save(save_path, np.array(all_audio_data))
    return labels


all_labels = []
categories = ["happy", "fear", "disgust", "anger", "neutral", "sad"]
for category in categories:
    category_labels = process_and_save_samples(category)
    all_labels.extend(category_labels)

np.save(os.path.join(output_dir, "labels.npy"), np.array(all_labels))
print("Done.")
