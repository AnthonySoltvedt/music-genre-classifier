import os
import librosa

# Path to your jazz folder
path = "genres_original/jazz"

# Loop through all .wav files
for file in os.listdir(path):
    if file.endswith(".wav"):
        file_path = os.path.join(path, file)
        try:
            # Try to load the audio
            audio, sr = librosa.load(file_path, sr=22050)
            print(f"Loaded {file}: {audio.shape}, Sample rate: {sr}")
            # Here you can add code to extract features or process audio
        except Exception as e:
            # If loading fails, print a warning and skip
            print(f"Corrupted or unreadable file: {file_path} -> {e}")
