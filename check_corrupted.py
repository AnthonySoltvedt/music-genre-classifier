import os
import librosa

path = "genres_original/jazz"
for file in os.listdir(path):
    if file.endswith(".wav"):
        file_path = os.path.join(path, file)
        try:
            audio, sr = librosa.load(file_path, sr=22050)
        except Exception as e:
            print(f"Corrupted file: {file_path} -> {e}")


