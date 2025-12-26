from pydub import AudioSegment
import os
import shutil

# Path to your jazz files
path = "genres_original/jazz"
corrupted_folder = os.path.join(path, "corrupted_files")

# Create the folder if it doesn't exist
os.makedirs(corrupted_folder, exist_ok=True)

for file in os.listdir(path):
    if file.endswith(".wav"):
        file_path = os.path.join(path, file)
        try:
            # Try to load the audio
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            print(f"Corrupted file: {file_path} -> {e}")
            # Move corrupted file
            shutil.move(file_path, os.path.join(corrupted_folder, file))

print("Corrupted files moved to:", corrupted_folder)


