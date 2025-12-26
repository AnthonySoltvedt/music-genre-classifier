import os
import numpy as np
import librosa
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from extract_features import extract_features  # Make sure this exists

# Path to dataset
DATASET_PATH = "genres_original"

# Get genres (only directories, skip hidden files)
genres = [d for d in os.listdir(DATASET_PATH)
          if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith(".")]

print("Genres found:", genres)

features = []
labels = []

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    for file_name in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file_name)
        # Only process .wav files and skip hidden files
        if file_name.endswith(".wav") and not file_name.startswith("."):
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(genre)

# Convert to numpy arrays
X = np.array(features)
y_labels = np.array([genres.index(l) for l in labels])
y = to_categorical(y_labels, num_classes=len(genres))

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# Build a simple CNN
model = models.Sequential()
model.add(layers.Input(shape=(X.shape[1], X.shape[2], 1)))  # Adjust if your MFCC is 2D
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(genres), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# Save the model
MODEL_PATH = "genre_model.h5"
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
