# train_model.py
import os
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from extract_features import extract_features  # Make sure this file exists

# Path to the dataset
DATASET_PATH = "genres_original"

# Detect genre folders (skip hidden files)
genres = sorted([d for d in os.listdir(DATASET_PATH)
                 if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')])
print("Genres detected:", genres)

# Lists to store features and labels
features = []
labels = []

# Loop through each genre folder
for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    for file_name in os.listdir(genre_path):
        if not file_name.endswith(".wav") or file_name.startswith('.'):
            continue  # Skip non-WAV or hidden files

        file_path = os.path.join(genre_path, file_name)

        # Extract MFCC features
        mfccs = extract_features(file_path)

        # Optional: normalize MFCCs
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

        features.append(mfccs)
        labels.append(genres.index(genre))

# Convert lists to NumPy arrays
X = np.array(features)
y = to_categorical(labels, num_classes=len(genres))

# Add channel dimension for CNN input
X = X[..., np.newaxis]

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2], 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(genres), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train model
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# Save trained model
MODEL_PATH = "genre_model.h5"
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
