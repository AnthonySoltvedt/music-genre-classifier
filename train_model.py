# Used to work with folders and file paths
import os

# Used for numerical arrays and matrix operations
import numpy as np

# Used to build neural network layers and models
from tensorflow.keras import layers, models

# Converts labels into one-hot encoded vectors
from tensorflow.keras.utils import to_categorical

# Imports the MFCC feature extraction function
from extract_features import extract_features

# Path to the dataset folder
DATASET_PATH = "genres_original"

# Gets a sorted list of genre folder names
genres = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

# Prints all detected genres
print("Genres:", genres)

# Stores extracted MFCC features
features = []

# Stores numerical genre labels
labels = []

# Loops through each genre
for genre in genres:
# Creates the full path to the genre folder
    genre_path = os.path.join(DATASET_PATH, genre)
# Loops through each audio file in the genre
    for file in os.listdir(genre_path):
# Ensures only WAV files are processed
        if file.endswith(".wav"):
# Builds the full file path
            file_path = os.path.join(genre_path, file)
# Extracts MFCC features from the audio
            mfcc = extract_features(file_path)
# Adds MFCC features to the feature list
            features.append(mfcc)
# Adds the genre index as the label
            labels.append(genres.index(genre))


# Converts feature list into a NumPy array
X = np.array(features)

# Converts labels into one-hot encoded format
y = to_categorical(labels, num_classes=len(genres))

# Adds a channel dimension for CNN input
X = X[..., np.newaxis]

# Prints shape of the feature array
print("X shape:", X.shape)

# Prints shape of the label array
print("y shape:", y.shape)

# Creates a sequential neural network model
model = models.Sequential([
# Defines the input shape of the MFCC data
    layers.Input(shape=(40, 174, 1)),
# First convolutional layer
    layers.Conv2D(32, (3, 3), activation="relu"),
# Reduces spatial dimensions
    layers.MaxPooling2D((2, 2)),
# Second convolutional layer
    layers.Conv2D(64, (3, 3), activation="relu"),
# Further reduces dimensions
    layers.MaxPooling2D((2, 2)),
# Converts 2D feature maps into 1D vector
    layers.Flatten(),
# Output layer for genre classification
    layers.Dense(128, activation="relu"),
    layers.Dense(len(genres), activation="softmax")
])

model.compile(
    optimizer="adam",
# Uses Adam optimizer
    loss="categorical_crossentropy",
# Loss function for multi-class classification
    metrics=["accuracy"]
# Tracks accuracy during training
)

# Prints model architecture
model.summary()

# Number of training cycles
model.fit(
    X, y,
    epochs=30,
# Number of samples per training step
    batch_size=32,
# Uses 20% of data for validation
    validation_split=0.2
)
# Saves the trained model to disk
model.save("genre_model.h5")
# Confirms the model was saved
print("Model saved.")
