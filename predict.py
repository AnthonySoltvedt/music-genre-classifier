# predict.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from extract_features import extract_features

# Path to the saved trained model
MODEL_PATH = "genre_model.h5"

# Load the trained genre classification model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# List of genres in the SAME order as used during training
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Predicts the genre of a single audio file
def predict_genre(file_path):
    try:
        # Extract MFCC features
        features = extract_features(file_path)

        # Optional: normalize features (same as training)
        features = (features - np.mean(features)) / np.std(features)

        # Add batch and channel dimensions
        features = features[np.newaxis, ..., np.newaxis]

        # Run model prediction
        prediction = model.predict(features, verbose=0)

        # Get predicted genre index
        predicted_index = np.argmax(prediction)

        # Return predicted genre name
        return genres[predicted_index]

    except Exception as e:
        print(f"Error predicting {file_path}: {e}")
        return None

# Example usage
audio_file = "my_song.wav"

if os.path.exists(audio_file):
    genre = predict_genre(audio_file)
    print(f"Predicted genre: {genre}")
else:
    print(f"Audio file '{audio_file}' not found.")
