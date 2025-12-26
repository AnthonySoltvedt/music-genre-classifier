# Used for numerical operations and arrays
import numpy as np
# Used to load the trained neural network model
from tensorflow.keras.models import load_model
# Imports the MFCC feature extraction function
from extract_features import extract_features
# Used to check file paths and file existence
import os

# Path to the saved trained model
MODEL_PATH = "genre_model.h5"
# Loads the trained genre classification model
model = load_model(MODEL_PATH)

# Confirms the model was loaded
print("Model loaded successfully.")

# List of genres in the same order used during training
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


# Predicts the genre of a single audio file
def predict_genre(file_path):
    try:
# Extracts MFCC features from the audio file
        features = extract_features(file_path)
# Extracts MFCC features from the audio file
        features = features[np.newaxis, ..., np.newaxis]  # add batch and channel dims
# Runs the model to get prediction probabilities
        prediction = model.predict(features)
# Finds the index of the highest predicted probability
        predicted_index = np.argmax(prediction)
# Returns the predicted genre name
        return genres[predicted_index]
# Catches errors during prediction
    except Exception as e:
# Returns None if prediction fails
        print(f"Error predicting {file_path}: {e}")
        return None

# Path to the audio file to be classified
audio_file = "my_song.wav"

# Checks if the audio file exists
if os.path.exists(audio_file):
# Predicts the genre of the audio file
    genre = predict_genre(audio_file)
# Prints the predicted genre
    print(f"Predicted genre: {genre}")
# Runs if the file does not exist
else:
# Prints an error message if file is missing
    print(f"Audio file '{audio_file}' not found.")
