# Library for loading and processing audio files
import librosa
# For numeral operations like padding arrays
import numpy as np

def extract_features(file_path, max_pad_len=174):
    # Load the audio file and convert it to waveform data
    # sr=22050 sets the same rate 22.05 kHz (standard for ML audio work)
    audio, sr = librosa.load(file_path, sr=22050)

    # Compute the MFCC feature matrix from the audio signal
    # n_mfcc=40 means we extract 40 coefficients per frame
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # If the MFCC sequence is shorter than the target length, pad it with zeros
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0,pad_width)))

    # if it longer, cut (truncate) it to the target length
    else:
        mfcc = mfcc[:, :max_pad_len]
    # Return the final fixed-size MFCC feature array
    return mfcc
