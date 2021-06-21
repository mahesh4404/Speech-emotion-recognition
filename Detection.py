from numpy import loadtxt
import IPython.display as ipd
import librosa
import numpy as np
from keras.models import load_model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # to use operating system dependent functionality
import librosa # to extract speech features
import wave # read and write WAV files
import matplotlib.pyplot as plt # to generate the visualizations
import numpy as np

# MLP Classifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

# LSTM Classifier
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import rmsprop


def extract_mfcc(wav_file_name):
    # This function extracts mfcc features and obtain the mean of each dimension
    # Input : path_to_wav_filehttps://jobseekers.learnsoftechs.com/?p=1437
    # Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

model_A = load_model('C:/Users/Dell/Downloads/MaheshMajor/Modeled_LSTM.h5')
path_ = 'C:/Users/Dell/Downloads/MaheshMajor/my_Audio_file.wav'
ipd.Audio(path_)
a = extract_mfcc(path_)
a.shape
a1 = np.asarray(a)
a1.shape
q = np.expand_dims(a1,-1)
qq = np.expand_dims(q,0)
qq.shape
pred = model_A.predict(qq)
pred
preds=pred.argmax(axis=1)
preds
emotions=['Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised']
print(emotions[preds[0]])
