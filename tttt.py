from tkinter import *
import sounddevice as sd
import soundfile as sf
import IPython.display as ipd
from keras.models import load_model
import librosa
import numpy as np
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
from keras.layers import *

main=Tk()

main.geometry("1920x1080")
main.configure(background="blue")
main.title("speech emotion recognition")

lab=Label(text="Speech Emotion Recognition ",bg="pink",fg="white", padx=30,pady=20,font=("comicsansms",36,"bold"),borderwidth=10)
lab.pack(side=TOP,pady=20)

img=PhotoImage(file="Emotions.png")
imglab=Label(image=img)
imglab.pack(side=RIGHT,padx=60,pady=10)

lab2=Label(text="Please Reccord the voice ",bg="blue",fg="white",font=("comicsansms",25,"bold"),borderwidth=10)
lab2.pack(anchor="sw",padx=80,pady=100)


frame=Frame(main,borderwidth=10,bg="grey",relief=SUNKEN)
frame.pack(side=TOP)



def Voice_rec():
    fs = 48000
    # seconds
    duration = 5
    myrecording = sd.rec(int(duration * fs),samplerate=fs, channels=2)
    sd.wait()
    # Save as FLAC file at correct sampling rate
    return sf.write('my_Audio_file.wav', myrecording, fs)





def extract_mfcc(wav_file_name):
    # This function extracts mfcc features and obtain the mean of each dimension
    # Input : path_to_wav_file
    # Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name, duration=3
                         , offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    return mfccs

def Recognition():
    model_A = load_model('Modeled_2LSTM.h5')
    path_ = 'C:/Users/Dell/Downloads/MaheshMajor/my_Audio_file.wav'
    ipd.Audio(path_)
    a = extract_mfcc(path_)
    a.shape
    a1 = np.asarray(a)
    a1.shape
    q = np.expand_dims(a1, -1)
    qq = np.expand_dims(q, 0)
    qq.shape
    pred = model_A.predict(qq)
    pred
    preds = pred.argmax(axis=1)
    preds
   # emotions = ['Neutral   ', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    emotions=['    Neutral  😐', '    Calm  😌', '    Happy  😃', '    Sad  😥 ', '   Angry  😡 ', '    Fearful  😨', '    Disgust  🤢', '   Surprised  😯']

    Output.insert(END,emotions[int(preds)])


b1=Button(frame,fg="white",text="Record your voice",bg="grey",font=("comicsansms",15,"bold"),command=Voice_rec)
b1.pack(padx=5,pady=5)

b2=Button(frame,fg="white",text="Recognize Emotion",bg="grey",font=("comicsansms",15,"bold"),command=Recognition)
b2.pack(padx=5,pady=5)

Output = Text(frame,fg="Yellow" ,height = 5,font=("comicsansms",22,"bold"),
              width = 25,
              bg = "Black")
Output.pack()

main.mainloop()
