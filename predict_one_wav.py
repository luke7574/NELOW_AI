import numpy as np
import pandas as pd
import librosa
import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalMaxPooling2D, LeakyReLU
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.signal import butter, lfilter, filtfilt
import os

from keras import backend as K

from scipy.fftpack import fft

AI_model_testing = 'MEL_Spectrogram/AI_MODEL/NELOW_MEL_model_V6.h5'
path = "C:/Users/user/중부발전/M2_Leak/0620_0703/V111/4663_20250622_030000/테스트/remove_elec/adjusted_output.wav"
model_testing = 1

def get_wav_clean1sec(signal,sr):
    SEC_0_1 = sr // 10  # 0.1초 샘플 개수
    SEC_1 = sr          # 1초 샘플 개수
    duration = int(len(signal) / sr)  # 오디오의 총 길이 (초단위)
    s_fft = []
    i_time = (duration - 1) * 10 - 1  # 검사할 1초 구간의 개수
    for i in range(i_time):
        u_data = signal[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1] # 100ms 간격으로 이동하며 1초 길이의 신호 추출
        s_fft.append(np.std(u_data))
    a = np.argmin(s_fft) + 1
    tfa_data = signal[a * SEC_0_1: a * SEC_0_1 + SEC_1]
    return tfa_data, sr

def get_wav_filtered_filt(signal,sr):
    minFreq=50; maxFreq=2000; order=5

    nyq = 0.5 * sr
    low = minFreq / nyq
    high = maxFreq / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered = filtfilt(b, a, signal)

    return filtered, sr

def get_spec(path):
    data, sr = librosa.load(path=path, sr=None)
    data = librosa.resample(data,orig_sr=sr,target_sr=8000) # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
    sr = 8000
    # data, sr = get_wav_clean1sec(data, sr)
    data, sr = get_wav_filtered_filt(data, sr)

    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # 로그 변환 적용

    return mel_spec_db

def load_npy(path):
    npy_table = []
    name_only = os.path.splitext(path)[0]
    print(name_only)
    fft_data = get_spec(path)
    fft_data = fft_data.reshape(-1, 128, 16, 1) # 2D CNN 입력 형태로 변환
    npy_table.append(fft_data)
    npy_table = np.array(npy_table) # input
    npy_table = npy_table.reshape(-1, 128, 16, 1)
    return npy_table

# Evaluate the model if model_testing flag is true
if model_testing:
    AI_model = load_model(AI_model_testing, compile=False)
    npy_table = load_npy(path)
     # Getting predictions
    AI_model_predictions = AI_model.predict(npy_table)
    # print(AI_model_predictions)
    AI_model_predictions_max = np.array(np.argmax(AI_model_predictions, axis=1))
    # print(AI_model_predictions_max)
    if 0 in AI_model_predictions_max:
        print("누수")
    elif 1 in AI_model_predictions_max:
        print("미터")
    else:
        print("비누수")
