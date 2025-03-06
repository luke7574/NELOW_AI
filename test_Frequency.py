import numpy as np
import pandas as pd
import librosa
import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalMaxPooling2D, LeakyReLU
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.signal import butter, lfilter, filtfilt
import os

from keras import backend as K

from scipy.fftpack import fft

AI_model_training = 'Frequency/AI_MODEL/NELOW_FREQ_model_1.h5'
AI_model_testing = 'Frequency/AI_MODEL/NELOW_FREQ_model_1.h5'

WAV_files_path_training = 'Frequency/test_WAV_FILES/'
Numpy_files_path_training = 'Frequency/test_NUMPY_FILES/'

WAV_files_path_testing = 'Testing_Aramoon/WAV_files/'
Numpy_files_path_testing = 'Testing_Aramoon/Numpy_files/'
CSV_files_path_testing = 'Testing_Aramoon/CSV_files/'

training_sound_preprocessing = 0   # 음성파일(wav) numpy배열로 변환하여 저장
model_training = 0

testing_sound_preprocessing = 0    # 음성파일(wav) numpy배열로 변환하여 저장
model_testing = 0


# Define recall metric
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Define precision metric
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Define F1 score metric
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Select clean 1-second segment from the signal
# 음성신호에서 1초 길이의 깨끗한 구간 추출
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

# Apply bandpass filter to the signal
# 20Hz에서 1000Hz까지의 대역통과필터(band) 적용
def get_wav_filtered(signal,sr):
    minFreq=20; maxFreq=1000; order=5

    nyq = 0.5 * sr
    low = minFreq / nyq
    high = maxFreq / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered = lfilter(b, a, signal)

    return filtered, sr

# 소리 최대 진폭(누수강도) , 소리 최대 주파수 구하기
def get_NELOW_values(wav_path):
    s_fft = []
    data, samplerate = librosa.load(wav_path, sr=None, duration=5)

    SEC_0_1 = int(samplerate / 10)
    SEC_1 = samplerate

    time_l = int(len(data) / SEC_1)
    i_time = (time_l - 1) * 10 - 1

    for i in range(i_time):
        u_data = abs(data[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1])
        s_fft.append(np.std(u_data))
    a = np.argmin(s_fft) + 1 # 표준편차가 가장 작은 1초 구간 선택

    tfa_data = abs(fft(data[a * SEC_0_1:a * SEC_0_1 + SEC_1])) # 시간 도메인 데이터(tfa_data)를 주파수 도메인으로 변환
    tfa_data3000 = tfa_data[0:3000]
    tfa_data3000[:50] = 0

    idx = np.argmax(tfa_data3000) # 가장 큰 주파수 성분의 인덱스(주파수 위치)

    startPos = 0

    if idx < 10:
        startPos = 0
    else:
        startPos = idx - 10
    stopPos = idx + 10

    # 누수강도
    wave_energy = np.average(tfa_data3000[startPos:stopPos])   # 가장 강한 주파수 대역의 평균 진폭 (누수 강도)
    # 최대주파수
    wave_max_frequency = np.argmax(tfa_data3000)               # 최대 주파수 성분의 위치(인덱스값) (주파수 Hz 단위)

    return wave_energy, wave_max_frequency


# 1초구간 찾고 -> 대역통과필터 거치고 -> fft 수행 -> input_data 완성
def get_spec(path):
    data, sr = librosa.load(path=path, sr=None)
    data = librosa.resample(data,orig_sr=sr,target_sr=8000) # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
    sr = 8000
    data, sr = get_wav_clean1sec(data, sr)
    data, sr = get_wav_filtered(data, sr)

    tfa_data = abs(fft(data))  # 시간 도메인 데이터(tfa_data)를 주파수 도메인으로 변환

    return tfa_data



# Save numpy array representations of spectrograms
# 디렉토리의 모든 WAV 파일의 MFCC 데이터를 NumPy 배열로 저장하여 추가 처리를 위해 사용함.
def save_npy(i_path,o_path):
    lis = os.listdir(i_path)
    for i in lis:
        if '.wav' in i:
            q=get_spec(i_path+i)
            np.save(o_path+i+'.npy',q)
    return

# Prepare data if sound_preprocessing flag is true
# 음성파일(wav) numpy배열로 변환하여 저장
if training_sound_preprocessing:
    save_npy(WAV_files_path_training, Numpy_files_path_training)



# lis = os.listdir(WAV_files_path_training)
# for i in lis:
#     if '.wav' in i:
#         path = WAV_files_path_training+i
#         # print(path)
#         data, sr = librosa.load(path=path, sr=None)
#         # print(f'data의 길이 ==== {len(data)}')
#         # print(sr)
#         data = librosa.resample(data, orig_sr=sr, target_sr=8000)  # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
#         sr = 8000
#         # print('=========================')
#         # print(f'data의 길이 ==== {len(data)}')
#         # print(sr)
#         data, sr = get_wav_clean1sec(data, sr)
#         data, sr = get_wav_filtered(data, sr)
#         tfa_data = abs(fft(data))  # 시간 도메인 데이터(tfa_data)를 주파수 도메인으로 변환
#         # print(tfa_data)
#         # print(len(tfa_data))
#         print('=====================================')
#         tfa_data3000 = tfa_data[0:3000]
#         tfa_data3000[:50] = 0
#         print(tfa_data3000)
#         # print(len(tfa_data3000))
#         print('=====================================')
#         NELOW_fft_data = tfa_data3000.tolist()
#         print(NELOW_fft_data)
#         # print(len(NELOW_fft_data))
#         print('=====================================')
#
#
npy_table = []
label = []
filename = []
lis = os.listdir(Numpy_files_path_training)

for i in lis:
    if '.npy' in i:
        fft_data = np.load(Numpy_files_path_training + i)
        # print(fft_data)    # [0.         0.         0.         ... 0.00143746 0.00144381 0.00146207]
        fft_data = fft_data.reshape(-1, 3000, 1) # 1D CNN 입력 형태로 변환
        # print('===========================')
        # print(fft_data)             #일렬로 값 나열
        npy_table.append(fft_data)
        if i[-9] == 'L':
            label.append(0)
        elif i[-9] == 'M':
             label.append(1)
        else:
            label.append(2)
        filename.append(i)


label = tf.keras.utils.to_categorical(label, num_classes=3)
npy_table = np.array(npy_table) # input
label = np.array(label)         # output
# print(label)
filename = np.array(filename)
# print(filename)
npy_table = npy_table.reshape(-1, 3000, 1)

label = label.reshape(-1, 3)

filename = filename.reshape(-1, 1)













