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


# WAV_files_path_training = 'NELOW_V5/Testing_dog/WAV_files/'
# Numpy_files_path_training = 'NELOW_V5/Testing_잡음/Numpy_files/'
# lis = os.listdir(Numpy_files_path_training)
#
# print(len(lis))
#
# label = []
#
# for i in lis:
#     if i[-9] == 'L':
#         label.append(0)
#     elif i[-9] == 'M':
#         label.append(1)
#     else:
#         label.append(2)
#
# print(label)
# print('---------------------------------------------------------------------')
# print(f'Leak count : {label.count(0)}')
# print(f'Meter count : {label.count(1)}')
# print(f'NO_Leak count : {label.count(2)}')
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

def get_spec(path):
    q, w = librosa.load(path=path, sr=None)
    q=librosa.resample(q,orig_sr=w,target_sr=8000) # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
    w=8000
    q, w = get_wav_clean1sec(q, w)
    q, w = get_wav_filtered(q, w)
    map=librosa.feature.mfcc(y=q,sr=w,n_fft=2048, hop_length=512,n_mfcc = 20)
    # y=q : 오디오 신호를 입력받음 / sr=w : 샘플링 레이트 / n_fft: FFT (Fast Fourier Transform)길이 지정 => 2048개의 샘플을 사용
    # hop_length: 프레임 간의 hop length (시간 간격)을 설정 / n_mfcc :  20개의 MFCC 계수를 계산하겠다는 의미
    # map은 2D numpy 배열로, 각 열은 각 시간 프레임의 MFCC값
    return map

def save_npy(i_path,o_path):
    lis = os.listdir(i_path)
    for i in lis:
        if '.wav' in i:
            q=get_spec(i_path+i)
            q=q[:,:-3] # 마지막 3개의 열만 제거 (오디오 길이가 일정하지 않거나, 마지막 프레임이 불완전할 가능성이 높기 때문에 제거하는 것)
            np.save(o_path+i+'.npy',q)
    return


path = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/no_bandfilter/185349_20241031_13_58_42_126_L.wav'


q, w = librosa.load(path=path, sr=None)
q_1, w_1 = get_wav_clean1sec(q, w)
q_f , w_f = get_wav_filtered(q_1, w_1)
map=librosa.feature.mfcc(y=q_f,sr=w_f,n_fft=2048, hop_length=512,n_mfcc = 20)
get__spec=map[:,:-3]
npy_table = get__spec.reshape(-1, 20, 13, 1)
npy_table_real = np.array(npy_table)

print(q)
print(w)
print(len(q))
print(q.shape)
# print('--------------------------------')
# print(q_1)
# print(w_1)
# print(len(q_1))
# print(q_1.shape)
# print('------------------------------')
# print(q_f)
# print(len(q_f))
# print('-------------------------------')
# print('MFCC')
# print(map)
# print(len(map))
# print(map.shape)
# print('-----------------------------------')
# print('get_spec')
# print(get__spec)
# print(len(get__spec))
# print(get__spec.shape)
# print('-----------------------------------')
# print('npy_table')
# print(npy_table)
# print(len(npy_table))
# print(npy_table.shape)
# print('-----------------------------------')
# print('npy_table real')
# print(npy_table_real)
# print(len(npy_table_real))
# print(npy_table_real.shape)


AI_model_testing = 'NELOW_AI_model/NELOW_GL_model_V3.h5'
AI_model_testing_edge = 'NELOW_AI_model/NELOW_GL_TFLite_model_V3.tflite'

AI_model = load_model(AI_model_testing, compile=False)

# npy_table, label, ee = load_npy(Numpy_files_path_testing)
# Getting predictions
# AI_model_predictions = AI_model.predict(npy_table_real)
# print(AI_model_predictions)
# print(np.argmax(AI_model_predictions, axis=1))

# def predict_with_tflite(interpreter, input_data):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#
#     interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     return output_data

# TFLite 모델 로드

input_data = npy_table_real.astype(np.float32)  # float32로 형 변환만 수행

interpreter = tf.lite.Interpreter(model_path=AI_model_testing_edge)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])


print(output_data)
