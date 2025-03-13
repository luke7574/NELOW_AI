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

# AI_model_training = 'NELOW_AI_model/NELOW_GL_model_SWM.h5'
# AI_model_testing = 'NELOW_AI_model/NELOW_GL_model_V3.h5'
#
# WAV_files_path_training = 'test/test_WAV_FILES/'
# Numpy_files_path_training = 'test/test_NUMPY_FILES/'
#
# WAV_files_path_testing = 'Testing_Aramoon/WAV_files/'
# Numpy_files_path_testing = 'Testing_Aramoon/Numpy_files/'
# CSV_files_path_testing = 'Testing_Aramoon/CSV_files/'
#
# training_sound_preprocessing = 0   # 음성파일(wav) numpy배열로 변환하여 저장
# model_training = 0
#
# testing_sound_preprocessing = 1    # 음성파일(wav) numpy배열로 변환하여 저장
# model_testing = 1

# Select clean 1-second segment from the signal
# 음성신호에서 1초 길이의 깨끗한 구간 추출
def get_wav_clean1sec(signal,sr):
    SEC_0_1 = sr // 10  # 0.1초 샘플 개수
    SEC_1 = sr          # 1초 샘플 개수
    duration = int(len(signal) / sr)  # 오디오의 총 길이 (초단위)  int(40020 / 8000) = 5
    s_fft = []
    i_time = (duration - 1) * 10 - 1  # 검사할 1초 구간의 개수
    for i in range(i_time):
        u_data = signal[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1] # 100ms 간격으로 이동하며 1초 길이의 신호 추출
        s_fft.append(np.std(u_data))
    a = np.argmin(s_fft) + 1
    tfa_data = signal[a * SEC_0_1: a * SEC_0_1 + SEC_1]
    return tfa_data, sr


def get_wav_filtered(signal,sr):
    minFreq=20; maxFreq=1000; order=5

    nyq = 0.5 * sr
    low = minFreq / nyq
    high = maxFreq / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered = lfilter(b, a, signal)

    return filtered, sr

# i_path = WAV_files_path_training
# o_path = Numpy_files_path_training
# lis = os.listdir(i_path)  # 리스트 열로 wav파일명 불러오기

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
    tfa_data3000[:50] = 0  # 50Hz 이하 제거

    idx = np.argmax(tfa_data3000)

    startPos = 0

    if idx < 10:
        startPos = 0
    else:
        startPos = idx - 10

    stopPos = idx + 10


    # json파일
    NELOW_fft_data = tfa_data3000.tolist()
    # 표준편차
    std_deviation = np.std(tfa_data)
    # 누수강도
    wave_energy = np.average(tfa_data3000[startPos:stopPos])
    # 최대주파수
    wave_max_frequency = np.argmax(tfa_data3000)

    return NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency




# for i in lis:
#     q, w = librosa.load(path=i_path+i, sr=None)  # q는 오디오 신호 데이터 (Amplitude 값) / w는 샘플링 레이트 (Sampling Rate, Hz)
#     # print(q)
#     # print(len(q))  # 40020 => 오디오 샘플링 레이트(sr) * 오디오 길이(초)
#     # print(w)       #  8000 =>
#     q=librosa.resample(q,orig_sr=w,target_sr=8000) # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
#     w=8000
#     # print(q)
#     # print(len(q))  # 40020 => 오디오 샘플링 레이트(sr) * 오디오 길이(초)
#     # print(w)
#     # print('====================================================================')
#     # 1초 깨끗한 구간 찾기
#     SEC_0_1 = w // 10  # 0.1초 샘플 개수 = 800
#     SEC_1 = w  # 1초 샘플 개수
#     duration = int(len(q) / w)  # 오디오의 총 길이 (초단위)  int(40020 / 8000) = 5
#     s_fft = []
#     i_time = (duration - 1) * 10 - 1  # 검사할 1초 구간의 개수 =  39
#     for j in range(i_time):
#         u_data = q[(j + 1) * SEC_0_1:(j + 1) * SEC_0_1 + SEC_1]  # 100ms 간격으로 이동하며 1초 길이의 신호 추출 1) 800:8800 2) 1600:9600 => 이런식으로 0.1초 간격으로 이동하며 8000개 샘플 추출
#         s_fft.append(np.std(u_data))  # np.std() = 표준표차
#     # print(s_fft)
#     a = np.argmin(s_fft) + 1          # np. argmin() = 최솟값의 인덱스를 반롼하는 함수  => 가장 일정한 구간, 즉 깨끗한 구간 1초를 구하기 위함
#     tfa_data = q[a * SEC_0_1: a * SEC_0_1 + SEC_1]
#
#     # 20Hz에서 1000Hz까지의 대역통과필터(band) 적용
#     minFreq = 20; maxFreq = 1000; order = 5
#     nyq = 0.5 * w   # 4000HZ
#     low = minFreq / nyq      # 0.005
#     high = maxFreq / nyq     # 0.25
#     b, a = butter(order, [low, high], btype='band')  # 대역통과 필터를 생성
#     filtered = lfilter(b, a, tfa_data)
#
#     # y=q : 오디오 신호를 입력받음 / sr=w : 샘플링 레이트 / n_fft: FFT (Fast Fourier Transform)길이 지정 => 2048개의 샘플을 사용 / hop_length: 프레임 간의 hop length (시간 간격)을 설정 / n_mfcc :  20개의 MFCC 계수를 계산하겠다는 의미
#     map = librosa.feature.mfcc(y=filtered, sr=w, n_fft=2048, hop_length=512, n_mfcc=20)
#     # print(map)
#     # print(len(map)) # (20,16)
#     q = map[:, :-3]  # 마지막 3개의 열만 제거 (오디오 길이가 일정하지 않거나, 마지막 프레임이 불완전할 가능성이 높기 때문에 제거하는 것)  (20,13)
#     np.save(o_path + i + '.npy', q)

npy_path = 'C:/Users/user/AI/NELOW/NELOW_AI/MEL_Spectrogram/test_NUMPY_FILES/138964_20230330_11_13_18_126_M.wav.npy'

npy_table = []
label = []
filename = []

fft_data = np.load(npy_path)
print(fft_data)
print(fft_data[0])

print("-----------------------------")
df = pd.DataFrame(fft_data)
print(df)
# 생략 없이 출력 (컬럼 개수 & 행 개수 제한 해제)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print("----------------------------------------------------------------")
print(df)
# # print("Shape of a array:", a.shape)
# fft_data = fft_data.reshape(-1, 3000, 1) # 1D CNN 입력 형태로 변환
# npy_table.append(fft_data)
# if i[-9] == 'L':
#     label.append(0)
# elif i[-9] == 'M':
#     label.append(1)
# else:
#     label.append(2)
# filename.append(i)
#
# label = tf.keras.utils.to_categorical(label, num_classes=3)
# npy_table = np.array(npy_table) # input
# label = np.array(label)         # output
# filename = np.array(filename)
#
# npy_table = npy_table.reshape(-1, 3000, 1)
# label = label.reshape(-1, 3)
# filename = filename.reshape(-1, 1)




