import numpy as np
import librosa
import sounddevice as sd
from scipy.signal import butter, lfilter, filtfilt
from scipy.io.wavfile import write
import os
import pandas as pd
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import librosa.display
import csv

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
    # tfa_data3000[:50] = 0

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

    return NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency, data, samplerate

def get_NELOW_values_dd(wav_path):
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
    # tfa_data3000[:50] = 0

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

    return NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency, data, samplerate



# 저장할 CSV 경로
output_csv = 'C:/Users/user/AI/NELOW/NELOW_AI/Training_2024_문욱_강도2/NELOW_energy_comparison.csv'

wav_file = 'C:/Users/user/AI/NELOW/NELOW_AI/Training_2024_문욱_강도2/WAV_files/'
lis = os.listdir(wav_file)
# print(lis)
# 결과 저장
results = []

for i in lis:
    wav_path = wav_file+i
    try:
        # 각각의 함수 실행
        _, _, wave_energy_normal, max1, _, _ = get_NELOW_values(wav_path)
        _, _, wave_energy_dd, max2, _, _ = get_NELOW_values_dd(wav_path)

        results.append({
            'filename': os.path.basename(wav_path),
            '강도값': wave_energy_normal,
            'Max_HZ': max1,
            'NELOW 웹 강도값': wave_energy_dd,
            'NELOW 웹 Max_HZ': max2
        })

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")

# CSV로 저장
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"CSV 저장 완료: {output_csv}")


























