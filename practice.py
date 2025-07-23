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

save_filter_wav = 0
amp3 = 0
save_filter_wav_float = 0
def get_wav_filtered_filt(signal,sr):
    minFreq=50; maxFreq=2000; order=5
    nyq = 0.5 * sr
    low = minFreq / nyq  # 0.0125
    high = maxFreq / nyq  # 0.5
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered, sr

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


def get_NELOW_values_filtfilt(wav_path):
    s_fft = []
    data, samplerate = librosa.load(wav_path, sr=None, duration=5)
    q_filtfilt, _ = get_wav_filtered_filt(data, samplerate)
    SEC_0_1 = int(samplerate / 10)
    SEC_1 = samplerate

    time_l = int(len(q_filtfilt) / SEC_1)
    i_time = (time_l - 1) * 10 - 1

    for i in range(i_time):
        u_data = abs(q_filtfilt[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1])
        s_fft.append(np.std(u_data))

    a = np.argmin(s_fft) + 1 # 표준편차가 가장 작은 1초 구간 선택

    tfa_data = abs(fft(q_filtfilt[a * SEC_0_1:a * SEC_0_1 + SEC_1])) # 시간 도메인 데이터(tfa_data)를 주파수 도메인으로 변환

    tfa_data3000 = tfa_data[0:4000]

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

    return NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency, q_filtfilt

original3 = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도3_3/182307_L.wav'
filtfilt_wav3 = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도3_3/filtered_filtfilt_182307_float.wav'
ampX3_3 = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도3_1/Amplitude_X_3_179450.wav'
ampX3_3_noclip = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도3_3/Amplitude_X_3_182307_float.wav'

original20 = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도20_3/185678_L.wav'
filtfilt_wav20 = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도20_3/filtered_filtfilt_185678_float.wav'
ampX3_20 = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도20_1/Amplitude_X_3_182602.wav'
ampX3_20_noclip = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도20_3/Amplitude_X_3_185678_float.wav'

#오리지널
NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency, q, w = get_NELOW_values(original3)

# 필터 적용
(NELOW_fft_data_filtfilt, std_deviation_filtfilt, wave_energy_filtfilt,
 wave_max_frequency_filtfilt, q_filtfilt) = get_NELOW_values_filtfilt(original20)

save_folder = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/강도20_3/'

# if save_filter_wav:
#     # 재생할 수 있도록 int16로 변환
#     q_filtfilt_int16 = np.int16(q_filtfilt / np.max(np.abs(q_filtfilt)) * 32767)
#     # 파일 경로 지정
#     file_filtfilt_path = os.path.join(save_folder, "filtered_filtfilt_182307.wav")
#     # 파일 저장
#     write(file_filtfilt_path, w, q_filtfilt_int16)
#     print('음성 필터 적용 파일 저장완료')


if save_filter_wav_float:
    # float32로 바로 저장 (스케일링 없이)
    q_filtfilt_float32 = q_filtfilt.astype(np.float32)
    # 파일 경로 지정
    file_filtfilt_path = os.path.join(save_folder, "filtered_filtfilt_185678_float.wav")
    # 파일 저장 (float32 형식으로)
    write(file_filtfilt_path, w, q_filtfilt_float32)
    print('음성 필터 적용 파일 저장완료 (float32 포맷)')

# print(q)
# print(q_filtfilt)
# a = np.max(np.abs(q_filtfilt))
# print(np.argmax(np.abs(NELOW_fft_data_filtfilt)))

save_folder_1 = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/TEST/"

if amp3:
    data, samplerate = librosa.load(file_filtfilt_path, sr=None, duration=5)
    data1 = data * 3
    # 클리핑을 허용하되, -32768 ~ 32767 범위를 초과하면 잘라냄
    # data1_clipped = np.clip(data1, -1.0, 1.0)
    q_filtfilt_float32 = data1.astype(np.float32)
    file_data1_path = os.path.join(save_folder, "Amplitude_X_3_185678_float.wav")
    write(file_data1_path, samplerate, q_filtfilt_float32)

def visualize_wav(path, title_prefix, filename):
    # 1. wav 파일 로드
    y, sr = librosa.load(path, sr=None, duration=5)

    # 2. 시간 벡터
    t = np.linspace(0, len(y) / sr, num=len(y))

    # 3. FFT 계산
    N = len(y)
    yf = np.abs(fft(y))[:N // 2]
    xf = np.linspace(0, sr / 2, N // 2)

    # 4. 시각화
    plt.figure(figsize=(18, 8))

    # (1) 시간 도메인
    plt.subplot(3, 1, 1)
    plt.plot(t, y)
    plt.title(f"{title_prefix} - Time Domain (Waveform)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # (2) 주파수 도메인 (FFT)
    plt.subplot(3, 1, 2)
    plt.plot(xf, yf)
    plt.title(f"{title_prefix} - Frequency Domain (FFT)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)

    # (3) 스펙트로그램
    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title_prefix} - Spectrogram")

    plt.tight_layout()
    plt.savefig(f"C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/TEST/{filename}.png")
    plt.close()


visualize_wav(filtfilt_wav3, "Filtered (filtfilt)", 'filt20_182307_float')
# visualize_wav(ampX3_20, "Amplitude x3 (use Clipping)", 'amp20_clipO_182602')
visualize_wav(ampX3_3_noclip, "Amplitude x3 (Clipping X)", 'amp20_clipX_182307_float')


data = pd.read_csv('C:/Users/user/AI/NELOW/NELOW_AI/Training_2024_문욱_강도2/NELOW_energy_comparison.csv')

a = data['강도값']
b = data['NELOW 웹 강도값']
c = data['filename']
for i in range(len(data)):
    if a[i] != b[i]:
        print(c[i])