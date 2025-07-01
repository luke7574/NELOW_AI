import numpy as np
import librosa
import sounddevice as sd
from scipy.signal import butter, lfilter, filtfilt
from scipy.io.wavfile import write
import os
import pandas as pd
from scipy.fftpack import fft
import matplotlib.pyplot as plt

save_filter_wav = 0
save_fft_data_csv = 0
save_fft_chart = 0
amp3 = 1

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

    return NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency, data, samplerate

def get_NELOW_values_lfilter(wav_path):
    s_fft = []
    data, samplerate = librosa.load(wav_path, sr=None, duration=5)
    q_lfilter, _ = get_wav_filtered(data, samplerate)
    SEC_0_1 = int(samplerate / 10)
    SEC_1 = samplerate

    time_l = int(len(q_lfilter) / SEC_1)
    i_time = (time_l - 1) * 10 - 1

    for i in range(i_time):
        u_data = abs(q_lfilter[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1])
        s_fft.append(np.std(u_data))

    a = np.argmin(s_fft) + 1 # 표준편차가 가장 작은 1초 구간 선택

    tfa_data = abs(fft(q_lfilter[a * SEC_0_1:a * SEC_0_1 + SEC_1])) # 시간 도메인 데이터(tfa_data)를 주파수 도메인으로 변환

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

    return NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency, q_lfilter

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

# 주파수 차트를 하나의 이미지로 저장
def save_combined_frequency_plot(original, lfilter, filtfilt, save_path):
    plt.figure(figsize=(18, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.plot(original, label='Original', color='blue')
    plt.title('FFT - Original')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()

    # lfilter
    plt.subplot(1, 3, 2)
    plt.plot(lfilter, label='lfilter', color='green')
    plt.title('FFT - lfilter')
    plt.xlabel('Frequency Bin')
    plt.grid(True)
    plt.tight_layout()

    # filtfilt
    plt.subplot(1, 3, 3)
    plt.plot(filtfilt, label='filtfilt', color='red')
    plt.title('FFT - filtfilt')
    plt.xlabel('Frequency Bin')
    plt.grid(True)
    plt.tight_layout()

    # 전체 이미지 저장
    plt.savefig(save_path)
    plt.close()


# 저장할 폴더 지정
save_folder = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/clear_bandfilter/"
path = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/원본파일/강도3/182307_L.wav'
#오리지널
NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency, q, w = get_NELOW_values(path)
# 필터 적용
NELOW_fft_data_lfilter, std_deviation_lfilter, wave_energy_lfilter, wave_max_frequency_lfilter, q_lfilter = get_NELOW_values_lfilter(path)
NELOW_fft_data_filtfilt, std_deviation_filtfilt, wave_energy_filtfilt, wave_max_frequency_filtfilt, q_filtfilt = get_NELOW_values_filtfilt(path)

if save_filter_wav:
    # 재생할 수 있도록 int16로 변환
    q_lfilter_int16 = np.int16(q_lfilter / np.max(np.abs(q_lfilter)) * 32767)
    q_filtfilt_int16 = np.int16(q_filtfilt / np.max(np.abs(q_filtfilt)) * 32767)
    # 파일 경로 지정
    file_lfilter_path = os.path.join(save_folder, "filtered_lfilter_182307.wav")
    file_filtfilt_path = os.path.join(save_folder, "filtered_filtfilt_182307.wav")
    # 파일 저장
    write(file_lfilter_path, w, q_lfilter_int16)
    write(file_filtfilt_path, w, q_filtfilt_int16)
    print('음성 필터 적용 파일 저장완료')



if save_fft_data_csv:
    # 필터 적용된 신호 간 비교 (대칭성 확인)
    # 두 필터링된 신호를 동일한 길이로 자르기 (만일 길이가 다를 경우 대비)
    original = np.array(NELOW_fft_data)
    q_lfilter_trim = np.array(NELOW_fft_data_lfilter)
    q_filtfilt_trim = np.array(NELOW_fft_data_filtfilt)
    # 차이 계산
    diff_lfilter = original - q_lfilter_trim
    diff_filtfilt = original - q_filtfilt_trim
    # 비교 결과 DataFrame 생성
    df_compare = pd.DataFrame({
        'HZ': np.arange(len(original)),
        'original': original,
        'lfilter': q_lfilter_trim,
        'filtfilt': q_filtfilt_trim,
        'diff_lfilter': diff_lfilter,
        'diff_filtfilt': diff_filtfilt
    })
    # CSV 파일 저장
    compare_csv_path = os.path.join(save_folder, "filter_comparison_182307.csv")
    df_compare.to_csv(compare_csv_path, index=False)
    print(f"비교 결과가 CSV로 저장되었습니다: {compare_csv_path}")

if save_fft_chart:
    # 저장 경로 지정
    combined_plot_path = os.path.join(save_folder, 'fft_comparison_all_182307.png')

    # 이미지 저장 실행
    save_combined_frequency_plot(
        NELOW_fft_data,
        NELOW_fft_data_lfilter,
        NELOW_fft_data_filtfilt,
        combined_plot_path
    )

    print(f"주파수 비교 이미지가 저장되었습니다: {combined_plot_path}")


path_BPF = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/BPF/강도3/filtered_filtfilt_182307.wav"
save_folder_1 = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/대표님_BPF/Amplitude_X_3/강도3/"
if amp3:
    data, samplerate = librosa.load(path_BPF, sr=None, duration=5)
    data1 = data * 3
    # 클리핑을 허용하되, -32768 ~ 32767 범위를 초과하면 잘라냄
    data1_clipped = np.clip(data1, -1.0, 1.0)
    data1_int16 = np.int16(data1_clipped * 32767)
    file_data1_path = os.path.join(save_folder_1, "Amplitude_X_3_182307.wav")
    write(file_data1_path, samplerate, data1_int16)


