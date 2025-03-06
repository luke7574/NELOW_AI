import numpy as np
import librosa
import os
import sys
import subprocess  # For a cross-platform solution
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv
import shutil
import keras
import tensorflow
from tensorflow.keras.models import load_model
from tqdm import tqdm  # tqdm 추가
from scipy.signal import butter, lfilter
from keras import backend as K
from tensorflow.keras.utils import custom_object_scope
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec  # Add gridspec for custom layout
from scipy.fftpack import fft

# WAV_files 폴더 안에 있는 모든 파일들을 leak / meter / no_leak 폴더 안으로 이동
train_move_wav = 0
val_move_wav = 0


# 주파수, 음차트 분류 작업
train_data_wav = 0
val_data_wav = 0
predict_data_wav = 0

test_move_dataset_wav = 1

def move_wav_files(wav_dir):
    # 📌 WAV 파일 목록 가져오기
    wav_files123 = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    max_amplitude = find_global_max_y(wav_files123)
    # 📌 이동할 폴더 경로 설정
    leak_dir = os.path.join(wav_dir, "leak")
    meter_dir = os.path.join(wav_dir, "meter")
    no_leak_dir = os.path.join(wav_dir, "no_leak")
    # 📌 폴더가 없으면 자동 생성
    # os.makedirs(leak_dir, exist_ok=True)
    # os.makedirs(meter_dir, exist_ok=True)
    # os.makedirs(no_leak_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    # 📌 파일 이동 실행
    for file_name in wav_files:
        file_path = os.path.join(wav_dir, file_name)
    #
        # 📌 L, M, N 여부 확인
        if "_L.wav" in file_name:  # 누수
            dest_path = os.path.join(leak_dir, file_name)
            shutil.move(file_path, dest_path)
            print(f"✅ {file_name} → leak 폴더 이동 완료")

        elif "_M.wav" in file_name:  # 수도미터
            dest_path = os.path.join(meter_dir, file_name)
            shutil.move(file_path, dest_path)
            print(f"✅ {file_name} → meter 폴더 이동 완료")

        elif "_N.wav" in file_name:  # 비누수
            dest_path = os.path.join(no_leak_dir, file_name)
            shutil.move(file_path, dest_path)
            print(f"✅ {file_name} → no_leak 폴더 이동 완료")

    print("🎯 모든 파일 이동 완료!")

    return max_amplitude

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
    # NELOW_fft_data -> FFT 변환된 주파수 스펙트럼 데이터 (리스트) / std_deviation -> 전체 주파수 분포의 표준편차 (진폭 변화량) / wave_energy -> 특정 주파수 범위 내 평균 에너지 (누수 강도) / wave_max_frequency -> 가장 강한 주파수 (누수 주파수)

def find_global_max_y(wav_files):
    """
    데이터셋 내 모든 주파수 데이터의 최대값을 찾아서 Y축을 통일
    """
    global_max_y = 0
    for wav_file in tqdm(wav_files, desc="🔍 전체 Y축 기준값 찾는 중", unit="file"):
        try:
            fft_data, _, _, _ = get_NELOW_values(wav_file)
            max_value = max(fft_data)
            global_max_y = max(global_max_y, max_value)  # 전체 최대값 갱신
        except Exception as e:
            print(f"❌ 오류 발생 (Y축 계산 중): {wav_file} - {e}")
    return global_max_y


def save_graphs(file_path, output_folder, global_max_y):
    """
    WAV 파일의 사운드 차트(Waveform) 및 주파수 차트(Spectrum)를 클래스별 폴더에 저장
    :param file_path: .wav 파일 경로
    :param output_folder: train_data 경로 (train_data/leak, train_data/meter, train_data/no_leak)
    """
    # 파일명 및 클래스 폴더 경로 설정
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    class_folder = os.path.basename(os.path.dirname(file_path))  # 폴더명(leak, meter, no_leak)

    # 저장할 클래스별 폴더 경로 설정
    save_dir = os.path.join(output_folder, class_folder)
    # os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성

    # 오디오 로드
    raw_y, raw_sr = librosa.load(file_path, sr=None)

    # ✅ 깨끗한 1초 구간 찾기
    clean_1sec, _ = get_wav_clean1sec(raw_y, raw_sr)

    # NELOW 스펙트럼 데이터 추출
    NELOW_fft_data, _, _, wave_max_frequency = get_NELOW_values(file_path)
    NELOW_fft_data = np.array(NELOW_fft_data)


    # # ✅ 선택된 1초 구간 가져오기
    # SEC_0_1 = int(raw_sr / 10)
    # SEC_1 = raw_sr
    # # ✅ start_sample을 안전하게 조정
    # start_sample = wave_max_frequency * SEC_0_1  # 기본 설정
    # start_sample = min(start_sample, max(0, len(raw_y) - SEC_1))  # 음수 방지 및 최대 길이 초과 방지
    #
    # # ✅ end_sample을 안전하게 조정
    # end_sample = min(len(raw_y), start_sample + SEC_1)  # 1초 길이



    # ✅ 저장할 경로 설정
    waveform_path = os.path.join(save_dir, f"{file_name}_waveform.png")
    spectrum_path = os.path.join(save_dir, f"{file_name}_spectrum.png")

    # # ✅ 빈 배열 방지
    # if end_sample <= start_sample or len(raw_y[start_sample:end_sample]) == 0:
    #     print(f"⚠️ 경고: {file_path}에서 유효한 1초 구간을 찾을 수 없음 (start_sample={start_sample}, end_sample={end_sample})")
    # else:
    #     # ✅ 1️⃣ 파형(Waveform) 그래프 생성 및 저장
    #     fig_wave, ax1 = plt.subplots(figsize=(10, 3))
    #     librosa.display.waveshow(raw_y[start_sample:end_sample], sr=raw_sr, ax=ax1)  # 선택된 1초만 표시
    #     ax1.set_ylim(-1, 1)
    #     fig_wave.savefig(waveform_path, bbox_inches='tight', dpi=300)
    #     plt.close(fig_wave)  # 메모리 해제

    # ✅ 1️⃣ "깨끗한 1초"를 사용하여 파형(Waveform) 그래프 저장
    fig_wave, ax1 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(clean_1sec, sr=raw_sr, ax=ax1)
    ax1.set_ylim(-1, 1)
    fig_wave.savefig(waveform_path, bbox_inches='tight', dpi=300)
    plt.close(fig_wave)  # 메모리 해제


    # ✅ 2️⃣ NELOW 스펙트럼 그래프 생성 및 저장
    fig_spec, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(NELOW_fft_data, color='purple')
    ax3.set_ylim(0, global_max_y * 1.3)  # 통일된 Y축 설정
    # 그래프 저장
    fig_spec.savefig(spectrum_path, bbox_inches='tight', dpi=300)
    plt.close(fig_spec)  # 메모리 해제

    print(f"✅ 저장 완료: {waveform_path}, {spectrum_path}")

# # 📌 WAV_files 폴더 안에 있는 모든 파일들을 leak / meter / no_leak 폴더 안으로 이동
# if train_move_wav:
#     wav_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/train_WAV_files"
#     move_wav_files(wav_dir)
#
# if val_move_wav:
#     wav_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/val_WAV_files"
#     move_wav_files(wav_dir)
#
#
# # 주파수차트 / 음차트 ===> 클래스 별로 저장
# if train_data_wav:
#     # 📌 WAV 파일이 있는 디렉토리에서 그래프 저장 실행
#     wav_root_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/train_WAV_files"  # WAV 파일이 포함된 폴더 (leak, meter, no_leak 포함)
#     train_data_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/train_data"  # train_data 폴더
#
#     # 📌 폴더 내 모든 .wav 파일에 대해 그래프 생성 및 저장 실행
#     for class_name in ["leak", "meter", "no_leak"]:
#         class_path = os.path.join(wav_root_dir, class_name)
#         if os.path.exists(class_path):  # 폴더가 존재하는 경우만 실행
#             wav_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.wav')]
#             max_amplitude = find_global_max_y(wav_files)
#             for wav_file in wav_files:
#                 save_graphs(wav_file, train_data_dir, max_amplitude)
#
# if val_data_wav:
#     # 📌 WAV 파일이 있는 디렉토리에서 그래프 저장 실행
#     wav_root_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/val_WAV_files"  # WAV 파일이 포함된 폴더 (leak, meter, no_leak 포함)
#     val_data_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/val_data"  # train_data 폴더
#
#     # 📌 폴더 내 모든 .wav 파일에 대해 그래프 생성 및 저장 실행
#     for class_name in ["leak", "meter", "no_leak"]:
#         class_path = os.path.join(wav_root_dir, class_name)
#         if os.path.exists(class_path):  # 폴더가 존재하는 경우만 실행
#             wav_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.wav')]
#
#             for wav_file in wav_files:
#                 save_graphs(wav_file, val_data_dir)
#
# if predict_data_wav:
#     # 📌 WAV 파일이 있는 디렉토리에서 그래프 저장 실행
#     wav_root_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/predict_WAV_files"  # WAV 파일이 포함된 폴더 (leak, meter, no_leak 포함)
#     predict_data_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/test/predict_data"  # train_data 폴더
#
#     # 📌 폴더 내 모든 .wav 파일 가져오기
#     wav_files = [os.path.join(wav_root_dir, f) for f in os.listdir(wav_root_dir) if f.endswith('.wav')]
#
#     for wav_file in wav_files:
#         save_graphs(wav_file, predict_data_dir)


if test_move_dataset_wav:
    wav_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/test_wav"
    max_amplitude = move_wav_files(wav_dir)
    print(max_amplitude)
    print(max_amplitude * 1.3)
    # 📌 WAV 파일이 있는 디렉토리에서 그래프 저장 실행
    wav_root_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/test_wav"  # WAV 파일이 포함된 폴더 (leak, meter, no_leak 포함)
    test_data_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/test_data"

    # 📌 폴더 내 모든 .wav 파일에 대해 그래프 생성 및 저장 실행
    for class_name in ["leak", "meter", "no_leak"]:
        class_path = os.path.join(wav_root_dir, class_name)
        if os.path.exists(class_path):  # 폴더가 존재하는 경우만 실행
            wav_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.wav')]
            for wav_file in wav_files:
                save_graphs(wav_file, test_data_dir, max_amplitude)

