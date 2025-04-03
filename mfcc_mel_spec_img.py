import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
import os
from tqdm import tqdm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def get_wav_clean1sec(signal, sr):
    SEC_0_1 = sr // 10
    SEC_1 = sr
    duration = int(len(signal) / sr)
    s_fft = []
    i_time = (duration - 1) * 10 - 1
    for i in range(i_time):
        u_data = signal[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1]
        s_fft.append(np.std(u_data))
    a = np.argmin(s_fft) + 1
    tfa_data = signal[a * SEC_0_1: a * SEC_0_1 + SEC_1]
    return tfa_data, sr
def get_wav_filtered(signal, sr):
    minFreq = 20
    maxFreq = 1000
    order = 5
    nyq = 0.5 * sr
    low = minFreq / nyq
    high = maxFreq / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = lfilter(b, a, signal)
    return filtered, sr
def get_NELOW_values(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    y = librosa.resample(y, orig_sr=sr, target_sr=8000)
    sr = 8000
    y, sr = get_wav_clean1sec(y, sr)
    y, sr = get_wav_filtered(y, sr)

    tfa_data = abs(fft(y))  # 시간 도메인 데이터(tfa_data)를 주파수 도메인으로 변환

    tfa_data3000 = tfa_data[0:3000]
    tfa_data3000[:50] = 0

    NELOW_fft_data = tfa_data3000.tolist()

    return NELOW_fft_data
def preprocess_wav(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    y = librosa.resample(y, orig_sr=sr, target_sr=8000)
    sr = 8000
    y, sr = get_wav_clean1sec(y, sr)
    y, sr = get_wav_filtered(y, sr)


    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=20)
    mfcc_reshaped = mfcc[:, :-3]
    mfcc_reshaped = mfcc_reshaped.reshape(-1, 20, 13, 1)  # Example reshaping, adjust as needed based on the model input
    return mfcc, mfcc_reshaped, y, sr

# 평균 멜 스펙트로그램 구하기
def compute_avg_melspectrogram(wav_path):
    mel_specs = []
    max_length = 0  # 최대 길이를 저장할 변수
    for i in tqdm(os.listdir(wav_path), desc="Processing WAV files", unit="file"):
        wav_file = os.path.join(wav_path, i)
        y, sr = librosa.load(wav_file, sr=8000)  # 샘플링 레이트 맞추기
        y = librosa.resample(y, orig_sr=sr, target_sr=8000)
        sr = 8000
        y, sr = get_wav_clean1sec(y, sr)
        y, sr = get_wav_filtered(y, sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # 🔍 길이 확인하여 가장 긴 멜 스펙트로그램 길이 저장
        max_length = max(max_length, mel_spec_db.shape[1])
        mel_specs.append(mel_spec_db)

    # 🔹 모든 멜 스펙트로그램을 동일한 길이로 맞추기 (짧은 데이터는 0-padding)
    mel_specs_fixed = [librosa.util.fix_length(m, size=max_length, axis=1) for m in mel_specs]
    # 평균 계산
    avg_mel_spec = np.mean(mel_specs_fixed, axis=0)
    return avg_mel_spec, sr


# 음성 파일 경로 (원하는 파일 경로로 변경)
wav_path = "C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/test_wav/"
# "C:/Users/user/AI/NELOW/NELOW_AI/Training/WAV_files/"
# "C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/leak_sound_wav/"
# "C:/Users/user/AI/NELOW/NELOW_AI/Graph_Image/train_WAV_files/leak/"
train_plot = "C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/test_mel_img/"

for i in tqdm(os.listdir(wav_path), desc="Processing WAV files", unit="file"):
    train_plot = f"C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/test_mel_img/{i}.png"
    mel_spec_plot_path = f"C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/test_mel_img/{i}_mel.png"
    wav_file = os.path.join(wav_path, i)

    # 1. 음성 파일 로드
    y, sr = librosa.load(wav_file, sr=None)  # 원본 샘플링 레이트 유지

    # 2. 전처리 거쳐 MFCC 추출
    processed_mfcc_original, _, processed_y_original, processed_sr_original = preprocess_wav(wav_file)

    # 3. 주파수 값 추출
    NELOW_fft_data = get_NELOW_values(wav_file)
    NELOW_fft_data = np.array(NELOW_fft_data)

    # 4. 멜 스펙트로그램 추출 후 로그 변환
    y = librosa.resample(y, orig_sr=sr, target_sr=8000)
    sr = 8000
    y, sr = get_wav_clean1sec(y, sr)
    y, sr = get_wav_filtered(y, sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # 로그 변환 적용

    # 🔹 4. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ✅ 1️⃣ MFCC 시각화
    ax1 = axes[0, 0]
    img1 = librosa.display.specshow(processed_mfcc_original, sr=sr, x_axis="time", cmap="coolwarm", ax=ax1)
    fig.colorbar(img1, ax=ax1, format="%+2.0f dB")
    ax1.set_title("MFCC")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("MFCC Coefficients")

    # ✅ 2️⃣ 멜 스펙트로그램 시각화
    ax2 = axes[0, 1]
    img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma", ax=ax2)
    fig.colorbar(img2, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Mel Spectrogram")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Frequency (Mel)")

    # ✅ 3️⃣ 음차트 시각화
    ax3 = axes[1, 0]
    librosa.display.waveshow(y, sr=sr, axis="time", ax=ax3)
    ax3.set_title("Waveform")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")

    # ✅ 4️⃣ 주파수 스펙트럼 (FFT) 시각화
    ax4 = axes[1, 1]
    # Plot NELOW Spectrum
    ax4.plot(NELOW_fft_data, color='purple')
    ax4.set_title("Frequency Spectrum")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Amplitude")
    # ax4.set_ylim(0, max(NELOW_fft_data) * 1.3)  # Set the y-axis limit
    ax4.grid()
#
#     # # 🎨 3. 멜 스펙트로그램 시각화 (단독 출력)
#     # plt.figure(figsize=(12, 6))  # 그래프 크기 조정
#     # librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
#     # plt.colorbar(format="%+2.0f dB")  # 컬러바 추가
#     # plt.title(f"Mel Spectrogram - {i}")  # 파일명 표시
#     # plt.xlabel("Time (s)")
#     # plt.ylabel("Frequency (Mel)")
#
    plt.tight_layout()
    plt.savefig(train_plot, dpi=300)
    # plt.show()
#
#     # ✅ 2️⃣ 멜 스펙트로그램 시각화 (해상도 조정)
#     fig_mel = plt.figure(figsize=(12, 6), dpi=1000)  # 멜 스펙트로그램 전용 Figure, dpi=500 적용
#     # 색상 명암 조정
#     norm = mcolors.Normalize(vmin=-50, vmax=20)  # -60 dB 이하를 어둡게, 0 dB를 밝게
#
#     ax_mel = fig_mel.add_subplot(111)  # 새로운 Figure에 서브플롯 추가
#     img_mel = librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="plasma", ax=ax_mel, norm=norm)
#
#     fig_mel.colorbar(img_mel, ax=ax_mel, format="%+2.0f dB")
#     ax_mel.set_title("Mel Spectrogram (High-Resolution)")
#     ax_mel.set_xlabel("Time")
#     ax_mel.set_ylabel("Frequency (Mel)")
#
#     # ✅ 멜 스펙트로그램만 고해상도로 저장
#     plt.savefig(mel_spec_plot_path, dpi=1000)  # 해상도 변경 적용
#     plt.close(fig_mel)  # 메모리 절약을 위해 Figure 닫기


################# 평균 멜 스팩트로그램 시각화 ###########################
# # 🔹 1️⃣ 색상 정의 (기존 magma + 특정 구간 강조)
# 평균 멜 스펙트로그램 구하기
# avg_mel_spec, sr = compute_avg_melspectrogram(wav_path)
# cmap = plt.get_cmap("magma")  # 기존 colormap
# colors = [
#     cmap(0.0), cmap(0.3), cmap(0.6),  # 어두운 영역 유지
#     "red"  # 🔥 -10dB보다 높은 영역 강조
# ]
# bounds = [-50, -30, -20, -10, 10]  # -10dB 이상을 강조하는 경계 설정
# # 🔹 2️⃣ 사용자 지정 컬러맵 생성
# custom_cmap = mcolors.ListedColormap(colors)
# norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
#
#
# # 평균 멜 스팩트로그램 시각화
# plt.figure(figsize=(10, 5))
# librosa.display.specshow(avg_mel_spec, sr=sr, x_axis="time", y_axis="mel", cmap=custom_cmap, norm=norm)
# plt.colorbar(format="%+2.0f dB")
# plt.title("Average Mel Spectrogram of Leak Sounds")
# plt.xlabel("Time")
# plt.ylabel("Frequency (Mel)")
# plt.show()


############## 3차원 그래프 만들기 ########################
# wav_file = "C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/185683_20241106_11_36_07_126_L.wav"
# frame_length_sec = 0.1  # 0.5초마다 잘라서 FFT 수행
# sr_target = 8000
#
# # 1. 오디오 불러오기
# y, sr = librosa.load(wav_file, sr=sr_target)
# y, sr = get_wav_clean1sec(y, sr)
# y, sr = get_wav_filtered(y, sr)
# # 2. 프레임 길이 설정
# frame_length = int(frame_length_sec * sr)
# num_frames = len(y) // frame_length
#
# # 3. 프레임별 FFT 수행
# spectrums = []
# for i in range(num_frames):
#     frame = y[i * frame_length : (i+1) * frame_length]
#     window = np.hanning(len(frame))
#     correction_factor = 1 / (np.sum(window) / len(window))  # 윈도우 보정 계수
#     fft_data = np.abs(np.fft.rfft(frame * window)) * 4 * correction_factor # 스케일링 및 보정
#     spectrums.append(fft_data)
# spectrums = np.array(spectrums)  # (프레임 수, 주파수 길이)
#
# # 4. 3D 플롯 준비
# freqs = np.linspace(0, sr/2, spectrums.shape[1])     # X축: 주파수
# times = np.arange(num_frames) * frame_length_sec     # Y축: 시간(프레임 인덱스)
# X, Y = np.meshgrid(freqs, times)                     # X, Y meshgrid
# Z = spectrums                                        # Z: 진폭 배열 (프레임 x 주파수)
#
# # Plotly용 Surface Plot 생성
# fig = go.Figure(data=[go.Surface(z=spectrums, x=freqs, y=times, colorscale='Plasma')])
# fig.update_layout(
#     title='3D Time-Frequency Spectrum (Interactive)',
#     scene=dict(
#         xaxis_title='Frequency (Hz)',
#         yaxis_title='Time (s)',
#         zaxis_title='Amplitude'
#     )
# )
# fig.show()
# # ✅ HTML 파일로 저장
# fig.write_html("C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/3D_frequency_img/185683.html")
# print("저장 완료! HTML 파일 열어서 마우스로 돌려볼 수 있어요.")
