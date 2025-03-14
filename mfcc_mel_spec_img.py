import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
import os
from tqdm import tqdm


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


# 음성 파일 경로 (원하는 파일 경로로 변경)
wav_path = "C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/test_wav/"

train_plot = "C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/test_mel_img/"

for i in tqdm(os.listdir(wav_path), desc="Processing WAV files", unit="file"):
    train_plot = f"C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/test_mel_img/{i}.png"
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
    ax4.set_ylim(0, max(NELOW_fft_data) * 1.3)  # Set the y-axis limit
    ax4.grid()

    # # 🎨 3. 멜 스펙트로그램 시각화 (단독 출력)
    # plt.figure(figsize=(12, 6))  # 그래프 크기 조정
    # librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    # plt.colorbar(format="%+2.0f dB")  # 컬러바 추가
    # plt.title(f"Mel Spectrogram - {i}")  # 파일명 표시
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Mel)")

    plt.tight_layout()
    plt.savefig(train_plot, dpi=300)
    # plt.show()
