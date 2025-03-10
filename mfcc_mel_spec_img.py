import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter

# 음성 파일 경로 (원하는 파일 경로로 변경)
wav_path = "C:/Users/user/AI/NELOW/NELOW_AI/Testing_Aramoon/184179_20241002_10_50_03_126_L.wav"

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

    # FFT 주파수 축 생성
    freqs = np.fft.fftfreq(len(tfa_data3000), d=1 / samplerate)

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

    return NELOW_fft_data, freqs[:len(freqs)//2], wave_energy, wave_max_frequency
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
# 1. 음성 파일 로드
y, sr = librosa.load(wav_path, sr=None)  # 원본 샘플링 레이트 유지

# 2. 전처리 거쳐 MFCC 추출
processed_mfcc_original, _, processed_y_original, processed_sr_original = preprocess_wav(wav_path)

# 3. 주파수 값 추출
NELOW_fft_data, freqs, _, _ = get_NELOW_values(wav_path)
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
ax4.plot(freqs[:len(NELOW_fft_data)], NELOW_fft_data[:len(freqs)], color="purple")
ax4.set_title("Frequency Spectrum")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Amplitude")
ax4.grid()


plt.tight_layout()
plt.show()
