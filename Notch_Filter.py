import numpy as np
import librosa
import soundfile as sf
from scipy.signal import iirnotch, filtfilt
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os

notch_filter = 0
remove_electronic = 0
apply_notch = 1

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
# 전기음 제거 알고리즘
def adjust_spectral_peaks_with_window(y, sr, window_size=50, threshold_ratio=3, method='mean'):
    # Perform STFT
    D = librosa.stft(y)
    D_magnitude, D_phase = librosa.magphase(D) # D_magnitude: 복소수 STFT의 크기 (진폭) / D_phase: 위상 (복원 시 필요)
    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])

    # Limit frequency range to 20-1000 Hz
    idx = np.where((freqs >= 20) & (freqs <= 1000))[0]
    limited_magnitude = D_magnitude[idx, :]

    # Calculate the overall median amplitude within the 20-1000 Hz range
    global_median = np.median(limited_magnitude) # 중앙값
    max_peak = np.max(limited_magnitude)         # 최대값

    # Only adjust peaks if the maximum peak is more than ten times the global median
    if max_peak > 10 * global_median:
        # Adjust the magnitude of peaks
        half_window = window_size // 2
        for t in range(D_magnitude.shape[1]): # 시간 프레임
            for i in range(D_magnitude.shape[0]): # 주파수
                # Define window boundaries
                start_index = max(i - half_window, 0)
                end_index = min(i + half_window + 1, D_magnitude.shape[0])
                # Compute the average or median magnitude within the window
                if method == 'median':
                    window_stat = np.median(D_magnitude[start_index:end_index, t])
                elif method == 'mean':
                    window_stat = np.mean(D_magnitude[start_index:end_index, t])

                # Check if the current point is a significant peak
                if D_magnitude[i, t] > threshold_ratio * window_stat:
                    D_magnitude[i, t] = window_stat

    # Reconstruct the STFT matrix
    adjusted_D = D_magnitude * D_phase
    # Perform the inverse STFT to convert back to time domain
    adjusted_y = librosa.istft(adjusted_D)
    return adjusted_y, sr


# IIR Notch 필터 적용 함수
def apply_notch_filter(signal, sr, notch_freq, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, sr)
    filtered_signal = filtfilt(b, a, signal)  # 양방향 필터링
    return filtered_signal

# 전기 제거 알고리즘 + 노치필터 적용
def adjust_peaks_with_notch(y, sr, window_size=50, threshold_ratio=5, method='mean', quality_factor=60):
    # 1. STFT로 변환
    D = librosa.stft(y)
    D_magnitude, D_phase = librosa.magphase(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    # 2. 피크 탐색 범위 인덱스
    idx = np.where((freqs >= 20) & (freqs <= 1000))[0]
    print(D_magnitude.shape[0])
    limited_magnitude = D_magnitude[idx, :]

    global_median = np.median(limited_magnitude)  # 중앙값
    max_peak = np.max(limited_magnitude)  # 최대값
    # 3. 각 프레임에서 피크 후보 탐색
    peak_freqs = set()
    half_window = window_size // 2
    if max_peak > 10 * global_median:
        for t in range(D_magnitude.shape[1]):
            mag_slice = D_magnitude[idx, t]
            for i, mag in enumerate(mag_slice):  # enumerate 사용!
                start = max(i - half_window, 0)
                end = min(i + half_window + 1, len(mag_slice))
                window_stat = np.median(mag_slice[start:end]) if method == 'median' else np.mean(mag_slice[start:end])
                if mag > threshold_ratio * window_stat:
                    peak_freq = freqs[idx[i]]  # i는 0~len(idx)-1
                    peak_freqs.add(round(peak_freq))

    print(f"Detected peak frequencies: {sorted(peak_freqs)}")

    # 4. time domain에서 노치필터 적용
    y_filtered = y.copy()
    for pf in peak_freqs:
        if pf < sr // 2:  # 나이퀴스트 초과 방지
            b, a = iirnotch(pf, quality_factor, sr)
            y_filtered = filtfilt(b, a, y_filtered)

    return y_filtered, sr

# ✅ FFT 이미지 저장 함수
def save_fft_plot(signal, samplerate, title, save_path):
    fft_data = np.abs(fft(signal))
    half_len = len(fft_data) // 2
    fft_data = fft_data[:half_len]
    freqs = np.linspace(0, samplerate / 2, half_len)

    plt.figure(figsize=(12, 5))
    plt.plot(freqs, fft_data)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ FFT 그래프 저장 완료: {save_path}")


# 1초 신호 로드
wav_path = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/notch_filter/변압기/변압기원본.wav"
signal, sr = librosa.load(wav_path, sr=None)  # 1초 신호만 로드
sec_signal, sr = get_wav_clean1sec(signal, sr)
output_path = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/notch_filter/변압기/원본1sec.wav"
sf.write(output_path, sec_signal.astype(np.float32), sr)


if notch_filter:
    # 전기음 주파수 제거 (60Hz, 120Hz, 180Hz, 240Hz)
    notch_freqs = [60, 120, 180, 240, 300, 360]
    filtered_signal = sec_signal
    for freq in notch_freqs:
        filtered_signal = apply_notch_filter(filtered_signal, sr, freq)

    # 결과 저장
    output_path = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/notch_filter/notch_1sec.wav"
    sf.write(output_path, filtered_signal.astype(np.float32), sr)
    print(f"✅ 필터링된 신호 저장 완료: {output_path}")


if remove_electronic:
    adjusted_data, samplerate = adjust_spectral_peaks_with_window(sec_signal, sr)

    # ✅ 파일 경로 지정 (이미 있는 주파수 범위 활용)
    path = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/notch_filter"
    # path = f"/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/FH102/3024_20250613_033000/테스트/remove_elec"       # Macbook
    # 음원 복원하기
    output_wav_path = os.path.join(path, "누수adjusted_output_1sec.wav")
    sf.write(output_wav_path, adjusted_data.astype(np.float32), samplerate)
    print(f"✅ 복원된 음원 저장 완료: {output_wav_path}")

if apply_notch:
    adjusted_data1, samplerate = adjust_peaks_with_notch(sec_signal, sr)

    # ✅ 파일 경로 지정 (이미 있는 주파수 범위 활용)
    path = "C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/notch_filter/변압기"
    # path = f"/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/FH102/3024_20250613_033000/테스트/remove_elec"       # Macbook
    # 음원 복원하기
    output_wav_path = os.path.join(path, "q_f60.wav")
    sf.write(output_wav_path, adjusted_data1.astype(np.float32), samplerate)
    print(f"✅ 복원된 음원 저장 완료: {output_wav_path}")