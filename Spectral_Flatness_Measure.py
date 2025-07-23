import numpy as np
import librosa
import soundfile as sf

def get_wav_clean1sec(signal, sr):
    SEC_0_1 = sr // 10  # 0.1초 샘플 개수
    SEC_1 = sr  # 1초 샘플 개수
    duration = int(len(signal) / sr)  # 오디오의 총 길이 (초단위)
    s_fft = []
    i_time = (duration - 1) * 10 - 1  # 검사할 1초 구간의 개수
    for i in range(i_time):
        u_data = signal[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1]  # 100ms 간격으로 이동하며 1초 길이의 신호 추출
        s_fft.append(np.std(u_data))
    a = np.argmin(s_fft) + 1
    tfa_data = signal[a * SEC_0_1: a * SEC_0_1 + SEC_1]
    start_time = a * 0.1  # 시작 시간 (초 단위)
    end_time = start_time + 1  # 끝 시간 (1초 길이)
    return tfa_data, start_time, end_time, sr


def get_wav_clean1sec_sfm(signal, sr):
    SEC_0_1 = sr // 10  # 0.1초 샘플 개수
    SEC_1 = sr  # 1초 샘플 개수
    duration = int(len(signal) / sr)
    s_sfm = []

    i_time = (duration - 1) * 10 - 1  # 검사할 1초 구간의 개수 (0.1초 간격으로 밀기 위해)
    for i in range(i_time):
        u_data = signal[i * SEC_0_1: (i * SEC_0_1 + SEC_1)]  # 0.1초 간격으로 이동하며 1초 길이의 신호 추출
        # 스펙트럼을 구하고, 스펙트럼 평탄도 계산
        S = np.abs(np.fft.rfft(u_data))  # FFT 계산 (주파수 스펙트럼)
        sfm = np.mean(S ** 2) / (np.mean(np.abs(S)) ** 2)  # Spectral Flatness Measure (평탄도 계산)
        s_sfm.append(sfm)

    # 가장 평탄한 구간을 선택 (SFM 값이 낮을수록 평탄한 구간)
    a = np.argmin(s_sfm)  # SFM이 가장 작은 구간 선택 (깨끗한 구간)
    start_time = a * 0.1  # 시작 시간 (초 단위)
    end_time = start_time + 1  # 끝 시간 (1초 길이)
    tfa_data = signal[a * SEC_0_1: a * SEC_0_1 + SEC_1]  # 해당 구간 추출

    return tfa_data, start_time, end_time, sr


# 음원 파일 로드
file_path = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/SFM/원본/누수.wav'
signal, sr = librosa.load(file_path, sr=None)

# 두 함수 실행
tfa_data_std, start_time_std, end_time_std, sr = get_wav_clean1sec(signal, sr)
tfa_data_sfm, start_time_sfm, end_time_sfm, sr = get_wav_clean1sec_sfm(signal, sr)

# 결과 출력
print(f"Standard Deviation method selected clean section: {start_time_std:.2f} - {end_time_std:.2f} seconds")
print(f"Spectral Flatness Measure method selected clean section: {start_time_sfm:.2f} - {end_time_sfm:.2f} seconds")

# 1초 구간을 각각 새로운 음원으로 저장
output_path_std = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/SFM/clean_section_std.wav'
output_path_sfm = 'C:/Users/user/AI/NELOW/NELOW_AI/새로운 데이터셋/SFM/clean_section_sfm.wav'

# 저장 (tfa_data_std와 tfa_data_sfm 각각 1초 구간 데이터)
sf.write(output_path_std, tfa_data_std.astype(np.float32), sr)
sf.write(output_path_sfm, tfa_data_sfm.astype(np.float32), sr)

print(f"Saved clean section from Standard Deviation method to {output_path_std}")
print(f"Saved clean section from Spectral Flatness Measure method to {output_path_sfm}")