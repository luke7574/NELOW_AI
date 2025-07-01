##### V6 전처리 ######


# 기존과 동일
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



# minFreq 20(기존) ==> 50(변경후), maxFreq 1000(기존) ==> 2000(변경후) , lfilter(기존) ==> filtfilt(변경후)
def get_wav_filtered_filt(signal,sr):
    minFreq=50; maxFreq=2000; order=5

    nyq = 0.5 * sr
    low = minFreq / nyq
    high = maxFreq / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered = filtfilt(b, a, signal)

    return filtered, sr


# MFCC (기존) ==> MEL_sepctrogram(변경후)
def get_spec(path):
    data, sr = librosa.load(path=path, sr=None)
    data = librosa.resample(data,orig_sr=sr,target_sr=8000) # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
    sr = 8000
    data, sr = get_wav_clean1sec(data, sr)
    data, sr = get_wav_filtered_filt(data, sr)

    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # 로그 변환 적용

    return mel_spec_db



# q=q[:,:-3] (기존)  ==>  제거 (변경후)
def save_npy(i_path,o_path):
    lis = os.listdir(i_path)

    for i in lis:
        if '.wav' in i:
            q=get_spec(i_path+i)
            np.save(o_path+i+'.npy',q)
    return



# .reshape(-1, 20, 13, 1) (기존)  ==> .reshape(-1, 128, 16, 1)
def load_npy(npy_path):  # npy_path == Testing/Numpy_files/
    npy_table = []
    label = []
    filename = []
    lis = os.listdir(npy_path)

    for i in lis:
        if '.npy' in i:
            fft_data = np.load(npy_path + i)
            # print("Shape of a array:", a.shape)
            fft_data = fft_data.reshape(-1, 128, 16, 1) # 2D CNN 입력 형태로 변환
            npy_table.append(fft_data)
            if i[-9] == 'L':
                label.append(0)
            elif i[-9] == 'M':
                label.append(1)
            else:
                label.append(2)
            filename.append(i)

    label = tf.keras.utils.to_categorical(label, num_classes=3)
    npy_table = np.array(npy_table) # input
    label = np.array(label)         # output
    filename = np.array(filename)

    npy_table = npy_table.reshape(-1, 128, 16, 1)
    label = label.reshape(-1, 3)
    filename = filename.reshape(-1, 1)

    return npy_table, label, filename