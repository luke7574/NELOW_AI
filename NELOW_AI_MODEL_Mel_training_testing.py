import numpy as np
import pandas as pd
import librosa
import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalMaxPooling2D, LeakyReLU
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.signal import butter, lfilter, filtfilt
import os

from keras import backend as K

from scipy.fftpack import fft

AI_model_training = 'MEL_Spectrogram/AI_MODEL/NELOW_MEL_model_1.h5'
AI_model_testing = 'MEL_Spectrogram/AI_MODEL/NELOW_MEL_model_1.h5'

# 훈련 데이터셋
WAV_files_path_training = 'MEL_Spectrogram/test_WAV_FILES/'
Numpy_files_path_training = 'MEL_Spectrogram/test_NUMPY_FILES/'

# 테스트 데이터셋
WAV_files_path_testing = 'MEL_Spectrogram/Testing/WAV_files/'
Numpy_files_path_testing = 'MEL_Spectrogram/Testing/Numpy_files/'
CSV_files_path_testing = 'MEL_Spectrogram/Testing/CSV_files/'

training_sound_preprocessing = 1   # 음성파일(wav) numpy배열로 변환하여 저장
model_training = 0
train_plot = 'C:/Users/gram/AI/NELOW/NELOW_AI/MEL_Spectrogram/plot_history/NELOW_MEL_model_1.png'   #학습 그래프 경로/파일명 설정

testing_sound_preprocessing = 0    # 음성파일(wav) numpy배열로 변환하여 저장
model_testing = 0


# Define recall metric
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Define precision metric
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Define F1 score metric
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Select clean 1-second segment from the signal
# 음성신호에서 1초 길이의 깨끗한 구간 추출
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

# 소리 최대 진폭(누수강도) , 소리 최대 주파수 구하기
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

    idx = np.argmax(tfa_data3000) # 가장 큰 주파수 성분의 인덱스(주파수 위치)

    startPos = 0

    if idx < 10:
        startPos = 0
    else:
        startPos = idx - 10
    stopPos = idx + 10

    # 누수강도
    wave_energy = np.average(tfa_data3000[startPos:stopPos])   # 가장 강한 주파수 대역의 평균 진폭 (누수 강도)
    # 최대주파수
    wave_max_frequency = np.argmax(tfa_data3000)               # 최대 주파수 성분의 위치(인덱스값) (주파수 Hz 단위)

    return wave_energy, wave_max_frequency

# 1초구간 찾고 -> 대역통과필터 거치고 -> fft 수행 -> input_data 완성
def get_spec(path):
    data, sr = librosa.load(path=path, sr=None)
    data = librosa.resample(data,orig_sr=sr,target_sr=8000) # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
    sr = 8000
    data, sr = get_wav_clean1sec(data, sr)
    data, sr = get_wav_filtered(data, sr)

    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # 로그 변환 적용

    return mel_spec_db

# Save numpy array representations of spectrograms
# 디렉토리의 모든 WAV 파일의 MFCC 데이터를 NumPy 배열로 저장하여 추가 처리를 위해 사용함.
def save_npy(i_path,o_path):
    lis = os.listdir(i_path)

    for i in lis:
        if '.wav' in i:
            q=get_spec(i_path+i)
            np.save(o_path+i+'.npy',q)
    return

# Load numpy arrays and prepare them for the model
# NumPy 배열로 저장된 전처리된 오디오 데이터를 로드하도록 설계됨.
# 파일 이름의 특정 문자에 따라 레이블이 할당되며 L, M, N은 각각 누수 소리, 미터 소리 및 누수 없는 소리로 표기.
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

# 학습 과정 그래프 출력
def plot_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Accuracy 그래프
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    # Loss 그래프
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Precision 그래프
    axes[1, 0].plot(history.history['precision_m'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision_m'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()

    # Recall & F1-score 그래프
    axes[1, 1].plot(history.history['recall_m'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall_m'], label='Validation Recall')
    axes[1, 1].plot(history.history['f1_m'], label='Train F1-score')
    axes[1, 1].plot(history.history['val_f1_m'], label='Validation F1-score')
    axes[1, 1].set_title('Recall & F1-score')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()

    plt.tight_layout()
    # 그래프를 이미지 파일로 저장
    plt.savefig(train_plot, dpi=300)
    plt.show()



# Prepare data if sound_preprocessing flag is true
# 음성파일(wav) numpy배열로 변환하여 저장
if training_sound_preprocessing:
    save_npy(WAV_files_path_training, Numpy_files_path_training)
if testing_sound_preprocessing:
    save_npy(WAV_files_path_testing, Numpy_files_path_testing)


# Train model if model_training flag is true
if model_training:
    model = tf.keras.models.Sequential()

    # Convolutional neural network architecture
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(128, 16, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))  # 활성함수 relu 변형함수로 음의 입력에 대해서도 작은 기울기를 제공
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

    q,w,e=load_npy(Numpy_files_path_training)
    # model.fit(q,w,epochs=100,batch_size=200)
    history = model.fit(q, w, epochs=100, batch_size=200, validation_split=0.2, shuffle=True)
    model.save(AI_model_training)
    # 그래프 출력
    plot_history(history)

from sklearn.metrics import accuracy_score
# Evaluate the model if model_testing flag is true
if model_testing:

    AI_model = load_model(AI_model_testing, compile=False)

    npy_table, label, ee = load_npy(Numpy_files_path_testing)
    # Getting predictions
    AI_model_predictions = AI_model.predict(npy_table)

    label_max = np.array(np.argmax(label, axis=1))
    # print(label_max)
    AI_model_predictions_max = np.array(np.argmax(AI_model_predictions, axis=1))

    accuracy = accuracy_score(label_max, AI_model_predictions_max)
    print(f"Accuracy : {accuracy:.4f}")

    max_amplitudes, max_frequencies = [], []
    for i in os.listdir(WAV_files_path_testing):
        wav_file = os.path.join(WAV_files_path_testing, i)
        max_amplitude, max_frequency = get_NELOW_values(wav_file)
        max_amplitudes.append(max_amplitude)
        max_frequencies.append(max_frequency)


    # Reshaping filenames for concatenation and removing the last 4 characters
    filenames = ee.reshape(len(ee), 1)
    filenames = np.array([fname[0][:-4] for fname in filenames]).reshape(len(filenames), 1)
    max_amplitudes = np.array(max_amplitudes).reshape(len(max_amplitudes), 1)  # Reshape max_amplitudes
    max_frequencies = np.array(max_frequencies).reshape(len(max_frequencies), 1)  # Reshape max_frequencies

    label_max_reshaped = label_max.reshape(len(label_max), 1)
    AI_model_predictions_max_reshaped = AI_model_predictions_max.reshape(len(AI_model_predictions_max),1)

    # Creating column names for the DataFrame
    columns = ['파일_이름', '소리_최대_진폭', '소리_최대_주파수', 'Label', 'model_2']
    fin = np.concatenate((filenames, max_amplitudes, max_frequencies, label_max_reshaped, AI_model_predictions_max_reshaped), axis=1)
    # Creating DataFrame
    final_df = pd.DataFrame(fin, columns=columns)
    # Ensuring 'Max_Amplitude' is treated as a numeric column
    final_df['소리_최대_진폭'] = pd.to_numeric(final_df['소리_최대_진폭'], errors='coerce')
    # Sorting DataFrame by 'Max_Amplitude' in descending order
    final_df = final_df.sort_values(by='소리_최대_진폭', ascending=False)
    # Saving to CSV
    final_df.to_csv(CSV_files_path_testing + 'fixed_predictions_comparison_model_2.csv', index=False, encoding='utf-8-sig')

    # Concatenating filenames, real labels, old predictions, and new predictions
    final_data = np.concatenate((filenames, max_amplitudes, max_frequencies, label, AI_model_predictions), axis=1)
    # Creating column names for the DataFrame
    columns_2 = ['파일_이름',  '소리_최대_진폭', '소리_최대_주파수', 'Label_L', 'Label_M', 'Label_N', 'model_2_L', 'model_2_M', 'model_2_N']
    # Creating DataFrame
    final_df_2 = pd.DataFrame(final_data, columns=columns_2)
    # Ensuring 'Max_Amplitude' is treated as a numeric column
    final_df_2['소리_최대_진폭'] = pd.to_numeric(final_df_2['소리_최대_진폭'], errors='coerce')
    # Sorting DataFrame by 'Max_Amplitude' in descending order
    final_df_2 = final_df_2.sort_values(by='소리_최대_진폭', ascending=False)
    # Saving to CSV
    final_df_2.to_csv(CSV_files_path_testing + 'probability_predictions_comparison_model_2.csv', index=False, encoding='utf-8-sig')

    summary = {}

    # Calculate counts of each label
    counts = np.bincount(AI_model_predictions_max, minlength=3)
    summary[AI_model_testing] = {
        "Leak": counts[0],
        "Meter": counts[1],
        "No leak": counts[2]
    }

    # Print the results
    for model, results in summary.items():
        print(f"Results for model_2:")
        print(f"Leak: {results['Leak']}, Meter: {results['Meter']}, No leak: {results['No leak']}")

    # Create a DataFrame and write to CSV
    df_summary = pd.DataFrame.from_dict(summary, orient='index')
    df_summary.to_csv(f'{CSV_files_path_testing}summary_predictions_comparison_model_2.csv', encoding='utf-8-sig')
