import numpy as np
import pandas as pd
import librosa
import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalMaxPooling2D, LeakyReLU
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.signal import butter, lfilter, filtfilt
import os

from keras import backend as K

from scipy.fftpack import fft

AI_model_testing = 'NELOW_AI_model/NELOW_GL_model_V3.h5'
AI_model_testing_edge = 'NELOW_AI_model/NELOW_GL_TFLite_model_test_V3.tflite'

WAV_files_path_testing = 'NELOW_V3/Testing_곡성/WAV_files/'
Numpy_files_path_testing = 'NELOW_V3/Testing_곡성/Numpy_files/'
CSV_files_path_testing = 'NELOW_V3/Testing_곡성/CSV_files/'

testing_sound_preprocessing = 0    # 음성파일(wav) numpy배열로 변환하여 저장

model_testing = 1
model_testing_edge = 1

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

    a = np.argmin(s_fft) + 1

    tfa_data = abs(fft(data[a * SEC_0_1:a * SEC_0_1 + SEC_1]))

    tfa_data3000 = tfa_data[0:3000]
    tfa_data3000[:50] = 0

    idx = np.argmax(tfa_data3000)

    startPos = 0

    if idx < 10:
        startPos = 0
    else:
        startPos = idx - 10

    stopPos = idx + 10


    # 누수강도
    wave_energy = np.average(tfa_data3000[startPos:stopPos])
    # 최대주파수
    wave_max_frequency = np.argmax(tfa_data3000)

    return wave_energy, wave_max_frequency

# Generate spectrogram from WAV file
# 오디오 파일을 로드하고 리샘플링을 수행한 다음 get_wav_clean1sec 및 get_wav_filtered 함수를 적용하여 오디오를 전처리하고 마지막으로 멜 주파수 켑스트럼 계수(MFCC)를 계산함.
def get_spec(path):
    q, w = librosa.load(path=path, sr=None)
    q=librosa.resample(q,orig_sr=w,target_sr=8000) # 원본 샘플링 레이팅에서 8000Hz 샘플링 레이트로 변환
    w=8000
    q, w = get_wav_clean1sec(q, w)
    q, w = get_wav_filtered(q, w)
    map=librosa.feature.mfcc(y=q,sr=w,n_fft=2048, hop_length=512,n_mfcc = 20)
    # y=q : 오디오 신호를 입력받음 / sr=w : 샘플링 레이트 / n_fft: FFT (Fast Fourier Transform)길이 지정 => 2048개의 샘플을 사용
    # hop_length: 프레임 간의 hop length (시간 간격)을 설정 / n_mfcc :  20개의 MFCC 계수를 계산하겠다는 의미
    # map은 2D numpy 배열로, 각 열은 각 시간 프레임의 MFCC값
    return map

# Save numpy array representations of spectrograms
# 디렉토리의 모든 WAV 파일의 MFCC 데이터를 NumPy 배열로 저장하여 추가 처리를 위해 사용함.
def save_npy(i_path,o_path):
    lis = os.listdir(i_path)

    for i in lis:
        if '.wav' in i:
            q=get_spec(i_path+i)
            q=q[:,:-3] # 마지막 3개의 열만 제거 (오디오 길이가 일정하지 않거나, 마지막 프레임이 불완전할 가능성이 높기 때문에 제거하는 것)

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
            a = np.load(npy_path + i)
            # print("Shape of a array:", a.shape)
            a = a.reshape(-1, 20, 13, 1)
            npy_table.append(a)
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

    npy_table = npy_table.reshape(-1, 20, 13, 1)
    label = label.reshape(-1, 3)
    filename = filename.reshape(-1, 1)

    return npy_table, label, filename

# TFLite 추론 함수
def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Prepare data if sound_preprocessing flag is true
# 음성파일(wav) numpy배열로 변환하여 저장
if testing_sound_preprocessing:
    save_npy(WAV_files_path_testing, Numpy_files_path_testing)


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

    # max_amplitudes, max_frequencies = [], []
    # for i in os.listdir(WAV_files_path_testing):
    #     wav_file = os.path.join(WAV_files_path_testing, i)
    #     max_amplitude, max_frequency = get_NELOW_values(wav_file)
    #     max_amplitudes.append(max_amplitude)
    #     max_frequencies.append(max_frequency)
    #
    #
    # # Reshaping filenames for concatenation and removing the last 4 characters
    # filenames = ee.reshape(len(ee), 1)
    # filenames = np.array([fname[0][:-4] for fname in filenames]).reshape(len(filenames), 1)
    # max_amplitudes = np.array(max_amplitudes).reshape(len(max_amplitudes), 1)  # Reshape max_amplitudes
    # max_frequencies = np.array(max_frequencies).reshape(len(max_frequencies), 1)  # Reshape max_frequencies
    #
    # label_max_reshaped = label_max.reshape(len(label_max), 1)
    # AI_model_predictions_max_reshaped = AI_model_predictions_max.reshape(len(AI_model_predictions_max),1)
    #
    # # Creating column names for the DataFrame
    # columns = ['파일_이름', '소리_최대_진폭', '소리_최대_주파수', 'Label', 'AI_모델_V3']
    # fin = np.concatenate((filenames, max_amplitudes, max_frequencies, label_max_reshaped, AI_model_predictions_max_reshaped), axis=1)
    # # Creating DataFrame
    # final_df = pd.DataFrame(fin, columns=columns)
    # # Ensuring 'Max_Amplitude' is treated as a numeric column
    # final_df['소리_최대_진폭'] = pd.to_numeric(final_df['소리_최대_진폭'], errors='coerce')
    # # Sorting DataFrame by 'Max_Amplitude' in descending order
    # final_df = final_df.sort_values(by='소리_최대_진폭', ascending=False)
    # # Saving to CSV
    # final_df.to_csv(CSV_files_path_testing + 'fixed_predictions_comparison_V3_hp.csv', index=False, encoding='utf-8-sig')

    # # Concatenating filenames, real labels, old predictions, and new predictions
    # final_data = np.concatenate((filenames, max_amplitudes, max_frequencies, label, AI_model_predictions), axis=1)
    # # Creating column names for the DataFrame
    # columns_2 = ['파일_이름',  '소리_최대_진폭', '소리_최대_주파수', 'Label_L', 'Label_M', 'Label_N', 'AI_모델_V5_L', 'AI_모델_V5_M', 'AI_모델_V5_N']
    # # Creating DataFrame
    # final_df_2 = pd.DataFrame(final_data, columns=columns_2)
    # # Ensuring 'Max_Amplitude' is treated as a numeric column
    # final_df_2['소리_최대_진폭'] = pd.to_numeric(final_df_2['소리_최대_진폭'], errors='coerce')
    # # Sorting DataFrame by 'Max_Amplitude' in descending order
    # final_df_2 = final_df_2.sort_values(by='소리_최대_진폭', ascending=False)
    # # Saving to CSV
    # final_df_2.to_csv(CSV_files_path_testing + 'probability_predictions_comparison_V5_곡성.csv', index=False, encoding='utf-8-sig')
    #
    # summary = {}
    #
    # # Calculate counts of each label
    # counts = np.bincount(AI_model_predictions_max, minlength=3)
    # summary[AI_model_testing] = {
    #     "Leak": counts[0],
    #     "Meter": counts[1],
    #     "No leak": counts[2]
    # }
    #
    # # Print the results
    # for model, results in summary.items():
    #     print(f"Results for AI_모델_곡성:")
    #     print(f"Leak: {results['Leak']}, Meter: {results['Meter']}, No leak: {results['No leak']}")
    #
    # # Create a DataFrame and write to CSV
    # df_summary = pd.DataFrame.from_dict(summary, orient='index')
    # df_summary.to_csv(f'{CSV_files_path_testing}summary_predictions_comparison_V5_곡성.csv', encoding='utf-8-sig')

if model_testing_edge:

    # 1. TFLite 모델 로드
    interpreter = tf.lite.Interpreter(model_path=AI_model_testing_edge)
    interpreter.allocate_tensors()
    # 2. 테스트 데이터 로드
    npy_table, label, ee = load_npy(Numpy_files_path_testing)
    AI_model_predictions = []
    for i in range(len(npy_table)):
        input_data = np.expand_dims(npy_table[i], axis=0)  # 배치 차원 추가
        output_data = predict_with_tflite(interpreter, input_data)
        AI_model_predictions.append(output_data[0])
    AI_model_predictions = np.array(AI_model_predictions)
    # 3. 예측 결과 정리
    label_max = np.array(np.argmax(label, axis=1))
    AI_model_predictions_max = np.argmax(AI_model_predictions, axis=1)
    accuracy = accuracy_score(label_max, AI_model_predictions_max)
    print(f"Accuracy : {accuracy:.4f}")

    # max_amplitudes, max_frequencies = [], []
    # for i in os.listdir(WAV_files_path_testing):
    #     wav_file = os.path.join(WAV_files_path_testing, i)
    #     max_amplitude, max_frequency = get_NELOW_values(wav_file)
    #     max_amplitudes.append(max_amplitude)
    #     max_frequencies.append(max_frequency)
    #
    # # Reshaping filenames for concatenation and removing the last 4 characters
    # filenames = ee.reshape(len(ee), 1)
    # filenames = np.array([fname[0][:-4] for fname in filenames]).reshape(len(filenames), 1)
    # max_amplitudes = np.array(max_amplitudes).reshape(len(max_amplitudes), 1)  # Reshape max_amplitudes
    # max_frequencies = np.array(max_frequencies).reshape(len(max_frequencies), 1)  # Reshape max_frequencies
    #
    # label_max_reshaped = label_max.reshape(len(label_max), 1)
    # AI_model_predictions_max_reshaped = AI_model_predictions_max.reshape(len(AI_model_predictions_max),1)
    #
    # # Creating column names for the DataFrame
    # columns = ['파일_이름', '소리_최대_진폭', '소리_최대_주파수', 'Label', 'EdgeAI']
    # fin = np.concatenate((filenames, max_amplitudes, max_frequencies, label_max_reshaped, AI_model_predictions_max_reshaped), axis=1)
    # # Creating DataFrame
    # final_df = pd.DataFrame(fin, columns=columns)
    # # Ensuring 'Max_Amplitude' is treated as a numeric column
    # final_df['소리_최대_진폭'] = pd.to_numeric(final_df['소리_최대_진폭'], errors='coerce')
    # # Sorting DataFrame by 'Max_Amplitude' in descending order
    # final_df = final_df.sort_values(by='소리_최대_진폭', ascending=False)
    # # Saving to CSV
    # final_df.to_csv(CSV_files_path_testing + 'fixed_predictions_comparison_EdgeAI_test_곡성.csv', index=False, encoding='utf-8-sig')
    #
    # # Concatenating filenames, real labels, old predictions, and new predictions
    # final_data = np.concatenate((filenames, max_amplitudes, max_frequencies, label, AI_model_predictions), axis=1)
    # # Creating column names for the DataFrame
    # columns_2 = ['파일_이름',  '소리_최대_진폭', '소리_최대_주파수', 'Label_L', 'Label_M', 'Label_N', 'EdgeAI_L', 'EdgeAI_M', 'EdgeAI_N']
    # # Creating DataFrame
    # final_df_2 = pd.DataFrame(final_data, columns=columns_2)
    # # Ensuring 'Max_Amplitude' is treated as a numeric column
    # final_df_2['소리_최대_진폭'] = pd.to_numeric(final_df_2['소리_최대_진폭'], errors='coerce')
    # # Sorting DataFrame by 'Max_Amplitude' in descending order
    # final_df_2 = final_df_2.sort_values(by='소리_최대_진폭', ascending=False)
    # # Saving to CSV
    # final_df_2.to_csv(CSV_files_path_testing + 'probability_predictions_comparison_EdgeAI_test_곡성.csv', index=False, encoding='utf-8-sig')
    #
    # summary = {}
    #
    # # Calculate counts of each label
    # counts = np.bincount(AI_model_predictions_max, minlength=3)
    # summary[AI_model_testing] = {
    #     "Leak": counts[0],
    #     "Meter": counts[1],
    #     "No leak": counts[2]
    # }
    #
    # # Print the results
    # for model, results in summary.items():
    #     print(f"Results for AI_모델_EdgeAI:")
    #     print(f"Leak: {results['Leak']}, Meter: {results['Meter']}, No leak: {results['No leak']}")
    #
    # # Create a DataFrame and write to CSV
    # df_summary = pd.DataFrame.from_dict(summary, orient='index')
    # df_summary.to_csv(f'{CSV_files_path_testing}summary_predictions_comparison_EdgeAI_test_곡성.csv', encoding='utf-8-sig')

