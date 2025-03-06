import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import numpy as np
import cv2
import os
from tqdm import tqdm  # tqdm 추가
import librosa
from scipy.fftpack import fft


# # 경로 설정
AI_model_training = 'Graph_Image/AI_Model/NELOW_GL_img_model_2.h5'
AI_model_testing = 'Graph_Image/AI_Model/NELOW_GL_img_model_2.h5'

# WAV_files_path_training = 'Training_SWM/WAV_files/'
# Numpy_files_path_training = 'Training_SWM/Numpy_files/'
#
# ✅ 예측할 데이터 폴더
wav_root_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/predict_WAV_files/"
output_data_dir = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/predict_data/"
csv_output_path = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/predict_result/"

#학습 그래프 경로/파일명 설정
train_plot = 'C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/plot_history/NELOW_GL_img_model_3.png'

model_training = 0

model_testing = 1

# 이미지 크기 설정
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100
CLASSES = 3  # 3가지 클래스

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

# 1️⃣ 데이터 로딩 및 전처리 함수 (이미지 2개를 1쌍으로 묶음)
def load_image_pair(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    img1 = cv2.resize(img1, IMG_SIZE) / 255.0  # 크기 변환 + 정규화
    img2 = cv2.resize(img2, IMG_SIZE) / 255.0  # 크기 변환 + 정규화

    return np.array(img1), np.array(img2)

# 2️⃣ CNN 모델 정의 (2개의 이미지 입력 → 3개 클래스 분류)
def build_dual_input_model():
    # 첫 번째 이미지 CNN 네트워크
    input1 = Input(shape=(128, 128, 3), name="image_1")
    x1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Flatten()(x1)

    # 두 번째 이미지 CNN 네트워크
    input2 = Input(shape=(128, 128, 3), name="image_2")
    x2 = Conv2D(32, (3, 3), activation="relu", padding="same")(input2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Flatten()(x2)

    # 두 개의 특징 벡터를 결합
    merged = Concatenate()([x1, x2])
    x = Dense(64, activation="relu")(merged)
    output = Dense(CLASSES, activation="softmax")(x)  # 3개 클래스 분류

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    return model

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

    # json파일
    NELOW_fft_data = tfa_data3000.tolist()  # 0~3000Hz까지의 FFT 변환 데이터 (주파수 스펙트럼)
    # 표준편차
    std_deviation = np.std(tfa_data)        # 표준편차 (소리의 변동 정도)
    # 누수강도
    wave_energy = np.average(tfa_data3000[startPos:stopPos])   # 가장 강한 주파수 대역의 평균 진폭 (누수 강도)
    # 최대주파수
    wave_max_frequency = np.argmax(tfa_data3000)               # 최대 주파수 성분의 위치(인덱스값) (주파수 Hz 단위)

    return NELOW_fft_data, std_deviation, wave_energy, wave_max_frequency

# ✅ WAV 파일의 Waveform 및 Spectrum 저장 함수
def save_graphs(file_path, output_folder):
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # ✅ 실제 클래스 자동 추출 (파일명 기준)
    actual_class = "unknown"
    if "_L" in file_name:
        actual_class = "leak"
    elif "_M" in file_name:
        actual_class = "meter"
    elif "_N" in file_name:
        actual_class = "no_leak"

    # ✅ 저장 폴더 설정 (클래스별 폴더)
    save_dir = os.path.join(output_folder, actual_class)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 파일 저장 경로 설정
    waveform_path = os.path.join(save_dir, f"{file_name}_waveform.png")
    spectrum_path = os.path.join(save_dir, f"{file_name}_spectrum.png")

    # ✅ 오디오 데이터 로드
    raw_y, raw_sr = librosa.load(file_path, sr=None)
    NELOW_fft_data, _, wave_energy, wave_max_frequency = get_NELOW_values(file_path)

    # ✅ 파형(Waveform) 그래프 저장
    fig_wave, ax1 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(raw_y, sr=raw_sr, ax=ax1)
    ax1.set_ylim(-1, 1)
    fig_wave.savefig(waveform_path, bbox_inches="tight", dpi=300)
    plt.close(fig_wave)

    # ✅ 주파수(Spectrum) 그래프 저장
    fig_spec, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(NELOW_fft_data, color="purple")
    ax3.set_ylim(0, max(NELOW_fft_data) * 1.3)
    fig_spec.savefig(spectrum_path, bbox_inches="tight", dpi=300)
    plt.close(fig_spec)

    return waveform_path, spectrum_path, actual_class, wave_energy, wave_max_frequency


# 3️⃣ 데이터 불러오기 함수 (폴더에서 이미지 불러와서 리스트로 저장)
def load_dataset(data_folder):
    images_1, images_2, labels = [], [], []
    class_mapping = {"leak": 0, "meter": 1, "no_leak": 2}  # 클래스 매핑

    for class_name in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_path):  # 폴더가 아닌 경우 무시
            continue

        files = sorted(os.listdir(class_path))  # 이미지 정렬
        for i in tqdm(range(0, len(files) - 1, 2), desc=f"📂 {class_name} 로딩 중", unit="pair"):  # 이미지 2개씩 쌍으로 불러오기
            img1_path = os.path.join(class_path, files[i])
            img2_path = os.path.join(class_path, files[i + 1])

            img1, img2 = load_image_pair(img1_path, img2_path)
            images_1.append(img1)
            images_2.append(img2)
            labels.append(class_mapping[class_name])  # 클래스 라벨 저장

    images_1 = np.array(images_1)
    images_2 = np.array(images_2)
    labels = tf.keras.utils.to_categorical(labels, num_classes=CLASSES)  # 원-핫 인코딩
    return images_1, images_2, labels
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

# 📌 이미지 예측 수행 함수
def predict_images(image_path1, image_path2, model):
    CLASSES = ["leak", "meter", "no_leak"]
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    img1 = cv2.resize(img1, (128, 128)) / 255.0
    img2 = cv2.resize(img2, (128, 128)) / 255.0

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    predictions = model.predict([img1, img2])
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASSES[predicted_class_idx]
    confidence = np.max(predictions) * 100

    return predicted_class, confidence


if model_training:
    # 4️⃣ 데이터 불러오기
    train_folder = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/train_data"  # 학습 데이터 폴더
    val_folder = "C:/Users/gram/AI/NELOW/NELOW_AI/Graph_Image/val_data"  # 검증 데이터 폴더

    x_train_1, x_train_2, y_train = load_dataset(train_folder)
    x_val_1, x_val_2, y_val = load_dataset(val_folder)

    print(f"학습 데이터 크기: {x_train_1.shape}, {x_train_2.shape}, {y_train.shape}")
    print(f"검증 데이터 크기: {x_val_1.shape}, {x_val_2.shape}, {y_val.shape}")

    # 5️⃣ 모델 생성 및 학습
    model = build_dual_input_model()
    model.summary()

    history = model.fit(
        [x_train_1, x_train_2], y_train, validation_data=([x_val_1, x_val_2], y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1
    )

    # 6️⃣ 모델 저장
    model.save(AI_model_training)
    print("모델 저장 완료!")
    # 그래프 출력
    plot_history(history)
    # 7️⃣ 모델 평가
    results = model.evaluate([x_val_1, x_val_2], y_val)

    # ✅ 평가 지표 출력
    metrics = ["Loss", "Accuracy", "F1 Score", "Precision", "Recall"]
    for metric, value in zip(metrics, results):
        print(f"{metric}: {value:.4f}")

if model_testing:
    # 📌 모델 로드 (컴파일 False 설정)
    AI_model = tf.keras.models.load_model(AI_model_testing, compile=False)

    # 📌 결과 CSV 파일 설정
    csv_file_path = os.path.join(csv_output_path, "test_results_신완주_소양.csv")

    # 📌 데이터 저장용 리스트
    filenames, max_amplitudes, max_frequencies = [], [], []
    real_label, AI_model_predict = [], []

    # 📌 WAV 파일 목록 가져오기 (폴더 내 모든 .wav 파일)
    wav_files = [os.path.join(wav_root_dir, f) for f in os.listdir(wav_root_dir) if f.endswith('.wav')]

    if not wav_files:
        print("⚠️ 경고: WAV 파일이 존재하지 않습니다. 파일을 확인하세요.")
    else:
        print(f"📂 총 {len(wav_files)} 개의 WAV 파일을 예측합니다.")

    # 📌 CSV 파일 작성 (쓰기 모드)
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["파일 이름", "소리 최대 진폭", "소리 최대 주파수", "실제 클래스", "예측 클래스"])  # CSV 헤더

        # 📌 WAV 파일 하나씩 처리 (진행률 표시)
        for wav_file in tqdm(wav_files, desc="🔍 WAV 파일 예측 중", unit="file"):
            try:
                # ✅ 음차트 및 주파수 차트 생성
                waveform_path, spectrum_path, actual_class, wave_energy, wave_max_frequency = save_graphs(wav_file, output_data_dir)
                print('최대 진폭', wave_energy)
                print('최대 주파수', wave_max_frequency)

                # ✅ 모델 예측 수행
                predicted_class, confidence = predict_images(waveform_path, spectrum_path, AI_model)
                print('예측 클래스', predicted_class, '======>', confidence)

                # # ✅ 최대 진폭 및 최대 주파수 값 계산
                max_amplitude = wave_energy
                max_frequency = wave_max_frequency

                # ✅ 데이터 리스트에 추가
                filenames.append(os.path.basename(wav_file))
                max_amplitudes.append(max_amplitude)
                max_frequencies.append(max_frequency)
                real_label.append(actual_class)
                AI_model_predict.append(predicted_class)
                print('clear data_list append')
                # ✅ CSV 파일에 저장
                writer.writerow(
                    [os.path.basename(wav_file), max_amplitude, max_frequency, actual_class, predicted_class])
                print('clear csv file save')
            except Exception as e:
                print(f"❌ 오류 발생: {wav_file} - {e}")

    # 📌 데이터프레임 생성
    final_df = pd.DataFrame({
        "파일_이름": filenames,
        "소리_최대_진폭": max_amplitudes,
        "소리_최대_주파수": max_frequencies,
        "실제_클래스": real_label,
        "예측_클래스": AI_model_predict
    })

    # ✅ 정렬: 소리 최대 진폭 기준으로 내림차순 정렬
    final_df = final_df.sort_values(by="소리_최대_진폭", ascending=False)

    # ✅ CSV 파일로 저장
    final_df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
    print(f"✅ 예측 결과가 CSV 파일로 저장되었습니다: {csv_file_path}")
