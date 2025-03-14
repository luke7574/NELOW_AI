import tkinter as tk
from tkinter import filedialog, font, PhotoImage, Label
import numpy as np
import librosa
import os
import sys
import subprocess  # For a cross-platform solution
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv

import keras
import tensorflow
from tensorflow.keras.models import load_model

from scipy.signal import butter, lfilter
from keras import backend as K
from tensorflow.keras.utils import custom_object_scope
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec  # Add gridspec for custom layout
from scipy.fftpack import fft


global wav_files, current_file_index, filename_var
wav_files = []
current_file_index = 0
global canvas
canvas = None  # Initialize the canvas as None

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


def adjust_spectral_peaks_with_window(y, sr, window_size=10, threshold_ratio=3, method='median'):
    """
    Adjust spectral peaks using a sliding window approach.

    :param y: Time-domain audio signal.
    :param sr: Sampling rate of the audio.
    :param window_size: Size of the sliding window used to calculate the average.
    :param threshold_ratio: Ratio above which a point is considered a peak.
    :return: Adjusted time-domain audio signal and sampling rate.
    """
    # Perform STFT
    D = librosa.stft(y)
    D_magnitude, D_phase = librosa.magphase(D)

    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0]) #주어진 샘플 레이티와 n_fft를 주파수 축을 계산값 => 이 값은 실제 D_magnitude 행렬의 각 행이 어떤 주파수를 나타내는지 매핑하는 역할

    # Limit frequency range to 20-1000 Hz
    idx = np.where((freqs >= 20) & (freqs <= 1000))[0]
    limited_magnitude = D_magnitude[idx, :]

    # Calculate the overall median amplitude within the 20-1000 Hz range
    global_median = np.median(limited_magnitude) # 전체 20~1000Hz 주파수 성분의 중앙값(중위수, median) 을 계산 → 평균적인 강도 수준 확인.
    max_peak = np.max(limited_magnitude) # 해당 주파수 범위에서 가장 강한(큰) 피크 값을 찾음.

    # Only adjust peaks if the maximum peak is more than ten times the global median
    if max_peak > 10 * global_median:
        # Adjust the magnitude of peaks
        half_window = window_size // 2
        for t in range(D_magnitude.shape[1]):
            for i in range(D_magnitude.shape[0]):
                # Define window boundaries
                start_index = max(i - half_window, 0)
                end_index = min(i + half_window + 1, D_magnitude.shape[0])
                # Compute the average or median magnitude within the window
                if method == 'median':
                    window_stat = np.median(D_magnitude[start_index:end_index, t])
                elif method == 'mean':
                    window_stat = np.mean(D_magnitude[start_index:end_index, t])

                # Check if the current point is a significant peak
                if D_magnitude[i, t] > threshold_ratio * window_stat: # D_magnitude[i, t] 현재값 / window_stat 중앙값
                    D_magnitude[i, t] = window_stat

    # Reconstruct the STFT matrix
    adjusted_D = D_magnitude * D_phase

    # Perform the inverse STFT to convert back to time domain
    adjusted_y = librosa.istft(adjusted_D)

    return adjusted_y, sr

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


def predict(models, processed_wav):
    results = []
    probabilities = []

    for model in models:
        prediction = model.predict(processed_wav, verbose=0)
        probabilities.append(prediction)
        results.append(np.argmax(prediction, axis=1))

    return results, probabilities

def load_models():
    model_paths = [
        # 'C:/Users/wipla/OneDrive/Emmanuel/NELOW_AI/NELOW_AI_python_project/NELOW_AI_model/NELOW_GL_model_V3.h5'
        #문욱경로
        'C:/Users/user/AI/NELOW/NELOW_AI/NELOW_AI_model/NELOW_GL_model_V3.h5'
        # 'C:/Users/wipla/OneDrive/Emmanuel/NELOW_AI/NELOW_AI_python_project/NELOW_AI_model/NELOW_GL_model_V3.0.1.h5'
    ]

    custom_metrics = {
        'recall_m': recall_m,
        'precision_m': precision_m,
        'f1_m': f1_m
    }

    models = []
    for mp in model_paths:
        with custom_object_scope(custom_metrics):
            model = load_model(mp)
            models.append(model)
    return models

def go_to_first_file():
    global current_file_index
    if wav_files:  # Check if the list is not empty
        current_file_index = 0
        load_file(wav_files[current_file_index])

def go_to_last_file():
    global current_file_index
    if wav_files:  # Check if the list is not empty
        current_file_index = len(wav_files) - 1
        load_file(wav_files[current_file_index])

def next_file(skip):
    global current_file_index
    if current_file_index < len(wav_files) - skip:
        current_file_index += skip
        load_file(wav_files[current_file_index])
    else:
        current_file_index = len(wav_files) - 1  # Prevents going out of range
    load_file(wav_files[current_file_index])

def previous_file(skip):
    global current_file_index
    if current_file_index > 0 + skip:
        current_file_index -= skip
    else:
        current_file_index = 0  # Prevents going out of range
    load_file(wav_files[current_file_index])


def open_wav_file(file_path):
    if os.name == 'nt':  # Windows
        os.startfile(file_path)
    else:  # MacOS and Linux settings
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, file_path])


# Define a global dictionary to store NELOW strengths for each file
file_nelow_strengths = {}

def calculate_nelow_strengths(directory):
    global file_nelow_strengths
    file_nelow_strengths = {}

    for file in wav_files:
        file_path = os.path.join(directory, file)
        _, _, NELOW_strength_value, _ = get_NELOW_values(file_path)
        file_nelow_strengths[file_path] = NELOW_strength_value

    # Sort the wav_files list based on the NELOW strength values
    wav_files.sort(key=lambda x: file_nelow_strengths[x], reverse=True)



def select_file():
    global wav_files, current_file_index
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    if file_path:
        directory = os.path.dirname(file_path)
        # Normalize file paths and ensure they are absolute
        wav_files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.wav')]
        file_path = os.path.abspath(file_path)  # Normalize the selected file path

        try:
            calculate_nelow_strengths(directory)  # Calculate and sort based on NELOW strength
            current_file_index = wav_files.index(file_path)
            load_file(wav_files[current_file_index])

        except ValueError:
            print("File not found in the directory listing. Please check the file path and directory.")
            return  # Exit the function if the file is not found


def load_file(file_path):
    global canvas, filename_var, NELOW_strength, NELOW_max_frequency, file_label_var

    if canvas:
        canvas.get_tk_widget().destroy()  # Destroy the existing canvas if it exists

    # Update the filename display to include index
    current_index_display = f"{current_file_index + 1}/{len(wav_files)}: {os.path.basename(file_path)}"
    filename_var.set(current_index_display)  # Update the filename display

    _, processed_wav_original, y_original, sr_original = preprocess_wav(file_path)
    _, _, NELOW_strength_value, NELOW_max_frequency_value = get_NELOW_values(file_path)

    NELOW_strength.set(f"NELOW 소리 강도: {NELOW_strength_value:.2f}")
    NELOW_max_frequency.set(f"NELOW 최대 주파수: {NELOW_max_frequency_value:.2f}")

    # Extract the label (last character before ".wav" in the filename)
    file_label = os.path.basename(file_path)[-5].upper()
    label_text_map = {'L': '누수', 'M': '미터', 'N': '비 누수음'}
    file_label_text = label_text_map.get(file_label, '알 수 없음')  # Default to '알 수 없음' if label is invalid

    file_label_var.set(f"파일 레이블: {file_label_text}")

    results_original, probabilities_original = predict(models, processed_wav_original)

    # Map numeric results to text labels
    labels = ["누수", "미터", "비 누수음"]
    labeled_results_original = []

    for result_original, probability_original in zip(results_original, probabilities_original):
        label = labels[int(result_original)]
        prob_text = f"{label} - ({probability_original[0][0] * 100:.2f}% Leak, {probability_original[0][1] * 100:.2f}% Meter, {probability_original[0][2] * 100:.2f}% No leak)"
        labeled_results_original.append(prob_text)

    color_map = {"누수": "red", "미터": "orange", "비 누수음": "green"}

    for i, result_label_original in enumerate(labeled_results_original):
        result_labels_original[i].config(text=f"{model_names[i]}: {result_label_original}", fg=color_map[labels[int(results_original[i])]])

    display_image(file_path)

    # Update the open file button to the selected file
    open_file_btn.config(command=lambda: open_wav_file(file_path), state='active')


def display_image(file_path):
    global canvas
    if canvas:
        canvas.get_tk_widget().destroy()  # Destroy the existing canvas if it exists

    raw_y, raw_sr = librosa.load(file_path, sr=None)
    processed_mfcc_original, _, processed_y_original, processed_sr_original = preprocess_wav(file_path)

    NELOW_fft_data, _, _, _ = get_NELOW_values(file_path)
    NELOW_fft_data = np.array(NELOW_fft_data)

    # Set the font properties to support Korean characters
    font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    korean_font = None
    for path in font_path:
        if 'malgun' in path.lower():  # Check for a font like Malgun Gothic
            korean_font = fm.FontProperties(fname=path)
            break
    if korean_font is None:
        print("Korean font not found. Default font will be used.")

    # Create a wider figure with custom layout for both waveform and spectrum
    fig = Figure(figsize=(18, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 2])  # Set first column (waveform) to be twice as wide

    # Plot the waveform
    ax1 = fig.add_subplot(gs[0])
    librosa.display.waveshow(raw_y, sr=raw_sr, ax=ax1)
    ax1.set_ylim(-1, 1)  # Set y-axis limits to min and max of waveform
    ax1.set_title('파형', fontproperties=korean_font)  # Title in Korean
    ax1.set_xlabel('시간(초)', fontproperties=korean_font)  # X-axis label in Korean
    ax1.set_ylabel('진폭', fontproperties=korean_font)  # Y-axis label in Korean

    # Plot MFCC
    ax2 = fig.add_subplot(gs[1])
    # cax = ax2.matshow(processed_mfcc_original, cmap='PiYG')  # Use a perceptually uniform colormap
    cax = ax2.matshow(processed_mfcc_original)  # Use a perceptually uniform colormap
    fig.colorbar(cax, ax=ax2)
    ax2.set_title('MFCC', fontproperties=korean_font)

    # Plot NELOW Spectrum
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(NELOW_fft_data, color='purple')
    ax3.set_title('NELOW 스펙트럼', fontproperties=korean_font)
    ax3.set_xlabel('주파수(Hz)', fontproperties=korean_font)  # X-axis label in Korean
    ax3.set_ylabel('진폭', fontproperties=korean_font)  # Y-axis label in Korean
    ax3.set_ylim(0, max(NELOW_fft_data) * 1.3)  # Set the y-axis limit
    ax3.legend()

    # Adjust layout and embed the plot in Tkinter
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plt.close(fig)  # Close the figure to free up memory


def rename_file(new_label):
    global current_file_index, wav_files
    if current_file_index is not None:
        current_path = wav_files[current_file_index ]
        directory, filename = os.path.split(current_path)
        # Construct the new filename by changing the [-5] character to the appropriate label
        new_filename = filename[:-5] + new_label + filename[-4:]
        new_path = os.path.join(directory, new_filename)
        os.rename(current_path, new_path)
        # Update the list and current index to reflect the new file name
        wav_files[current_file_index] = new_path
        load_file(new_path)  # Reload the file to reflect the name change in the GUI

def set_no_leak():
    rename_file('N')

def set_meter():
    rename_file('M')

def set_leak():
    rename_file('L')

def export_results_to_csv(wav_files, models):
    if not wav_files:
        print("No WAV files to process.")
        return

    # Get the directory where the WAV files are stored
    directory = os.path.dirname(wav_files[0])

    # Get the current folder name
    folder_name = os.path.basename(directory)

    # Prepare the CSV file path (same directory as WAV files)
    csv_file_path = os.path.join(directory, f"analysis_results_{folder_name}.csv")

    # Prepare the CSV file header
    header = ["Index", "File Name", "Strength Value", "Max Frequency", "Label"] + [f"AI Result Model {i+1} (L/M/N)" for i in range(len(models))] + [f"Leak Probability Model {i+1}" for i in range(len(models))]

    # Prepare a list to hold the data rows
    data_rows = []

    # Process each WAV file
    for file_path in wav_files:
        # Extract the NELOW values and AI prediction results
        _, _, NELOW_strength_value, NELOW_max_frequency_value = get_NELOW_values(file_path)
        _, processed_wav, _, _ = preprocess_wav(file_path)

        # Extract the label (last character before ".wav" in the filename)
        label = os.path.basename(file_path)[-5].upper()

        # Prepare lists to hold results for all models
        ai_results = []
        leak_probabilities = []

        # Predict using the models
        for i, model in enumerate(models):
            results, probabilities = predict([model], processed_wav)

            # Get prediction and leak probability for the current model
            predicted_label = results[0][0]  # Taking the first result from the model's predictions
            leak_probability = probabilities[0][0][0]  # Leak probability from the current model

            # Map AI result to the label
            ai_result = ['L', 'M', 'N'][predicted_label]  # 'L' for Leak, 'M' for Meter, 'N' for No leak

            # Append the result and probability for the current model
            ai_results.append(ai_result)
            leak_probabilities.append(leak_probability)

        # Store the data row for this file
        data_rows.append([os.path.basename(file_path), NELOW_strength_value, NELOW_max_frequency_value, label] + ai_results + leak_probabilities)

    # Sort the rows by Strength Value in descending order
    data_rows.sort(key=lambda x: x[1], reverse=True)

    # Add an index to each row
    for idx, row in enumerate(data_rows):
        row.insert(0, idx + 1)  # Add index as the first column (starting from 1)

    # Write the data to CSV
    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header
            writer.writerows(data_rows)  # Write the data rows
        print(f"Results successfully exported to {csv_file_path}")
    except Exception as e:
        print(f"Error exporting results to CSV: {e}")

# Optional: Add a button to perform an action with the number input
def use_number_input():
    try:
        entered_value = float(number_input.get())  # Convert input to a float
        print(f"입력된 숫자: {entered_value}")

        # Loop through all WAV files in the folder
        for file in wav_files:
            file_path = os.path.join(os.path.dirname(file), file)
            # Get the NELOW strength value for the current file
            _, _, NELOW_strength_value, _ = get_NELOW_values(file_path)

            # If the strength value is below the entered threshold, rename the file
            if NELOW_strength_value < entered_value:
                # Extract the directory and filename
                directory, filename = os.path.split(file_path)
                # Construct the new filename by changing the label to "N"
                new_filename = filename[:-5] + 'N' + filename[-4:]
                new_path = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(file_path, new_path)

                # Update the wav_files list with the new file name
                wav_files[wav_files.index(file)] = new_path
                print(f"파일 {filename}이(가) {new_filename}로 이름이 변경되었습니다.")

        # Reload the file display (you can add more logic to refresh the UI)
        load_file(wav_files[current_file_index])

    except ValueError:
        print("유효하지 않은 숫자입니다.")


# Load models
models = load_models()
model_names = ["AI 모델 V3.0.0", "AI 모델 V3.0.1"]


# Setup GUI
root = tk.Tk()
root.title("AI 모델 WAV 분석")

filename_var = tk.StringVar()
NELOW_strength = tk.StringVar()
NELOW_max_frequency = tk.StringVar()
file_label_var = tk.StringVar()

large_font = font.Font(size=14, weight="bold")  # Bold font
normal_font = font.Font(size=14, weight="normal")
small_font = font.Font(size=12, weight="normal")


# Button for selecting the file
select_button = tk.Button(root, text="WAV 파일 선택", command=select_file, font=large_font)
select_button.pack(pady=10)

# Button for opening the file, initially disabled
open_file_btn = tk.Button(root, text="WAV 파일 열기", state='disabled', font=large_font)
open_file_btn.pack(pady=10)

# Add an "Export to CSV" button
export_button = tk.Button(root, text="결과 CSV로 내보내기", font=large_font, command=lambda: export_results_to_csv(wav_files, models), bg='lightblue')
export_button.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

# show_mfcc_button = tk.Button(root, text="Show MFCC and Spectrum images", command=lambda: None, state='disabled', font=large_font)
# show_mfcc_button.pack(pady=10)

# Adding buttons to go to the first and last file
first_file_button = tk.Button(root, text="<<<", command=go_to_first_file, font=large_font)
first_file_button.pack(side=tk.LEFT, padx=10, pady=10)

last_file_button = tk.Button(root, text=">>>", command=go_to_last_file, font=large_font)
last_file_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Buttons for larger skips
next_10_button = tk.Button(root, text=">>", command=lambda: next_file(10), font=large_font)
next_10_button.pack(side=tk.RIGHT, padx=10, pady=10)

prev_10_button = tk.Button(root, text="<<", command=lambda: previous_file(10), font=large_font)
prev_10_button.pack(side=tk.LEFT, padx=10, pady=10)

# Buttons for 1-file skips
next_button = tk.Button(root, text=">", command=lambda: next_file(1), font=large_font)
next_button.pack(side=tk.RIGHT, padx=10, pady=10)

prev_button = tk.Button(root, text="<", command=lambda: previous_file(1), font=large_font)
prev_button.pack(side=tk.LEFT, padx=10, pady=10)



# Buttons for classification actions
# Create a frame to hold the classification buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

# Place buttons within the frame, with specific colors for each
no_leak_button = tk.Button(button_frame, text="비 누수음", font=large_font, command=set_no_leak, bg='green')
no_leak_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

meter_button = tk.Button(button_frame, text="미터", font=large_font, command=set_meter, bg='orange')
meter_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

leak_button = tk.Button(button_frame, text="누수", font=large_font, command=set_leak, bg='red')
leak_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

filename_label = tk.Label(root, textvariable=filename_var, font=small_font)
filename_label.pack()

# Add the number input widgets and button here
number_label = tk.Label(root, text="숫자 입력:", font=normal_font)
number_label.pack(pady=5)

number_input = tk.Entry(root, font=large_font)
number_input.pack(pady=5)

number_button = tk.Button(root, text="사용하기", font=large_font, command=use_number_input)
number_button.pack(pady=10)

results_header = tk.Label(root, text="결과", font=large_font)
results_header.pack()

NELOW_strength_label = tk.Label(root, textvariable=NELOW_strength, font=small_font)
NELOW_strength_label.pack()

NELOW_frequency_max_label = tk.Label(root, textvariable=NELOW_max_frequency, font=small_font)
NELOW_frequency_max_label.pack()

# Create the label with the determined background color
file_label = tk.Label(root, textvariable=file_label_var, font=small_font)
file_label.pack()


result_labels_original = [tk.Label(root, text="", font=normal_font, justify=tk.LEFT) for _ in model_names]
for label in result_labels_original:
    label.pack()
result_labels = [tk.Label(root, text="", font=normal_font, justify=tk.LEFT) for _ in model_names]
for label in result_labels:
    label.pack()

root.mainloop()