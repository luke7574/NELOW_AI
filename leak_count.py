import os
import numpy as np
import pandas as pd
import tensorflow as tf

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

    # label = tf.keras.utils.to_categorical(label, num_classes=3)
    npy_table = np.array(npy_table) # input
    label = np.array(label)         # output
    filename = np.array(filename)

    npy_table = npy_table.reshape(-1, 20, 13, 1)
    # label = label.reshape(-1, 3)
    filename = filename.reshape(-1, 1)

    return npy_table, label, filename

Numpy_files_path_training = 'Testing/Numpy_files/'  # numpy 데이터 경로 설정  C:\Users\gram\AI\NELOW\NELOW_AI\Testing\Numpy_files
q,w,e=load_npy(Numpy_files_path_training)

sum_0 = 0
sum_1 = 0
sum_2 = 0

for i in range(len(w)):
    a = w[i]
    if a == 0:
        sum_0 += 1
    elif a == 1:
        sum_1 += 1
    elif a == 2:
        sum_2 += 1

print(f"Leak: {sum_0}, Meter: {sum_1}, No leak: {sum_2}")