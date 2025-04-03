import tensorflow as tf
from keras import backend as K
from keras.utils import get_custom_objects

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

# Register the custom metric
get_custom_objects()['f1_m'] = f1_m
get_custom_objects()['precision_m'] = precision_m
get_custom_objects()['recall_m'] = recall_m

# Load the Keras model
model = tf.keras.models.load_model('NELOW_AI_model/NELOW_GL_model_test_V3.h5')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('NELOW_AI_model/NELOW_GL_TFLite_model_test_V3.tflite', 'wb') as f:
    f.write(tflite_model)






