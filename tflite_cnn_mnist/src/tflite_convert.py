import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
def MeanOverTime():
    lam_layer = layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (1, s[2]))
    return lam_layer

model = tf.keras.models.load_model('C:/Users/sanda/Documents/esp_dev_files/tensor_test_3_cnn/src/models/cnn_ecg_keras_small_4.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('C:/Users/sanda/Documents/esp_dev_files/tensor_test_3_cnn/src/models/cnn_ecg_keras_small_4.tflite', 'wb') as f:
  f.write(tflite_model)

# !xxd -i cnn_ecg_keras_small_4.tflite > cnn_ecg_keras_small_4.cc