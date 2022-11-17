import tensorflow as tf

model = tf.keras.models.load_model('/Users/macbookpro/Desktop/445CS/CNN_train/mnist_cnn_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('/Users/macbookpro/Desktop/445CS/CNN_train/cnn_mnist_model.tflite', 'wb') as f:
  f.write(tflite_model)