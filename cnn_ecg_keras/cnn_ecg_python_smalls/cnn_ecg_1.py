import numpy as np
import pandas as pd
import os
import h5py
import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline
# matplotlib.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Keras
import keras
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import backend as K
from keras import regularizers

# Tensorflow
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Custom imports
from physionet_processing import (fetch_h5data, spectrogram, 
                                  special_parameters, transformed_stats)

from physionet_generator import DataGenerator

print('Tensorflow version:', tf.__version__)
print('Keras version:', keras.__version__)

#Open hdf5 file, load the labels and define training/validation splits

# Data folder and hdf5 dataset file
data_root = os.path.normpath('.')
#data_root = os.path.normpath('/media/sf_vbshare/physionet_data/')
#data_root = os.path.normpath('/home/ubuntu/projects/csproject')
# hd_file = os.path.join(data_root, 'physio.h5')
hd_file = "/scratch/thurasx/ecg_project_2/cnn_ecg_keras/physio.h5"
label_file = "/scratch/thurasx/ecg_project_2/cnn_ecg_keras/REFERENCE-v3.csv"

# mac 
# hd_file = "/Users/macbookpro/Documents/physio.h5"
# label_file = "/Users/macbookpro/Documents/ecg_project_2/cnn_ecg_keras/REFERENCE-v3.csv"


# Open hdf5 file
h5file =  h5py.File(hd_file, 'r')

# Get a list of dataset names 
dataset_list = list(h5file.keys())

# Load the labels
label_df = pd.read_csv(label_file, header = None, names = ['name', 'label'])
# Filter the labels that are in the small demo set
label_df = label_df[label_df['name'].isin(dataset_list)]

# Encode labels to integer numbers
label_set = list(sorted(label_df.label.unique()))
encoder = LabelEncoder().fit(label_set)
label_set_codings = encoder.transform(label_set)
label_df = label_df.assign(encoded = encoder.transform(label_df.label))

# print('Unique labels:', encoder.inverse_transform(label_set_codings))
# print('Unique codings:', label_set_codings)
# print('Dataset labels:\n', label_df.iloc[100:110,])

# Split the IDs in training and validation set
test_split = 0.33
idx = np.arange(label_df.shape[0])
id_train, id_val, _, _ = train_test_split(idx, idx, 
                                         test_size = test_split,
                                         shuffle = True,
                                         random_state = 123)

# Store the ids and labels in dictionaries
partition = {'train': list(label_df.iloc[id_train,].name), 
             'validation': list(label_df.iloc[id_val,].name)}

labels = dict(zip(label_df.name, label_df.encoded))

#set up batch generator
# Parameters needed for the batch generator

# Maximum sequence length
max_length = 18286

# Output dimensions
sequence_length = max_length
spectrogram_nperseg = 64 # Spectrogram window
spectrogram_noverlap = 32 # Spectrogram overlap
n_classes = len(label_df.label.unique())
batch_size = 32

# calculate image dimensions
data = fetch_h5data(h5file, [0], sequence_length)
_, _, Sxx = spectrogram(data, nperseg = spectrogram_nperseg, noverlap = spectrogram_noverlap)
dim = Sxx[0].shape
# print(dim)
# print(Sxx.shape)
# print('Maximum sequence length:', max_length)



params = {'batch_size': batch_size,
          'dim': dim,
          'nperseg': spectrogram_nperseg,
          'noverlap': spectrogram_noverlap,
          'n_channels': 1,
          'sequence_length': sequence_length,
          'n_classes': n_classes,
          'shuffle': True}

train_generator = DataGenerator(h5file, partition['train'], labels, augment = True, **params)
val_generator = DataGenerator(h5file, partition['validation'], labels, augment = False, **params)

# for i, batch in enumerate(train_generator):
#     if i == 1:
#         break

# X = batch[0]
# y = batch[1]

# print('X shape:', X.shape)
# print('y shape:', y.shape)
# print('X type:', np.dtype(X[0,0,0,0]))


def MeanOverTime():
    lam_layer = layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (1, s[2]))
    return lam_layer

#model define
#define model
model = Sequential()
model.add(layers.Conv2D(64, (5,5), input_shape=(*dim, 1)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D(pool_size=(3,3)))

model.add(layers.Conv2D(64, (5,5)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D(pool_size=(3,3)))

model.add(layers.core.Masking(mask_value = 0.0))
model.add(MeanOverTime())

model.add(layers.Flatten())
model.add(layers.Dense(4, activation='relu', kernel_regularizer = regularizers.l2(0.1)))


model.summary()

# Compile the model and run a batch through the network
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['acc'])


h = model.fit_generator(generator = train_generator,
                              steps_per_epoch = 50,
                              epochs = 1,
                              validation_data = val_generator,
                              validation_steps = 50, verbose=1)



df = pd.DataFrame(h.history)
df.head()
df.to_csv('/scratch/thurasx/ecg_project_2/cnn_ecg_keras/history_small_1.csv')

# model.save('/scratch/thurasx/ecg_project_2/cnn_ecg_keras/cnn_ecg_keras_small_3.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('/scratch/thurasx/ecg_project_2/cnn_ecg_keras/cnn_ecg_keras_tflites/keras_ecg_cnn_small_1.tflite', 'wb+') as f:
    f.write(tflite_model)

#tsp -m python /scratch/thurasx/ecg_project_2/cnn_ecg_keras/cnn_ecg_python_small.py
