import numpy as np
import pandas as pd
import os
import h5py
import matplotlib
import math
from matplotlib import pyplot as plt
import pickle
# %matplotlib inline
# matplotlib.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Keras
# import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

# Tensorflow

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Custom imports
from physionet_processing import (fetch_h5data, spectrogram, 
                                  special_parameters, transformed_stats)

from physionet_generator import DataGenerator

FILENAME = "13"

print('Tensorflow version:', tf.__version__)
# print('Keras version:', keras.__version__)

#Open hdf5 file, load the labels and define training/validation splits

# Data folder and hdf5 dataset file
data_root = os.path.normpath('.')
#data_root = os.path.normpath('/media/sf_vbshare/physionet_data/')
#data_root = os.path.normpath('/home/ubuntu/projects/csproject')
# hd_file = os.path.join(data_root, 'physio.h5')
hd_file = "/scratch/thurasx/ecg_project_2/cnn_ecg_keras/physio.h5"
label_file = "/scratch/thurasx/ecg_project_2/cnn_ecg_keras/REFERENCE-v3.csv"

# # mac 
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
labels = dict(zip(label_df.name, label_df.encoded))

# print(label_df)
encoded = label_df['encoded'].to_numpy()
a, b = np.unique(encoded, return_counts=True)
print(a)
print(b)
"""broad casting"""
l0 = label_df['name'].to_numpy()[encoded == 0]
l1 = label_df['name'].to_numpy()[encoded == 1]
l2 = label_df['name'].to_numpy()[encoded == 2]
l3 = label_df['name'].to_numpy()[encoded == 3]

print(len(l0),len(l1),len(l2),len(l3))
print(type(l2))

""" Train data """
train_data = np.array([])
random_count_train = len(l3)
train_data = np.concatenate((train_data,np.random.choice(l0, size=random_count_train, replace=False)), axis=0)
train_data = np.concatenate((train_data,np.random.choice(l1, size=random_count_train, replace=False)), axis=0)
train_data = np.concatenate((train_data,np.random.choice(l2, size=random_count_train, replace=False)), axis=0)
train_data = np.concatenate((train_data,l3), axis=0)
assert(len(train_data) == random_count_train * 4)
print(len(train_data))


""" Test data """
test_data = np.array([])
random_count_test = math.floor(len(l3) * 0.5)
print(random_count_test)
test_data = np.concatenate((test_data,np.random.choice(l0, size=random_count_test, replace=False)), axis=0)
test_data = np.concatenate((test_data,np.random.choice(l1, size=random_count_test, replace=False)), axis=0)
test_data = np.concatenate((test_data,np.random.choice(l2, size=random_count_test, replace=False)), axis=0)
test_data = np.concatenate((test_data,np.random.choice(l3, size=random_count_test, replace=False)), axis=0)
assert(len(test_data) == random_count_test * 4)
print(len(test_data), random_count_test * 4)

# Split the IDs in training and validation set
test_split = 0.30
idx = np.arange(label_df.shape[0])
id_train, id_val, _, _ = train_test_split(train_data, train_data, 
                                        test_size = test_split,
                                        shuffle = True,
                                        random_state = 123)

partition = { 'train' : id_train,
             'validation': id_val,
             'test' : test_data}

with open(f"/scratch/thurasx/ecg_project_2/cnn_ecg_keras/cnn_ecg_testing/cnn_small_{FILENAME}_testlabel.pcl", "wb") as f:
    pickle.dump(partition["test"], f)

#set up batch generator
# Parameters needed for the batch generator

# Maximum sequence length
max_length = 18286

# Output dimensions
sequence_length = max_length
spectrogram_nperseg = 64 # Spectrogram window
spectrogram_noverlap = 32 # Spectrogram overlap
n_classes = len(label_df.label.unique())
batch_size = 15

# calculate image dimensions
data = fetch_h5data(h5file, [0], sequence_length)
_, _, Sxx = spectrogram(data, nperseg = spectrogram_nperseg, noverlap = spectrogram_noverlap)
dim = Sxx[0].shape




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

#model define

# Convolutional blocks
def conv2d_block(model, depth, layer_filters, filters_growth, 
                 strides_start, strides_end, input_shape, first_layer = False):
    
    ''' Convolutional block. 
    depth: number of convolutional layers in the block (4)
    filters: 2D kernel size (32)
    filters_growth: kernel size increase at the end of block (32)
    first_layer: provide input_shape for first layer'''
    
    # Fixed parameters for convolution
    conv_parms = {'kernel_size': (3, 3),
                  'padding': 'same',
                  'dilation_rate': (1, 1),
                  'activation': None,
                  'data_format': 'channels_last',
                  'kernel_initializer': 'glorot_normal'}

    for l in range(depth):

        if first_layer:
            
            # First layer needs an input_shape 
            model.add(layers.Conv2D(filters = layer_filters,
                                    strides = strides_start,
                                    input_shape = input_shape, **conv_parms))
            first_layer = False
        
        else:
            # All other layers will not need an input_shape parameter
            if l == depth - 1:
                # Last layer in each block is different: adding filters and using stride 2
                layer_filters += filters_growth
                model.add(layers.Conv2D(filters = layer_filters,
                                        strides = strides_end, **conv_parms))
            else:
                model.add(layers.Conv2D(filters = layer_filters,
                                        strides = strides_start, **conv_parms))
        
        # Continue with batch normalization and activation for all layers in the block
        model.add(layers.BatchNormalization(center = True, scale = True))
        model.add(layers.Activation('relu'))
    
    return model

def MeanOverTime():
    lam_layer = layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (1, s[2]))
    return lam_layer


# Define the model
# Define the model
# Model parameters
filters_start = 32 # Number of convolutional filters
layer_filters = filters_start # Start with these filters
filters_growth = 32 # Filter increase after each convBlock
strides_start = (1, 1) # Strides at the beginning of each convBlock
strides_end = (2, 2) # Strides at the end of each convBlock
depth = 2 # Number of convolutional layers in each convBlock
n_blocks = 1 # Number of ConBlocks
n_channels = 1 # Number of color channgels
input_shape = (*dim, n_channels) # input shape for first layer

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

print("Data Input Shape : ", input_shape)
model = Sequential()

for block in range(n_blocks):

    # Provide input only for the first layer
    if block == 0:
        provide_input = True
    else:
        provide_input = False
    
    model = conv2d_block(model, depth,
                         layer_filters,
                         filters_growth,
                         strides_start, strides_end,
                         input_shape,
                         first_layer = provide_input)
    
    # Increase the number of filters after each block
    layer_filters += filters_growth



# Remove the frequency dimension, so that the output can feed into LSTM
# Reshape to (batch, time steps, filters)
model.add(layers.Reshape((-1, 1088)))
# model.add(layers.core.Masking(mask_value = 0.00))
model.add(MeanOverTime())
model.add(layers.Dense(4, activation='sigmoid', kernel_regularizer = regularizers.l2(0.1)))

"""
def custom_cross_entropy_loss(y_true, y_pred):
    # y_true = tf.reshape(y_true, [-1])
    try:
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        loss = - y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        loss = tf.where(tf.equal(y_true, 3), loss * 10.0, loss)
        loss = tf.reduce_mean(loss)
        return loss
    except:
        return 0.5

def custom_categorical_crossentropy(y_true, y_pred):
    weights = tf.constant([1.0, 1.0, 2.0, 1.0]) # set higher weight for class 3
    weights = tf.cast(weights, dtype=tf.float32)
    
    # convert y_true to one-hot encoded tensor
    y_true_one_hot = tf.one_hot(tf.argmax(y_true, axis=-1), depth=4)
    
    # compute categorical cross-entropy loss
    loss = tf.losses.categorical_crossentropy(y_true_one_hot, y_pred, weights=weights)
    
    return loss
"""
def custom_categorical_crossentropy(y_true, y_pred):
    # Define the weight for class 3
    weight_class_3 = 50.0

    # Calculate the normal categorical crossentropy loss
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Extract the class labels from y_true and y_pred
    true_class = tf.argmax(y_true, axis=-1)
    pred_class = tf.argmax(y_pred, axis=-1)

    # Calculate the loss mask to penalize predictions for class 3
    loss_mask = tf.cast(tf.equal(true_class, 2), dtype=tf.float32) * weight_class_3

    # Return the weighted loss
    return loss + loss_mask

# Compile the model and run a batch through the network
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=METRICS)


model.summary()
h = model.fit(train_generator,
            steps_per_epoch = 50,
            epochs = 500,
            
            validation_data = val_generator,
            validation_steps = 22, verbose=1)



model.save(f'/scratch/thurasx/ecg_project_2/cnn_ecg_keras/cnn_ecg_keras_tflites/keras_ecg_cnn_small_{FILENAME}.h5')
df = pd.DataFrame(h.history)
df.head()
df.to_csv(f'/scratch/thurasx/ecg_project_2/cnn_ecg_keras/history_small_{FILENAME}.csv')
batch_size = 1
input_shape = model.inputs[0].shape.as_list()
input_shape[0] = batch_size
func = tf.function(model).get_concrete_function(
    tf.TensorSpec(input_shape, model.inputs[0].dtype))
converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
# converter = tf.lite.TFLiteConverter.from_keras_model(model) 
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] 
tflite_model = converter.convert()
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

with open(f'/scratch/thurasx/ecg_project_2/cnn_ecg_keras/cnn_ecg_keras_tflites/keras_ecg_cnn_small_{FILENAME}.tflite', 'wb+') as f:
    f.write(tflite_model)
