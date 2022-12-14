'''***********************************************
*
*       project: physioNet
*       created: 18.04.2017
*       purpose: convolutional neural network (CNN) class
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

import tensorflow as tf

from definitions import *
from network.network_upgrade import Network
import utils.nn_layers_upgrade as nn
import utils.tf_helper_upgrade as tfh

'''***********************************************
*           Classes
***********************************************'''

class CNN(Network):
    
    def __init__(self):
        Network.__init__(self)
        self.n_conv_blocks      = None
        self.kernel_size        = None
        self.n_channels_first   = None
        self.dilation_rates     = None
        self.growth_block_end   = None
        self.strides_block_end  = None
        self.max_pooling        = None


    def create_model(self, data):

        length = tfh.get_length(data)

        with tf.compat.v1.name_scope('preshape'):
            conv_data = tf.expand_dims(data, 3)
            print(tfh.get_static_shape(conv_data))

        with tf.compat.v1.name_scope('convolutions'):
            for i in range(self.n_conv_blocks):
                with tf.compat.v1.name_scope('conv_block' + str(i)):
                    n_channels = None
                    if i == 0:
                        n_channels = self.n_channels_first
                    [conv_data, length] = nn.conv2d_block(
                            inputs=conv_data,
                            length=length,
                            kernel_size=self.kernel_size,
                            n_channels=n_channels,
                            dilation_rates=self.dilation_rates,
                            growth=self.growth_block_end,
                            strides_end=self.strides_block_end,
                            max_pooling=self.max_pooling,
                            is_training=self.is_training,
                            drop_rate=self.drop_rate)
                print(tfh.get_static_shape(conv_data))
                
        with tf.compat.v1.name_scope('postshape'):
            [_, t_s, f_s, c_s] = tfh.get_static_shape(conv_data)
            feature_seq = tf.reshape(conv_data, [-1, t_s, f_s * c_s])
            print(tfh.get_static_shape(feature_seq))

        with tf.compat.v1.name_scope('average_features'):
            features = nn.average_features(feature_seq, length)

        with tf.compat.v1.name_scope('linear_layer'):
            pred = tf.compat.v1.layers.dense(inputs=features, units=self.n_classes)

        return pred


    def get_modelParameters(self):
        dict = {
            'n_conv_blocks'     : self.n_conv_blocks,
            'kernel_size'       : self.kernel_size,
            'n_channels_first'  : self.n_channels_first,
            'dilation_rates'    : self.dilation_rates,
            'growth_block_end'  : self.growth_block_end,
            'strides_block_end' : self.strides_block_end,
            'max_pooling'       : self.max_pooling
        }
        return dict


    def set_modelParameters(self, dict):
        self.n_conv_blocks      = dict['n_conv_blocks']
        self.kernel_size        = dict['kernel_size']
        self.n_channels_first   = dict['n_channels_first']
        self.dilation_rates     = dict['dilation_rates']
        self.growth_block_end   = dict['growth_block_end']
        self.strides_block_end  = dict['strides_block_end']
        self.max_pooling        = dict['max_pooling']

