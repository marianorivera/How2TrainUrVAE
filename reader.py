"""
reader.py: Residual Encoder and Decoder, with attention in the encoder
"""

__author__      = "Mariano Rivera"
__copyright__   = "CC BY-NC 4.0"



from typing import Tuple, List
import numpy as np

import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras import layers

#  Self PixelAttention Layer
from pixel_attention import PixelAttention2D_v1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ResEncoder and ResDecodes Version 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class ResBlockId(tf.Module):
    def __init__(self, filters, kernel_sz=(3,3), activation='relu', strides=1, padding="same"):
        
        super().__init__()
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=kernel_sz, activation=activation, strides=strides, padding=padding)
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=kernel_sz, activation=None,       strides=strides, padding=padding)
        self.add   = layers.Add()

    def __call__(self, x):
        y   = self.conv1(x)
        y   = self.conv2(y)
        out = self.add([y, x])
        return out    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class ResBlockConv(tf.Module):
    def __init__(self, filters, kernel_sz=(3,3), activation='relu', strides=1, padding="same"):
        
        super().__init__()
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=kernel_sz, activation=activation, strides=strides,   padding=padding)
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=kernel_sz, activation=None,       strides=2*strides, padding=padding)
        self.conv3 = layers.Conv2D(filters=filters, kernel_size=kernel_sz, activation=None,       strides=2*strides, padding=padding)
        self.bnorm = layers.BatchNormalization()
        self.add   = layers.Add()
        
    def __call__(self, x):
        y1  = self.conv1(x)
        y1  = self.bnorm(y1)
        y1  = self.conv2(y1)
        y2  = self.conv3(x)
        out = self.add([y1, y2])
        return out

    
def ConvMix(input_dim=(32,32,1), filters=None, name='convmix'):
    '''
    Residual convolutional Block without size reduction
    '''
    # blocks
    conv1 = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', strides=1, padding='same')
    conv2 = layers.Conv2D(filters=filters, kernel_size=3, activation=None,   strides=1, padding='same')
    conv3 = layers.Conv2D(filters=filters, kernel_size=3, activation=None,   strides=1, padding='same')
    bnorm = layers.BatchNormalization()
    add   = layers.Add()
    
    # model
    inputs  = layers.Input(shape=input_dim)
    y1  = bnorm(conv1(inputs))
    y1  = conv2(y1)
    y2  = conv3(inputs)
    outputs = add([y1, y2])
    return Model(inputs, outputs, name=name)
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class ResBlockConvT(tf.Module):
    def __init__(self, filters, kernel_sz=(3,3), activation='relu', strides=1, padding="same"):
        
        super().__init__()
        self.conv1  = layers.Conv2D(filters=filters, kernel_size=kernel_sz, activation=activation,    strides=strides,   padding=padding)
        self.bnorm  = layers.BatchNormalization()
        self.conv2T = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_sz, activation=None, strides=2*strides, padding=padding)
        self.conv3T = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_sz, activation=None, strides=2*strides, padding=padding)
        self.add    = layers.Add()
        
    def __call__(self, x):
        y1  = self.conv1(x)
        y1  = self.bnorm(y1)
        y1  = self.conv2T(y1)
        y2  = self.conv3T(x)
        out = self.add([y1, y2])
        return out
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ResEncoder and ResDecodes Version 2
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ConvTriBlockV2(tf.Module):
    '''
    Three Convolutions in cascade
    '''
    def __init__(self, filters, kernel_sz=(3,3), activation='relu', strides=[1,1,1], padding="same"):
        
        super().__init__()
        self.bnorm1 = layers.BatchNormalization()
        self.bnorm2 = layers.BatchNormalization()
        self.bnorm3 = layers.BatchNormalization()
        self.relu   = layers.Activation('relu')
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=3, activation=None, strides=strides[0], padding=padding)
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=3, activation=None, strides=strides[1], padding=padding)
        self.conv3 = layers.Conv2D(filters=filters, kernel_size=1, activation=None, strides=strides[2], padding=padding)
        
    def __call__(self, x):
        y = self.relu(self.bnorm1(x))
        y = self.conv1(y)
        y = self.relu(self.bnorm2(y))
        y = self.conv2(y)
        y = self.relu(self.bnorm3(y))
        out = self.conv3(y)
        return out    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class ResBlockIdV2(tf.Module):
    def __init__(self, filters, kernel_sz=(3,3), activation='relu', strides=[1,1,1], padding="same"):
        
        super().__init__()
        self.convTri = ConvTriBlockV2(filters, kernel_sz=kernel_sz, activation=activation, strides=strides, padding=padding)
        self.add   = layers.Add()

    def __call__(self, x):
        y   = self.convTri(x)
        out = self.add([y, x])
        return out

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class ResBlockConvV2(tf.Module):
    def __init__(self, filters, kernel_sz=(3,3), activation='relu', strides=[1,1,2], padding="same"):
        
        super().__init__()
        self.convTri = ConvTriBlockV2(filters, kernel_sz=kernel_sz, activation=activation, strides=strides, padding=padding)
        self.bnorm4  = layers.BatchNormalization()
        self.relu    = layers.Activation('relu')
        self.conv4   = layers.Conv2D(filters=filters, kernel_size=1, activation=None, strides=strides[2], padding=padding)
        self.add     = layers.Add()
        
    def __call__(self, x):        
        y   = self.convTri(x)
        
        yy  = self.relu(self.bnorm4(x))
        yy  = self.conv4(yy)
        
        out = self.add([y, yy])
        return out
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class ResBlockConvTV2(tf.Module):
    def __init__(self, filters, kernel_sz=(3,3), activation='relu', strides=1, padding="same"):
        
        super().__init__()
        self.conv1  = layers.Conv2D(         filters=filters, kernel_size=1, activation=None, strides=strides,   padding=padding)
        self.conv2T = layers.Conv2DTranspose(filters=filters, kernel_size=3, activation=None, strides=2*strides, padding=padding)
        self.conv3T = layers.Conv2DTranspose(filters=filters, kernel_size=3, activation=None, strides=2*strides, padding=padding)
        self.relu   = layers.Activation('relu')
        self.bnorm1 = layers.BatchNormalization()
        self.bnorm2 = layers.BatchNormalization()
        self.bnorm3 = layers.BatchNormalization()
        self.add    = layers.Add()
        
    def __call__(self, x):
        y = self.relu(self.bnorm1(x))
        y = self.conv1(y)
        y = self.relu(self.bnorm2(y))
        y = self.conv2T(y)
        
        yy = self.relu(self.bnorm3(x))
        yy = self.conv3T(yy)
        
        #print(x.shape,y.shape,yy.shape)
        out = self.add([y, yy])
        return out

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Residual Encoder and Decoder
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def ResEncoder(input_dim=(32,32,1), latent_dim=32, num_hidden_residuals_blocks=2, listNumFilters=[64,64,64,64], attention=True, name='encoder'):
    
    encoder_inputs = layers.Input(shape=input_dim)
    x = layers.LayerNormalization(axis=(1,2))(encoder_inputs)            
    x = layers.Conv2D(filters= listNumFilters[0], kernel_size=3, padding='same')(x)
    for idx,nfilters in enumerate(listNumFilters[:-1]):
        for i in range(num_hidden_residuals_blocks):
            x = ResBlockIdV2(nfilters)(x)
        x = ResBlockConvV2(listNumFilters[idx+1])(x)  

    if attention:
        #x = PixelAttention2D_v1(latent_dim)(x)
        x = Multihead_PixelAttention_v1(latent_dim)(x,x)  # default: 3 heads
    else:
        x = layers.Conv2D(latent_dim, 1, padding="same")(x)
        
    return Model(encoder_inputs, x, name=name)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# if V1  num_hidden_residuals_blocks=2,
def ResDecoder(input_dim=(5,5,32), start_conv_dim=None, output_channels=1, latent_dim=32, num_hidden_residuals_blocks=1, listNumFilters=[64,64,64,64], name='decoder'):
    
    decoder_inputs = layers.Input(shape=[input_dim])  
    x = layers.Dense(np.prod(np.array(start_conv_dim)))(decoder_inputs)
    x = layers.Reshape(start_conv_dim)(x)
        
    x = layers.Conv2D(filters=listNumFilters[0], kernel_size=3, padding='same')(x)    
    for idx,nfilters in enumerate(listNumFilters[:-1]):
        for i in range(num_hidden_residuals_blocks):
            x = ResBlockIdV2(nfilters)(x)
        x = ResBlockConvTV2(listNumFilters[idx+1])(x) 
        
    decoder_outputs = layers.Conv2D(filters=output_channels, kernel_size=1, padding="same")(x)
    
    return Model(decoder_inputs, decoder_outputs, name=name)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Gaussian Sampler
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Sampler(keras.layers.Layer):
    """
    From a Tensor (1D) x generate the Tensors (1D) z_mean and z_log_var to sample z.

    Input : vector x
    Computes:
            z_mean    = DFC1(x)
            z_log_var = DFC2(x)   
            z = z_mean+exp(0.5*z_log_var)*epsilon
    Return:
            [z, z_mean, z_log_var]
    """
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.flatten    =  layers.Flatten()
        self.dense1     =  layers.Dense(self.latent_dim, name="z_mean")
        self.dense2     =  layers.Dense(self.latent_dim, name="z_log_var")
        #self.built = True
        
    def call(self, input_data):
        '''
        input_dim is a vector in the latent (codified) space
        '''
        #input_data = layers.Input(shape=self.input_dim)
        x          = self.flatten(input_data)
        z_mean     = self.dense1(x)
        z_log_var  = self.dense2(x)
        
        self.batch = tf.shape(z_mean)[0] 
        self.dim   = tf.shape(z_mean)[1]
        epsilon    = tf.random.normal(shape=(self.batch, self.dim))
        
        z = z_mean + tf.math.exp(0.5 * z_log_var) * epsilon
        
        return [z, z_mean, z_log_var]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    

