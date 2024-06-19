"""discriminator.py: PatchGan discriminator """

__author__      = "Mariano Rivera"
__copyright__   = "CC BY-NC 4.0"


import tensorflow as tf
import keras
import keras.layers as layers
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.layers import Dropout, BatchNormalization, Activation, LeakyReLU
from keras.models import Model, Sequential

class Discriminator(keras.Model):
    def __init__(self, 
                 input_dim, 
                 discriminator_conv_filters, 
                 discriminator_conv_kernel_size, 
                 discriminator_conv_strides, 
                 use_batch_norm=True, use_dropout=True, **kwargs):
        '''
        '''
        #super(Discriminator, self).__init__(**kwargs)
        super().__init__(**kwargs)

        # lista are transformed to ListWrapper => L -> LW[0]
        self.input_dim                       = input_dim,
        self.discriminator_conv_filters      = discriminator_conv_filters,
        self.discriminator_conv_kernel_size  = discriminator_conv_kernel_size,
        self.discriminator_conv_strides      = discriminator_conv_strides,
        
        self.use_batch_norm                  = use_batch_norm,
        self.use_dropout                     = use_dropout,
        self.n_layers_discriminator          = len(discriminator_conv_filters)
        
        self.model = self.discriminator_model()
        self.built = True
             
    def get_config(self):
        config = super().get_config()
        #config.update({"units": self.units})
        return config
    
    def discriminator_model(self):
        '''
        '''
        discriminator_input = layers.Input(shape=self.input_dim[0], name='discriminator' )
        x = discriminator_input
        
        for i in range(self.n_layers_discriminator):            
            #print(i,  self.encoder_conv_filters[0][i])
            x = Conv2D(filters     = self.discriminator_conv_filters[0][i],
                       kernel_size = self.discriminator_conv_kernel_size[0][i],
                       strides     = self.discriminator_conv_strides[0][i],
                       padding     = 'same',
                       name        = 'discriminator_conv_' + str(i),)(x)
            if self.use_batch_norm: 
                x = BatchNormalization()(x)
            if i < self.n_layers_discriminator-1: # no in the last conv layer
                x = LeakyReLU()(x)
            if self.use_dropout:    
                x = Dropout(rate = 0.25)(x)
                
        self.last_conv_size = x.shape[1:]
        discriminator_output = x
        model = keras.Model(discriminator_input, discriminator_output)
        return model
        
    
    def call(self, inputs):
        '''
        '''
        return self.model(inputs) 
