"""
vae.py: variational Autoencoder

with residuals encoder and decoder, and attention in the encoder.
"""

__author__      = "Mariano Rivera"
__copyright__   = "CC"


import tensorflow as tf
import keras
import keras.layers as layers
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.layers import Dropout, BatchNormalization, Activation, LeakyReLU
from keras.models import Model, Sequential

from reader import ResEncoder, ResDecoder, Sampler

class VAE(keras.Model):
    def __init__(self, 
                 input_dim, latent_dim,
                 encoder_conv_filters,   encoder_conv_kernel_size,   encoder_conv_strides,
                 decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
                 r_loss_factor=1, summary=False,**kwargs):
        
        super().__init__(**kwargs)

        # Architecture
        self.input_dim                 = input_dim
        self.latent_dim                = latent_dim
        
        self.encoder_conv_filters      = encoder_conv_filters
        self.encoder_conv_kernel_size  = encoder_conv_kernel_size
        self.encoder_conv_strides      = encoder_conv_strides
        
        self.decoder_conv_t_filters    = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size= decoder_conv_t_kernel_size
        self.decoder_conv_t_strides    = decoder_conv_t_strides
        
        self.n_layers_encoder          = len(self.encoder_conv_filters)
        self.n_layers_decoder          = len(self.decoder_conv_t_filters)

        self.r_loss_factor             = r_loss_factor
    
        # Encoder
        self.encoder_model   = ResEncoder(input_dim     = self.input_dim, 
                                          latent_dim    = self.latent_dim//8,
                                          listNumFilters= self.encoder_conv_filters, 
                                          attention     = False,
                                          name          = f'encoder',) 
        self.encoder_output_dim = self.encoder_model.output_shape[1:] #(4,4,self.latent_dim) #self.encoder_model.last_conv_size

        # Sampler
        self.sampler_model   = Sampler(input_dim        = self.encoder_output_dim,
                                       latent_dim       = self.latent_dim)     
        # Decoder
        self.decoder_model  = ResDecoder(input_dim       = self.latent_dim,
                                         start_conv_dim  = self.encoder_output_dim,
                                         output_channels = self.input_dim[-1],
                                         listNumFilters  = self.decoder_conv_t_filters, 
                                         latent_dim      = self.latent_dim,
                                         name            = 'decoder')    
        self.built = True
    """
    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({"units": self.units})
        return config 
    """;
    
    @tf.function
    def generate(self, z_sample):
        return self.decoder_model(z_sample)

    @tf.function
    def codify(self, images):
        x = self.encoder_model.predict(images)
        z, z_mean, z_log_var= self.sampler_model(x)
        return z, z_mean, z_log_var
    
    @tf.function
    def call(self, inputs, training=False):
        '''
        '''
        x                    = self.encoder_model(inputs)
        z, z_mean, z_log_var = self.sampler_model(x)
        pred                 = self.decoder_model(z)        
        return pred, z, z_mean, z_log_var