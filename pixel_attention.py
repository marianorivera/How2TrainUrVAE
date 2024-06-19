"""pixel_attention.py: Pixel Attention"""

__author__      = "Mariano Rivera"
__copyright__   = "CC BY-NC 4.0"


import tensorflow as tf
import keras
from keras.layers import Conv2D, AveragePooling2D
from keras.activations import sigmoid, relu

# https://arxiv.org/pdf/2010.01073.pdf 
# https://pypi.org/project/visual-attention-tf/
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
class PixelAttention2D_v1(keras.layers.Layer):
    """
    Implements Pixel Attention insprired in (Hengyuan Zhao et al) for 
    convolutional networks in tensorflow
    
    Inputs need to be Conv2D feature maps x.
    1. y = Conv2D with k=1 
    2. Atention matrix Aij = x_i' y_j 
    2. A = Sigmoid(A) to create attention maps
    4. x = x+Ax  Residual upgrade of the original tensor 
    
    Args:
    * nf [int]: number of filters or channels
    * name : Name of layer
    Call Arguments:
    * Feature maps : Conv2D feature maps of the shape `[batch,W,H,C]`.
    Output;
    Attention activated Conv2D features of shape `[batch,W,H,C]`.

    Here is a code example for using `PixelAttention2D_v2` in a CNN:
    ```python
    x = Input(shape=(B, 256,256,3))
    x = Conv2D(filters=32,  kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)  # (B, 128,128,32)
    x = Conv2D(filters=62,  kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)  # (B, 64,64,64)
    x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)  # (B, 32,32,128)
    x_attended = PixelAttention(cnn_layer.shape[-1])(x) # (B, 32,32,128)
    ```
    """
    def __init__(self, nf,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = nf
        self.conv1 = Conv2D(filters=nf, kernel_size=1)
        self.conv2 = Conv2D(filters=nf, kernel_size=1)
    
    @tf.function
    def call(self, x):
        '''
        computes the self attention matrix of the tensor x (b, h, w, c)
        resulting in a matrix of dimensions (b, h*w, h*w). Then apply a sigmoidal 
        activationt to the matrix and finally apply the attention to the input tensor x

        MR ago 2023
        '''
        # dimensions of the input tensor
        batch_size, height, width, channels = x.shape
        # apply a convolution of 1x1 to the input tensor
        y = self.conv1(x)
        x = self.conv2(x)
        
        # Reshape tensors(b, h*w, c)
        y_reshaped = tf.reshape(y, [-1, height*width, self.latent_dim])
        x_reshaped = tf.reshape(x, [-1, height*width, self.latent_dim])

        #print(x_reshaped, y_reshaped, '--' ,end='')
        # Compute attention --autocorrelation-- matrix (b, h*w, h*w)
        self.attention_matrix= sigmoid(tf.einsum('bic,bjc->bij', x_reshaped, y_reshaped))
        # Compute the output (b, h*w, c)
        out = tf.einsum('bij,bjc->bic', self.attention_matrix, x_reshaped)
        # Reshape attented tensor (b,h,w,c)
        out = tf.reshape(out, [-1, height, width, self.latent_dim])
        return x+out
    
    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.latent_dim})
        return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
class PixelCrossAttention2D_v1(keras.layers.Layer):
    """
    Implements Pixel Attention insprired in (Hengyuan Zhao et al) for 
    convolutional networks in tensorflow
    
    Inputs need to be Conv2D feature maps x, and reference y.
    1. x = Conv2D(x) with k=1 
       y = Conv2D(y) with k=1 
    2. Atention matrix Aij = x_i' y_j 
    2. A = Sigmoid(A) to create attention maps
    4. x = x+Ay  Residual upgrade of the original tensor with the relevant patches of y
    
    Args:
    * nf [int]: number of filters or channels
    * name : Name of layer
    Call Arguments:
    * Feature maps : Conv2D feature maps of the shape `[batch,W,H,C]`.
    Output;
    Attention activated Conv2D features of shape `[batch,W,H,C]`.

    Here is a code example for using `PixelAttention2D_v2` in a CNN:
    ```python
    x = Input(shape=(B, 256,256,3))
    y = Input(shape=(B, 256,256,3))
    x = ConvolutionalStageX(...)(x)  # (B, 32,32,128)
    y = ConvolutionalStageY(...)(x)  # (B, 32,32,128)
    x_attended = PixelCrossAttention(cnn_layer.shape[-1])(x,y) # (B, 32,32,128)
    ```
    """
    def __init__(self, nf, is_residual=True,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = nf
        self.is_residual = is_residual
        self.conv1 = Conv2D(filters=nf, kernel_size=1)
        self.conv2 = Conv2D(filters=nf, kernel_size=1)
    
    @tf.function
    def call(self, x,y):
        '''
        computes the self attention matrix of the tensor x (b, h, w, c)
        resulting in a matrix of dimensions (b, h*w, h*w). Then apply a sigmoidal 
        activationt to the matrix and finally apply the attention to the input tensor x

        MR ago 2023
        '''
        # dimensions of the input tensor
        batch_size, height, width, channels = x.shape
        # apply a convolution of 1x1 to the input tensor
        x = self.conv1(x)
        y = self.conv2(y)
  
        # Reshape tensors(b, h*w, c)
        y_reshaped = tf.reshape(y, [-1, height*width, self.latent_dim])
        x_reshaped = tf.reshape(x, [-1, height*width, self.latent_dim])

        #print(x_reshaped, y_reshaped, '--' ,end='')
        # Compute attention --autocorrelation-- matrix (b, h*w, h*w)
        self.attention_matrix= sigmoid(tf.einsum('bic,bjc->bij', y_reshaped, x_reshaped))
        # Compute the output (b, h*w, c)
        out = tf.einsum('bij,bjc->bic', self.attention_matrix, y_reshaped)
        # Reshape attented tensor (b,h,w,c)
        out = tf.reshape(out, [-1, height, width, self.latent_dim])
        if self.is_residual: out +=x
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.latent_dim,  "residual":self.is_residual})
        return config
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
class Multihead_PixelAttention_v1(keras.layers.Layer):
    '''
    Multiple had attentions
    '''
    def __init__(self, numfilters, numheads=3, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = numfilters
        self.numheads   = numheads
        self.conv1      = Conv2D(filters=numfilters, kernel_size=1)
        self.Heads      = []
        for h in range(self.numheads):
            self.Heads.append(PixelCrossAttention2D_v1(numfilters, is_residual=False))
        
    @tf.function
    def call(self, x,y):
        x_attended = self.conv1(x)
        for head in self.Heads:
            x_attended += head(x,y)
        return x_attended

    def get_config(self):
        config = super().get_config()
        config.update({"Num_heads: ": self.Heads})
        return config
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         

