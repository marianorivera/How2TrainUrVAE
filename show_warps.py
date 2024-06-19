"""show_warps.py: PatchGan discriminator """

__author__      = "Mariano Rivera"
__copyright__   = "CC BY-NC 4.0"


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def show_many(Imgs, m, n, space=None, offset=0, fname=None,  show=True):
    '''
    '''
    if len(Imgs.shape)==5:
        numblk, num_im_rec, imrows, imcols, imchn = Imgs.shape
    if len(Imgs.shape)==4:
        numblk=1
        num_im_rec, imrows, imcols, imchn = Imgs.shape
    if space == None:
        space=20
       
    canvas = np.ones((imrows*m, numblk*imcols*n + (numblk-1)*space, imchn)) 
    grid_x = np.linspace(0,n, n)

    for k in range(numblk):
        for i in range(m):
            for j in range(n):            
                idx = i+j*m+offset
                r0,r1 = i*imrows,(i+1)*imrows, 
                c0,c1 = (j+(k)*n)*imcols + (k)*space, (j+1+(k)*n)*imcols + (k)*space, 
                im = Imgs[k,idx,:,:,:] if len(Imgs.shape)==5 else Imgs[idx,:,:,:] 
                canvas[r0:r1,c0:c1,:]  =  np.clip((im+1)/2, 0,1)

    if fname!=None:
        image = Image.fromarray((255*canvas).astype('uint8'))
        image.save(fname)
        
    if show:
        plt.figure(figsize=(numblk*3*n, 3*m))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(canvas, cmap="Greys_r")
        plt.show()

def plot_warping(model=None, z1=0, z2=-1, n=3, dims = [256,256,3], save_plot=True):
    
    imrows, imcols, imchn =dims
    canvas = np.ones((imrows, imcols*n, imchn), dtype=np.uint8)          

    grid_x = np.linspace(0,n, n)
    
    for i in range(n):
        alpha = i/(n-1)
        z_new = (1-alpha)*z1 + alpha*z2
        #z_new = tf.expand_dims(z_new, axis=0)
        pred  = model.decoder_model(z_new)
        img = pred[0]

        img = tf.clip_by_value((img+1)*127.5, 0,255) #predictions [-1,1]->[0,255]
        img = tf.cast(img, tf.uint8)
        canvas[0:imrows, i*imcols:(i+1)*imcols, :] = img
       
    plt.figure(figsize=(3*n,3))
    plt.axis('off')
    plt.imshow(canvas, cmap="Greys_r")

    if save_plot:
        im = Image.fromarray(canvas)
        im.save('transition_2.png')

def plot_latent_space(model, image, n=30,figsize=10, scale=1.,latents_start=[0,1], save_plot=True):

    # display a n*n 2D manifold of digits
    input_size=image.shape
    canvas = np.ones((input_size[0]*n, input_size[1]*n,input_size[2]), dtype=np.uint8)
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    
    image = np.expand_dims(image, axis=0)
    predictions, z_sample, z_mean, z_log_var = model(image, training=False)
    
    z = z_mean.numpy().copy()    
    std = np.exp(0.5*z_log_var.numpy()).copy()

    idx1, idx2 = latents_start
    z1 = z_mean[0,idx1]
    z2 = z_mean[0,idx2]
    std1 = std[0,idx1]
    std2 = std[0,idx2]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):

            val1 = z1+std1*yi,  
            val2 = z2+std2*xi 
            z[:,idx1] = val1
            z[:,idx2] = val2
            img = model.decoder_model(z)
            
            img = tf.clip_by_value((img+1)*127.5, 0,255) #predictions [-1,1]->[0,255]
    
            canvas[i*input_size[0] : (i + 1)*input_size[0],
                   j*input_size[1] : (j + 1)*input_size[1],
                   : ] = img

    plt.figure(figsize=(figsize, figsize))
    start_range    = input_size[0] // 2
    end_range      = n*input_size[0] + start_range
    pixel_range    = np.arange(start_range, end_range, input_size[0])
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("$z_{}$".format(latents_start[1]), fontsize=18)
    plt.ylabel("$z_{}$".format(latents_start[0]), fontsize=18)
    plt.tight_layout()
    plt.imshow(canvas, cmap="Greys_r")

    #data = tf.cast(predictions, tf.uint8)    
    if save_plot:
        im   = Image.fromarray(canvas)
        im.save('generated_variants_2.png')
    