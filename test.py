# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:48:25 2020

@author: INFO-H-501

test.py is used to do all kinds of tests on test_set on the resulting image (post-proccessing)
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage import io
from data import load_test_data
from train import preprocess



if __name__ == '__main__':
    
    #get mean and std of train data
    
    [mean, std] = np.load('mean_std.npy')
    
    print('-'*30)
	
    print('Loading and preprocessing test data...')
	
    print('-'*30)


    #get imgs_test
	
    imgs_test, masks_test = load_test_data()
	
    if imgs_test.shape[1] == 512: #used if not subsampled
		
        imgs_test = imgs_test[..., np.newaxis]
		
    else:
		
        imgs_test = preprocess(imgs_test)  #used for subsampling

    imgs_test = imgs_test.astype('float32')
	
    imgs_test -= mean
	
    imgs_test /= std
	
	
    #load unet_mask_test from Unet

    print('-'*30)
	
    print('Loading unet_mask_test ...')
	
    print('-'*30)
    
    unet_mask_test = np.load(os.path.join('unet_numpy_masks','unet_mask_test.npy'))

    
    #choice of the image with k in test set
	
    k=70
	
    pred_dir = 'preds'

    df = 1 #used for rescaling
	
    a=rescale_intensity(imgs_test[k][:,:,0],out_range=(-1,1))
	
    b=(unet_mask_test[k][:,:,0])
	
    c=masks_test[k]
    
    b = b.astype('uint16')
	
    c = c.astype('uint16')

    #plot boundaries of the unet in red and the original in blue
    
    boundaries_unet = mark_boundaries(a[::df,::df], b, color=(1,0,0))
	
    boundaries_both = mark_boundaries(boundaries_unet, c[::df,::df], color=(0,0,1))
    
	
    print('-'*30)
	
    print('show result of slice ' + str(k) + ' of test set')
	
    print('-'*30)
    
    
    plt.figure(0)
	
    plt.title('Unet mask')
	
    io.imsave(os.path.join(pred_dir, '0' + str(k) + '_pred.png'), boundaries_both)
	
    io.imshow(os.path.join(pred_dir, '0' + str(k) + '_pred.png'))
	
    io.show()
