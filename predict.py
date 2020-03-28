# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:48:33 2020

@author: Alain

this program enables the use of the determined weight on any type of input data (train or test set)
and thus predict the values of the model on any image and also show the predicted and ground truth 
on the original image

It also gives information about the error, recall and precision of the dataset taken
"""
import os
from skimage.exposure import rescale_intensity
import numpy as np
from skimage import io
from skimage.segmentation import mark_boundaries
from data import load_test_data, load_train_data
from train import preprocess, get_unet
from skimage.metrics import adapted_rand_error
import matplotlib.pyplot as plt


import time

start_time = time.time()

if __name__ == '__main__':
    
    
	
    print('-'*30)
	
    print('Loading and preprocessing test data...')
	
    print('-'*30)


    #get images and masks from data set
	
    imgs, masks = load_test_data()  #change here which data set, may need to adapt shape for test_set
	
    #length = imgs.shape[0]
    
	
	### separated into 4 packs for faster computation time
    ### also need to change some lines:  92, 121 and 144 
    ### due to no access to better gpu (covid19 case)

	
    #######   select one of the following  #####
    
    #imgs = imgs[0:length//4]
    
    #imgs = imgs[length//4:length//2]
    
    #imgs = imgs[length//2:(length*3)//4]
    
    #imgs = imgs[(length*3)//4:length]
	
    
    #######  select the corresponding one ######
    
    #masks = masks[0:length//4]
    
    #masks = masks[length//4:length//2]
    
    #masks = masks[length//2:(length*3)//4]
    
    #masks = masks[(length*3)//4:length]
	
	
	
    if imgs.shape[1] == 512: #used if not subsampled
		
        imgs = imgs[..., np.newaxis]
		
    else:
		
        imgs = preprocess(imgs)  #used for subsampling
	
    imgs = imgs.astype('float32')
    
    #Normalization of the test set

    mean, std = np.load('mean_std.npy')
    
    imgs -= mean
    
    imgs /= std
    
    
    print('-'*30)
	
    print('Loading unet on model...')
    
    model = get_unet()
	
    print('Loading saved weights...')
    
    model.load_weights('weights.h5')
	
    print('Predicting masks on test data...')
	
    print('-'*30)
	
    unet_mask = model.predict(imgs, verbose=1)[:,:,:,0]
    
    imgs = imgs[:,:,:,0]

    
	#####  saving unet mask
    
    np.save(os.path.join('unet_numpy_masks','unet_predict.npy'), unet_mask)
    
    
    #####  used if prediction already done
    
    #unet_mask = np.load(os.path.join('unet_numpy_masks','unet_predict.npy'))
		
    unet_mask = unet_mask.astype('uint8')

    
    #######  select the corresponding one if loaded  ######

    #unet_mask = unet_mask[0:length//4]

    #unet_mask = unet_mask[length//4:length//2]
    
    #unet_mask = unet_mask[length//2:(length*3)//4]
    
    #unet_mask = unet_mask[(length*3)//4:length]
	
    print('imgs, unet_mask and masks shape: ')
    
    print(imgs.shape)	
    
    print(unet_mask.shape)
	
    print(masks.shape)
    
    print("--- %s seconds ---" % (time.time() - start_time))
	
    
    #### used to save the data files for data set
    
    
    print('-' * 30)
	
    print('Saving predicted masks to files...')
	
    print('-' * 30)
    
    pred_dir = 'predicted'
    
	
    if not os.path.exists(pred_dir): #creates the file pred if it does not exist
    
        os.mkdir(pred_dir)

    for k in range(len(unet_mask)): 
        
        a = rescale_intensity(imgs[k],out_range=(-1,1))
        
        b = unet_mask[k]
        
        c = masks[k]
                
        #plot boundaries of the unet in <<red>> and the original in <<blue>>
        
        boundaries_unet = mark_boundaries(a, b, color=(1,0,0)) #red is unet
        
        boundaries_both = mark_boundaries(boundaries_unet, c, color=(0,0,1)) #blue is original
        
        ####  str(k+x) x can be 0, length//4, length//2, (length*3)//4 
        ####  depending on the part taken of data set
        
        io.imsave(os.path.join(pred_dir, str(k) + '_pred.png'), boundaries_both)
        
        ###used to look at picture k
        
        if k == 0:
            
            plt.figure(1)
    
            plt.imshow(boundaries_both, cmap=plt.cm.gray)    #show mask of image
    
            io.show()
     
        
    
    print('-'*30)
	
    print('Compute metrics...')
	
    print('-'*30)

    error, precision, recall = adapted_rand_error(masks.astype('uint8'), unet_mask.astype('uint8'))
	
	
    print(f"Adapted Rand error: {error}")
	
    print(f"Adapted Rand precision: {precision}")
	
    print(f"Adapted Rand recall: {recall}")
	
	
    
    
    