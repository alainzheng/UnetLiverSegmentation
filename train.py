# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:25:24 2019

@author: Alain

inspired by: https://github.com/soribadiaby/Deep-Learning-liver-segmentation-project.git
"""

from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.segmentation import mark_boundaries
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from keras.callbacks import History
from skimage import io
from data import load_train_data, load_test_data



K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#We divide here the number of rows and columns by df because we undersample our data (We take one pixel over two)
df = 1 #data factor (resize the data to 256,128 etc)

img_rows = int(512/df)

img_cols = int(512/df)

smooth = 1.




#The functions return our metric and loss

def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):
    
    return -dice_coef(y_true, y_pred)


    
#The different layers in our neural network model (including convolutions, maxpooling and upsampling)

def get_unet():
    
    inputs = Input((img_rows, img_cols, 1))
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9) # because binary problem

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])

    return model

    


def preprocess(imgs):
    
    #We adapt here our dataset samples dimension so that we can feed it to our network
    
    #the first line is only useful if image dimension are subsampled with df
    
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint16) #initialize dimension of processed images
    
    for i in range(imgs.shape[0]):
        
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True) #resize the image if df !=1,  subsampling
        
    imgs_p = imgs_p[..., np.newaxis]
    
    print(imgs_p.shape)
    
    return imgs_p
    


def train_and_predict():
    
    print('-'*30)
    
    print('Loading and preprocessing train data...')
    
    print('-'*30)
    
    imgs_train, masks_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    
    masks_train = preprocess(masks_train)
    
    
    #transform to float so that values can be used for calculating mean and std
    
    imgs_train = imgs_train.astype('float32')
    
    mean = np.mean(imgs_train)  # mean for data centering
    
    std = np.std(imgs_train)  # std for data normalization
    
    np.save('mean_std.npy', [mean,std]) #saved to be used elsewhere
    
	#Normalization of the train set

    imgs_train -= mean
    
    imgs_train /= std

    masks_train = masks_train.astype('float32')

    print('-'*30)
    
    print('Creating and compiling model...')
    
    print('-'*30)
    
    
    model = get_unet()
    
    #modifiy this line to have other set of weights, used to handle fitting
    #Saving the weights and the loss of the best predictions we obtained

    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    
    print('Fitting model...')
    
    print('-'*30)
    
    history = model.fit(imgs_train, masks_train, batch_size=10, epochs=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint]) #validation split chosen as 0.2 
    
    
    
    
############################   testing test data ###############
   
    print('-'*30)
    
    print('Loading and preprocessing test data...')
    
    print('-'*30)
    
    imgs_test, masks_test = load_test_data()
    
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    
    #Normalization of the test set

    imgs_test -= mean
    
    imgs_test /= std
    

    print('-'*30)
    
    print('Loading saved weights...')
    
    print('-'*30)
    
    model.load_weights('weights.h5')

    print('-'*30)
    
    print('Predicting masks on test data...')
    
    print('-'*30)
    
    
    
    unet_mask_test = model.predict(imgs_test, verbose=1)
    
    np.save(os.path.join('unet_numpy_masks','unet_mask_test.npy'), unet_mask_test)  #saving unet mask
    
    
    #Saving our predictions in the directory 'preds'

    print('-' * 30)
    
    print('Saving predicted masks to files...')
    
    print('-' * 30)
    
    pred_dir = 'preds'
    
    
    if not os.path.exists(pred_dir): #creates the file pred if it does not exist
    
        os.mkdir(pred_dir)

    for k in range(len(unet_mask_test)):
        
        a = rescale_intensity(imgs_test[k][:,:,0],out_range=(-1,1))
        
        b = (unet_mask_test[k][:,:,0]).astype('uint8')
        
        c = masks_test[k]
                
        #plot boundaries of the unet in <<red>> and the original in <<blue>>
        
        boundaries_unet = mark_boundaries(a, b, color=(1,0,0)) #red is unet
        
        boundaries_both = mark_boundaries(boundaries_unet, c, color=(0,0,1)) #blue is original
        
        io.imsave(os.path.join(pred_dir, '0' + str(k) + '_pred.png'), boundaries_both)
    
    #plotting our dice coeff results in function of the number of epochs

    
    plt.plot(history.history['dice_coef'])
    
    plt.plot(history.history['val_dice_coef'])
    
    plt.title('Model dice coeff')
    
    plt.ylabel('Dice coeff')
    
    plt.xlabel('Epoch')
    
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig('dice_epoch')
    
    plt.show()
    


if __name__ == '__main__':
    
    train_and_predict()