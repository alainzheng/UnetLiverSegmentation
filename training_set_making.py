# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:35:04 2020

@author: Alain

the following code is spefically used to create the full training set
bcs gpu does not support the whole dataset

"""

import numpy as np
import os

# used to concatenante the 4 training set made for easier computation time
df = 1

image_rows = int(512/df)

image_cols = int(512/df) 




train_set = np.ndarray((2161, image_rows, image_cols), dtype=np.uint16)

train_set1 = np.load(os.path.join('unet_numpy_masks','unet_mask_train1.npy'))
	
train_set2 = np.load(os.path.join('unet_numpy_masks','unet_mask_train2.npy'))		
	
train_set3 = np.load(os.path.join('unet_numpy_masks','unet_mask_train3.npy'))
	
train_set4 = np.load(os.path.join('unet_numpy_masks','unet_mask_train4.npy'))

print(train_set1[0])

for index, img in enumerate(train_set1):
    
    train_set[index, :, :] = img
    
for index, img in enumerate(train_set2):
    
    index = index+train_set1.shape[0]
    
    train_set[index, :, :] = img
    
for index, img in enumerate(train_set3):
    
    index = index+train_set1.shape[0]+train_set2.shape[0]
    
    train_set[index, :, :] = img
    
for index, img in enumerate(train_set4):
    
    index = index+train_set1.shape[0]+train_set2.shape[0]+train_set3.shape[0]
    
    train_set[index, :, :] = img
    
print(train_set.shape)

np.save(os.path.join('unet_numpy_masks','unet_mask_train'), train_set)


