# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:22:12 2019

@author: Alain

inspired by: https://github.com/soribadiaby/Deep-Learning-liver-segmentation-project.git

By changing the load test data, you can choose between the test set with only liver and the one without
"""

import numpy as np # linear algebra
import pydicom
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Some constants 

INPUT_FOLDER = 'liver/Train_Sets/CT/'

patients = os.listdir(INPUT_FOLDER) #list of patients 'number'

k_cross_val = 2 #number of patient taken as the test set

df = 1 #data factor (resize the data to 256,128 etc), need to be 1 for final result

image_rows = int(512/df)

image_cols = int(512/df) 

#slices_patient = np.zeros(len(patients))  # for information purposes

################# load images and mask (2873 slices)  ###############################


def create_images(path):

    slices = [(path + '/' + s) for s in os.listdir(path)]

    return slices



def create_masks(path):

    masks = [(path + '/' + s) for s in os.listdir(path)]
    
    return masks


########################   create train data   #############################


def create_train_data():
    
    print('creating train data')

    imgs_train = []    #training images

    masks_train = []    #training masks (corresponding to the liver)

    train_images = []  # create train file names list

    train_masks = []  #create mask file names list


    for patient_number in range(len(patients)-k_cross_val): #takes all the patients except 2 for test set

        train_images.extend(create_images(INPUT_FOLDER + patients[patient_number] + '/DICOM_anon/'))

        train_masks.extend(create_masks(INPUT_FOLDER + patients[patient_number] + '/Ground/'))
    
    for liver, mask in zip(train_images, train_masks): # transform images into np.array 2D
        
        image_2d = pydicom.dcmread(liver).pixel_array[::df,::df]   # change borders for less data size

        mask_2d = mpimg.imread(mask)[::df,::df]

        if len(np.unique(mask_2d)) != 1:  # mask without liver not taken

            masks_train.append(mask_2d)

            imgs_train.append(image_2d)
            
            
    imgs = np.ndarray((len(imgs_train), image_rows, image_cols), dtype=np.uint16)

    imgs_mask = np.ndarray((len(masks_train), image_rows, image_cols), dtype=np.uint8)    


    for index, img in enumerate(imgs_train):

        imgs[index, :, :] = img

        

    for index, img in enumerate(masks_train):

        imgs_mask[index, :, :] = img



    np.save('imgs_train.npy', imgs)

    np.save('masks_train.npy', imgs_mask)

    print('Saving to .npy files done.')



def load_train_data():

    imgs_train = np.load('imgs_train.npy')

    masks_train = np.load('masks_train.npy')

    return imgs_train, masks_train
    
    
###########################   create test data   #########################


def create_test_data():
    
    print('creating test data')

    imgs_test = []    #training images

    masks_test = []    #training masks (corresponding to the liver)

    test_images = []  # create test file names list

    test_masks = []  #create mask file names list


    for patient_number in range(len(patients)-k_cross_val, len(patients)): #takes some patients as test

        test_images.extend(create_images(INPUT_FOLDER + patients[patient_number] + '/DICOM_anon/'))

        test_masks.extend(create_masks(INPUT_FOLDER + patients[patient_number] + '/Ground/'))


    for liver, mask in zip(test_images, test_masks): # transform images into np.array 2D

        image_2d = pydicom.dcmread(liver).pixel_array[::df,::df]

        mask_2d = mpimg.imread(mask)[::df,::df]
        

        masks_test.append(mask_2d)

        imgs_test.append(image_2d)
    
    
    imgs = np.ndarray((len(imgs_test), image_rows, image_cols), dtype=np.uint16)

    imgs_mask = np.ndarray((len(masks_test), image_rows, image_cols), dtype=np.uint8)    


    for index, img in enumerate(imgs_test):

        imgs[index, :, :] = img

        

    for index, img in enumerate(masks_test):

        imgs_mask[index, :, :] = img



    np.save('imgs_test.npy', imgs)   # added a 1 for smaller size  of slices

    np.save('masks_test.npy', imgs_mask)

    print('Saving to .npy files done.')


    
def load_test_data():

    imgs_test = np.load('imgs_test.npy')  # added a 1 for smaller size  of slices

    masks_test = np.load('masks_test.npy')

    return imgs_test, masks_test



########################   main function for data     ####################


if __name__ == '__main__':
    
    print('-'*30)

    print('Data explanation')

    print('-'*30)
    
	# only for information purpose
	
    slices_patient = np.load('liver slices.npy')
	
    tot_train = 0
	
    tot_test = 0
    
	
    for i in range(len(patients)-k_cross_val):    
		
        tot_train += len(os.listdir(INPUT_FOLDER + patients[i] + '/DICOM_anon/'))
        
        print('patient ' + str(patients[i]) +  ' has ' + 

              str(len(os.listdir(INPUT_FOLDER + patients[i] + '/DICOM_anon/'))) + ' slices and ' + 

              str(slices_patient[i]) + ' contain liver')

    print('-'*30)
		
    print('Training set contains patients: ' + '\n' + str(patients[:len(patients)-k_cross_val]) + ' and ' +

        str(np.sum(slices_patient[:len(patients)-k_cross_val])) + ' out of ' + str(tot_train))



    print('-'*30)

    for i in range(len(patients)-k_cross_val, len(patients)): 

        tot_test += len(os.listdir(INPUT_FOLDER + patients[i] + '/DICOM_anon/'))
		
        print('patient ' + str(patients[i]) +  ' has ' + 

              str(len(os.listdir(INPUT_FOLDER + patients[i] + '/DICOM_anon/'))) + ' slices and ' + 

              str(slices_patient[i]) + ' contain liver')

    print('-'*30)

    print('Test set contains patients: ' + '\n' + str(patients[len(patients)-k_cross_val:]) + ' and ' +

         str(np.sum(slices_patient[len(patients)-k_cross_val:])) + ' out of ' + str(tot_test))
    
    
    # only necessary to create the files .npy, run once at the start of project
    
    """
    create_train_data()
    
    create_test_data()
    """
    
    print('-'*30)
    
    print('Data created')
    
    print('-'*30)
    
    
    #just show an example based on image number
    
    """
    im, ma = load_train_data() # train data only has data with liver
    
    image_number = 100  #from 0 to 2179 (2872 if non-liver mask taken)

    plt.figure(1)
    
    plt.imshow(im[image_number], cmap=plt.cm.gray)    #show mask of image
    
    io.show()
    """


    # show n first image & mask

    """
    plt.figure()
    
    plt.figure(figsize=[n*4,6])
    
    n=10
    
    for i in range(1,n):
    
        plt.subplot(2,n,i)
        
        plt.imshow(im[i-1], cmap=plt.cm.gray)    #show image
        
        plt.subplot(2,n,10+i)
        
        plt.imshow(ma[i-1], cmap=plt.cm.gray)    #show mask of image
    """
