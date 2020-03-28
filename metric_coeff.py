# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:36:11 2020

@author: Alain

This code is used to obtain the dice, jaccard coefficient of test set and training set
after using the predict.py program. the code load the unet data that was predicted and 
computes several metrics to assess the segmentation result.
in the main 4 functions can be called to obtain the element of interest
the start of the main is used to select which set you would like to takes

"""

import matplotlib.pyplot as plt
import numpy as np
import os
from functools import reduce
from data import load_test_data, load_train_data
from skimage.metrics import adapted_rand_error
import metrics
from sklearn.metrics import auc



def dice_coef(y_true, y_pred):
    
    div = np.count_nonzero(y_true==1) + np.count_nonzero(y_pred==1)
    
    if div != 0:
    
        res = 2*np.sum(y_true*y_pred)/div
    
    else:
    
        res = 0
    
    return res   


def jaccard_coef(y_true, y_pred):
    
    union = y_true+y_pred
   
    union[union==2] = 1 #for getting the union sum can give 0, 1 or 2
    
    if np.sum(union)!=0:
    
        res = np.sum(y_true*y_pred)/(np.sum(union))
    
    else: 
    
        res = 0
    
    return res   


def add_nums(a, b):

    return a + b


def get_dice_and_jaccard(masks, unet_mask):
	
	##########################################################################
	####compute dice coefficient and jaccard coefficient for test set  #######
    
    step = 50

    setLen = unet_mask.shape[0]
    
    dicex = np.zeros(step-1)
    
    dicey = np.zeros(step-1)

    diceyarr = np.zeros((step-1,setLen))
    
    jaccy = np.zeros(step-1)

    jaccyarr = np.zeros((step-1,setLen))
    

    for i in range(1,step):
        
        Threshold = i/step
        
        dicex[i-1] = Threshold
        

        for k in range(setLen):
            
            b=unet_mask[k]

            c=masks[k]

            d=b.copy()

            d[d<=Threshold] = 0

            d[d>Threshold] = 1

            diceyarr[i-1][k] = dice_coef(c,d)

            jaccyarr[i-1][k] = jaccard_coef(c,d)
        

        dicey[i-1] = reduce(add_nums,diceyarr[i-1])/setLen

        jaccy[i-1] = reduce(add_nums,jaccyarr[i-1])/setLen
		
	
    print('dice: maximum at x = ' + str(np.argmax(dicey)/step)+ ' with value y = ' + str(max(dicey)))
	
    print('jaccard: maximum at x = ' + str(np.argmax(jaccy)/step)+ ' with value y = ' + str(max(jaccy)))


    plt.figure(1)

    plt.plot(dicex,dicey, 'r')

    plt.title('Dice coeff_threshold')

    plt.ylabel('Dicey')

    plt.xlabel('Threshold')
    
    plt.savefig('dice_predict')
    
    
    plt.figure(2)

    plt.plot(dicex,jaccy, 'g')

    plt.title('Jaccard coeff_threshold')

    plt.ylabel('jaccy')

    plt.xlabel('Threshold')

    plt.savefig('jaccard_predict')    
    

def get_test_error_precision_recall():
	
	# error precision and recall for test set
    
    print('-'*30)
	
    print('Compute metrics for test set...')
	
    print('-'*30)
    
    #get imgs_test and masks_test
    
    imgs_test, masks_test = load_test_data()    

    #load unet_mask_test from Unet

    unet_mask_test = np.load(os.path.join('unet_numpy_masks','unet_mask_test.npy'))    
    
    unet_mask_test = unet_mask_test[:,:,:,0]
	
    error, precision, recall = adapted_rand_error(masks_test.astype('uint16'), unet_mask_test.astype('uint16'))
	
    F1 = 2*1/((1/precision)+(1/recall))

	
    print(f"Adapted Rand test error: {error}")
	
    print(f"Adapted Rand test precision: {precision}")
	
    print(f"Adapted Rand test recall: {recall}")
    
    print('F1 test score: ' + str(F1))
	
    np.save(os.path.join('epr', 'err_prec_rec_test.npy'), [error,precision,recall])
	
	
def get_train_error_precision_recall():
	# error precision and recall for train set

    print('-'*30)
	
    print('Compute metrics for train set...')
	
    print('-'*30)
    

    train_set1 = np.load(os.path.join('epr','err_prec_rec_train1.npy'))
	
    train_set2 = np.load(os.path.join('epr','err_prec_rec_train2.npy'))		
	
    train_set3 = np.load(os.path.join('epr','err_prec_rec_train3.npy'))
	
    train_set4 = np.load(os.path.join('epr','err_prec_rec_train4.npy'))
	
    error = (train_set1[0]+train_set2[0]+train_set3[0]+train_set4[0])/4
	
    precision = (train_set1[1]+train_set2[1]+train_set3[1]+train_set4[1])/4
	
    recall = (train_set1[2]+train_set2[2]+train_set3[2]+train_set4[2])/4
    
    F1 = 2*1/((1/precision)+(1/recall))
	
    
    print(f"Adapted Rand train error: {error}")
	
    print(f"Adapted Rand train precision: {precision}")
	
    print(f"Adapted Rand train recall: {recall}")
    
    print('F1 train score: ' + str(F1))
	
    np.save(os.path.join('epr', 'err_prec_rec_train.npy'), [error,precision,recall])


def get_surface_distances(masks, unet_mask):
	#surface distance -> used to compare outlines
    print('-'*30)
	
    print('Compute surface distance metrics...')
	
    print('-'*30)    
    
    threshold = 0.88  #chosen for best dice coefficient
	
    unet_mask[unet_mask<=threshold] = 0
	
    unet_mask[unet_mask>threshold] = 1
	
	
    surface_distances = metrics.compute_surface_distances(masks.astype(np.bool), unet_mask.astype(np.bool), [0.687671,0.687671,3])
	
    average = metrics.compute_average_surface_distance(surface_distances)
	
    hausdorff = metrics.compute_robust_hausdorff(surface_distances, 100)
	
    overlap = metrics.compute_surface_overlap_at_tolerance(surface_distances, 5)
	
    
    print('Average surface distance : ' + str(average))
	
    print('Hausdorff distance : ' + str(hausdorff))
	
    print('Ground truth / Unet mask: ' + str(overlap[0]))
	
    print('Unet mask / Ground truth: ' + str(overlap[1]))
    
    np.save(os.path.join('surfd', 'surfd_predict.npy'), [average, hausdorff, overlap[0], overlap[1]])


def get_train_set_surface_distances():
    
    
    print('-'*30)
	
    print('Compute surface distance for train set...')
	
    print('-'*30)
    

    train_set1 = np.load(os.path.join('surfd', 'surfd_train1.npy'),allow_pickle=True)
	
    train_set2 = np.load(os.path.join('surfd', 'surfd_train2.npy'),allow_pickle=True)
	
    train_set3 = np.load(os.path.join('surfd', 'surfd_train3.npy'),allow_pickle=True)
	
    train_set4 = np.load(os.path.join('surfd', 'surfd_train4.npy'),allow_pickle=True)
    
    averagex = (train_set1[0][0]+train_set2[0][0]+train_set3[0][0]+train_set4[0][0])/4
    
    averagey = (train_set1[0][1]+train_set2[0][1]+train_set3[0][1]+train_set4[0][1])/4    
	
    hausdorff = max(train_set1[1],train_set2[1],train_set3[1],train_set4[1])
	
    overlap0 = (train_set1[2]+train_set2[2]+train_set3[2]+train_set4[2])/4
    
    overlap1 = (train_set1[3]+train_set2[3]+train_set3[3]+train_set4[3])/4

    print('Average surface distance : ' + str(averagex) + ', '+ str(averagey))
	
    print('Hausdorff distance : ' + str(hausdorff))
	
    print('Ground truth / Unet mask: ' + str(overlap0))
	
    print('Unet mask / Ground truth: ' + str(overlap1))
    
    np.save(os.path.join('surfd', 'surfd_train.npy'), [(averagex,averagey), hausdorff, overlap0, overlap1])

    

def get_ROC_curve(masks, unet_mask):
    
    print('-'*30)
	
    print('Compute ROC curve...')
	
    print('-'*30)    
    
    step = 50
    
    specx = np.zeros(step-1)
    
    sensy = np.zeros(step-1)
    
    for i in range(1,step):
    
        Threshold = i/step
    
        d=unet_mask.copy()

        d[d<Threshold] = 0

        d[d>=Threshold] = 1
        
        sensy[i-1], specx[i-1] = metrics.compute_sensitivity_specificity(masks, d)
        
    print(sensy)
    
    print(specx)
    
    print('computed AUC using sklearn.metrics.auc: {}'.format(auc(specx,sensy)))    
    
    plt.figure(1)

    plt.plot(specx,sensy, 'b')

    plt.title('ROC curve')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')
    
    #plt.savefig('ROC')


if __name__ == '__main__':
    
    ##############################################
    ################   test set    ###############
    ##############################################
    
    ### loading test set
    
    print('-'*30)
	
    print('Loading predict set...')
	
    print('-'*30)  
    
    imgs, masks = load_test_data()    #in function of what is chosen in predict.py

    unet_mask = np.load(os.path.join('unet_numpy_masks','unet_mask_test.npy'))    
    
    unet_mask = unet_mask[:,:,:,0]
    
    ##############################################
    ###############   train set    ###############
    ##############################################
    """ loading trainset
    
    print('-'*30)
	
    print('Loading train set...')
	
    print('-'*30)  
    
    imgs, masks = load_train_data()    

    unet_mask = np.load(os.path.join('unet_numpy_masks','unet_mask_train.npy'))   
    """

    
    ##############################################
    #########  smaller train set    ##############
    ##############################################    
    """ load smaller training set
    
    print('-'*30)
	
    print('Loading test set...')
	
    print('-'*30)  
    
    imgs, masks = load_train_data()    
    
    length = imgs.shape[0]

    unet_mask = np.load(os.path.join('unet_numpy_masks','unet_mask_train4.npy'))

    #######  select the corresponding one ######
    
    #masks = masks[0:length//4]
    
    #masks = masks[length//4:length//2]
    
    #masks = masks[length//2:(length*3)//4]
    
    masks = masks[(length*3)//4:length]    
    """
	
		
    
    print('unet_mask and masks shape: ')
	
    print(unet_mask.shape)
	
    print(masks.shape)	
	
	
	############# get dice coefficient plot in function of threshold #########
    
    #get_dice_and_jaccard(masks, unet_mask)
    
	
	############# get metrics for test set   #############
    
    #get_test_error_precision_recall()
    
	
	############## get metrics for train set  ###############
    ########### based on already made metrics calculation ###
    #########      bcs too big for pc gpu      ###########
    
    #get_train_error_precision_recall()
    
	
    ############### get surface distance coefficients for test set only ##############
    
    #get_surface_distances(masks, unet_mask)    

    ############### get surface distance coefficients for train set only ##############
    
    #get_train_set_surface_distances()    		
    
    ################ get ROC curve ###################################
    ################ STILL NEED SOME ADAPTATION#################
    #get_ROC_curve(masks, unet_mask)
