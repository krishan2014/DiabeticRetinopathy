#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import sys
import cv2
import matplotlib
from subprocess import check_output
from PIL import Image
import matplotlib.pyplot as plt
import glob
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import shutil



## Sources
# How to load and preprocess data: https://www.kaggle.com/meenavyas/diabetic-retinopathy-detection
# How to do transfer learning: https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4


# ## Data cleanup

# Read the data and preprocess it. Center the data and get rid of the black borders. Reshape to all 256x256 pixels. 

# In[2]:


#Read and preproces data
def readData(trainDir):
    # loop over the input images
    images = os.listdir(trainDir)
    print("Number of files in " + trainDir + " is " + str(len(images)))
    i = 0
    for imageFileName in images:
        if (imageFileName == "trainLabels.csv"):
            continue
        
        imageFullPath = os.path.join(os.path.sep, trainDir, imageFileName)
        
        trainDirNew = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/test_clean/"
        imageNewPath = os.path.join(os.path.sep, trainDirNew, imageFileName)
        

        im = cv2.imread(imageFullPath)

        if im.size != 0: 

            imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(imgray,8,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            areas = [cv2.contourArea(contour) for contour in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)

            if w / h > 16 / 9:
                # increase top and bottom margin
                newHeight = w / 16 * 9
                y = y - (newHeight - h ) / 2
                h = newHeight
            
            # Crop with the largest rectangle
            crop = im[int(y):int(y+h),int(x):int(x+w)] 
            
            if crop.size != 0:
                resized_img = cv2.resize(crop,(256,256))
                resized_img_npy = np.array(resized_img)
                cv2.imwrite(imageNewPath,resized_img_npy)
                #cv2.imwrite(imageFullPath,resized_img_npy)

                i = i+1

            if i%100 == 0:
                print(i)


    return


# ## Restructure how the data is saved to match format of Imageloader
# 
# For 5 class or 3 class problem

# In[11]:


#Read and preproces data
def reorgData(trainDir):
    # loop over the input images
    images = os.listdir(trainDir)
    print("Number of files in " + trainDir + " is " + str(len(images)))
    i = 0
    train_names = set(train_labels['image'])
    val_names = set(val_labels['image'])
    test_names = set(test_labels['image'])
    for imageFileName in images:
        currPath = os.path.join(os.path.sep, trainDir, imageFileName)
        
        filename = os.path.splitext(imageFileName)[0]

        #figure out where to save the image
        #train or val or test
        #then, which class
        if filename in set(train_labels['image']):
            #it is in the training set
            #data_type = 'train'
            if three_class:
                data_type = 'train3'
            else:
                data_type = 'train'
            filename_label = train_labels.loc[train_labels['image'] == filename, num_levels].iloc[0]
        elif filename in set(test_labels['image']):
            #it is in the test set
            #data_type = 'test'
            if three_class:
                data_type = 'test3'
            else:
                data_type = 'test'
            filename_label = test_labels.loc[test_labels['image'] == filename, num_levels].iloc[0]
        else:
            #it is in the validation set
            #data_type = 'val'
            if three_class:
                data_type = 'val3'
            else:
                data_type = 'val'
            filename_label = val_labels.loc[val_labels['image'] == filename, num_levels].iloc[0]
            
        
        #Generate in the appropriate path
        newPath = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/" + data_type + "/class" + str(filename_label)
        
        #Save in the correct place
        shutil.copy(currPath, newPath)

        if i%1000 == 0:
            print(i)
        i += 1

    return



# In[12]:


reorg = False
if reorg:
    reorgData(cleantrainDir)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




