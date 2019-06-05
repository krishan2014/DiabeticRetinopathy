#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import seaborn as sns
from sklearn.metrics import confusion_matrix
import collections

## Sources
# How to load and preprocess data: https://www.kaggle.com/meenavyas/diabetic-retinopathy-detection
# How to do transfer learning: https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4


# In[2]:


#Read the training labels
all_labels = pd.read_csv('trainLabels.csv')
#all_labels.head()


# In[3]:


#Create labels for the relaxed 3-class problem
all_labels['level3'] = 0
all_labels.loc[all_labels['level'] > 2, 'level3'] = 2
all_labels.loc[(all_labels['level'] > 0) & (all_labels['level'] < 3), 'level3'] = 1



# In[4]:


#Get information about the cleaned training and test sets
cleantrainDir = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train_clean/"
images_train_all = os.listdir(cleantrainDir)
n_train = len(images_train_all)


# In[5]:


#Find the images that were deleted since they did not make the preprocessing cut
deleted_im = []
for im in all_labels['image']:
    name = im +  ".jpeg"
    if name not in images_train_all:
        deleted_im.append(im) 

#Remove these images from the labels dataframe
all_labels = all_labels[all_labels.image != deleted_im[0]]
all_labels = all_labels[all_labels.image != deleted_im[1]]


# ## Split into test, training and validation sets

# In[9]:


#Do a 25/75 train/test split

three_class = False
if three_class:
    num_levels = 'level3'
else:
    num_levels = 'level'


train_full_ids, test_ids = train_test_split(all_labels['image'], 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = all_labels[num_levels])
#Get the distribution of labels
train_full_labels = all_labels[all_labels['image'].isin(train_full_ids)]
test_labels = all_labels[all_labels['image'].isin(test_ids)]

print('Num train:', train_full_labels.shape[0], 'Num validation:', test_labels.shape[0])
print(train_full_labels[num_levels].value_counts())
print(test_labels[num_levels].value_counts())


# In[10]:


#Do a 25/75 train/val split
train_ids, val_ids = train_test_split(train_full_labels['image'], 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = train_full_labels[num_levels])

#Get the distribution of labels

train_labels = train_full_labels[train_full_labels['image'].isin(train_ids)]
val_labels = train_full_labels[train_full_labels['image'].isin(val_ids)]

print('Num train:', train_labels.shape[0], 'Num validation:', val_labels.shape[0])
print(train_labels[num_levels].value_counts())
print(val_labels[num_levels].value_counts())


# ## Adjust the ordering of the labels to match the order of files in the directories

# In[11]:


if three_class:
    d0 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train3/class0"
    d1 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train3/class1"
    d2 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train3/class2"
    train_class0 = os.listdir(d0)
    train_class1 = os.listdir(d1)
    train_class2 = os.listdir(d2)

else:
    d0 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train/class0"
    d1 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train/class1"
    d2 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train/class2"
    d3 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train/class3"
    d4 = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/train/class4"
    train_class0 = os.listdir(d0)
    train_class1 = os.listdir(d1)
    train_class2 = os.listdir(d2)
    train_class3 = os.listdir(d3)
    train_class4 = os.listdir(d4)


# In[12]:


columns = ['image','level', 'level3']
training_labels_ordered = pd.DataFrame(columns=columns)

for im in train_class0:
    filename = os.path.splitext(im)[0]
    row = train_labels.loc[train_labels['image'] == filename]
    if row.empty:
        print("empty!")
    training_labels_ordered = training_labels_ordered.append(row)

for im in train_class1:
    filename = os.path.splitext(im)[0]
    row = train_labels.loc[train_labels['image'] == filename]
    if row.empty:
        print("empty!")
    training_labels_ordered = training_labels_ordered.append(row)

for im in train_class2:
    filename = os.path.splitext(im)[0]
    row = train_labels.loc[train_labels['image'] == filename]
    if row.empty:
        print("empty!")
    training_labels_ordered = training_labels_ordered.append(row)
    
if three_class == False:
    for im in train_class3:
        filename = os.path.splitext(im)[0]
        row = train_labels.loc[train_labels['image'] == filename]
        if row.empty:
            print("empty!")
        training_labels_ordered = training_labels_ordered.append(row)

    for im in train_class4:
        filename = os.path.splitext(im)[0]
        row = train_labels.loc[train_labels['image'] == filename]
        if row.empty:
            print("empty!")
        training_labels_ordered = training_labels_ordered.append(row)


# ## Fix class imbalance

# In[63]:


class_sample_counts = training_labels_ordered[num_levels].value_counts().tolist() 
train_targets = training_labels_ordered[num_levels].tolist() 
probs = class_sample_counts/np.sum(class_sample_counts)
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

#weights = weights.double()

samples_weight = weights[train_targets]

sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement = True)


# ## Additional image augmentation

# In[36]:


# Data augmentation and normalization for training
# Just normalization for validation

if three_class:
    training_set = 'train3'
    val_set = 'val3'
    test_set = 'test3'
else:
    training_set = 'train'
    val_set = 'val'
    test_set = 'test'

data_transforms = {
    training_set: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5),
        transforms.ToTensor()    ]),
    val_set: transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    test_set: transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}


# In[37]:


data_dir = '/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in [training_set, val_set, test_set]}


# In[122]:



use_weighted_sampling = False

if use_weighted_sampling:
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets[training_set], 
                                                   batch_size=128, sampler = sampler, shuffle=False) , 
'val': torch.utils.data.DataLoader(image_datasets[val_set], batch_size=128,shuffle=True),
             'test': torch.utils.data.DataLoader(image_datasets[test_set], batch_size=128, shuffle=True)}
else:
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets[training_set], 
                                                   batch_size=128, shuffle=True) , 
'val': torch.utils.data.DataLoader(image_datasets[val_set], batch_size=128,shuffle=True),
             'test': torch.utils.data.DataLoader(image_datasets[test_set], batch_size=128, shuffle=True)}
    


# In[39]:


dataset_sizes = {x: len(image_datasets[x]) for x in [training_set, val_set,test_set]}
class_names = image_datasets[training_set].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ## Check if balancing classes is working

# In[ ]:


labs = []
for _ , labels in dataloaders['train']:
    labs = np.append(labs, labels.numpy())

plt.hist(labs)
#seems to be imbalanced


# In[ ]:


train_labels.head()


# In[ ]:


m = list(sampler)
class_counts = np.zeros((3))
for elem in m:
    r = train_labels.iloc[[elem]]
    lev = r.iloc[0]['level3']
    class_counts[lev] += 1

class_counts

#Hmm, weird, the classes seem to be balanced here??


# ## Training loop

# In[43]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    true_train_labels = []
    predicted_train_labels = []
    
    #class_weights.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print("in training")
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                print("in validation")
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #if phase == 'train':
                    #true_train_labels = np.append(true_train_labels, labels.numpy())
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #predicted_train_labels = np.append(predicted_train_labels,preds.cpu().numpy())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if three_class:
                numel = dataset_sizes[phase + str(3)]
            else:
                numel = dataset_sizes[phase]
                
            epoch_loss = running_loss / numel
            epoch_acc = running_corrects.double() / numel
            
            #Save the accuracy and losses
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.cpu().item())
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.cpu().item())
                scheduler.step(epoch_loss) #try this??
                    

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, train_acc, val_loss, val_acc, true_train_labels, predicted_train_labels


# ## Evaluate model

# In[44]:


def evaluate_model(model):

    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    phase = 'test'

    true_labels = []
    predicted_labels = []
    
    

    for inputs, labels in dataloaders['test']:
        true_labels = np.append(true_labels, labels.numpy())
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model_conv(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)


        predicted_labels = np.append(predicted_labels, preds.cpu().numpy())
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    num_correct = running_corrects.cpu().item()
    return true_labels, predicted_labels, running_loss, num_correct


# In[45]:


def summarize_results(true_labels, predicted_labels, running_loss):
    #Get the loss
    print('Loss on Test Data:' ,running_loss/true_labels.shape[0])
    #Get accuracy on the test set:
    print('Accuracy on Test Data: %2.5f%%' % (accuracy_score(true_labels, predicted_labels)))
    print(classification_report(true_labels, predicted_labels))  


# In[46]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# inputs, classes = next(iter(dataloaders['test']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])


# ## Load a model

# In[70]:


weights = weights.to(device)
weights


# In[71]:


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False


# In[72]:


num_ftrs = model_conv.fc.in_features

if three_class:
    num_classes = 3
else:
    num_classes = 5
    
model_conv.fc = nn.Linear(num_ftrs, num_classes)
model_conv = model_conv.to(device)
#criterion = nn.CrossEntropyLoss(weight= weights)
#criterion = nn.CrossEntropyLoss()

if use_weighted_sampling:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss(weight = weights)
    


# In[73]:


optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
optimizer_conv_adam = optim.Adam(model_conv.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 2 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=2, gamma=0.1)
plat_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv_adam, patience = 5, verbose= True)


# ## Run the model

# In[74]:


model_conv, train_loss, train_acc, val_loss, val_acc, true_train_labels, predicted_train_labels = train_model(model_conv, criterion, optimizer_conv_adam,
                         plat_lr_scheduler, num_epochs=20)


# In[78]:


plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Val")
plt.title("Loss")
plt.legend()
#plt.savefig("Loss_res5class_weightloss.png")


# In[79]:


plt.plot(train_acc, label="Train")
plt.plot(val_acc, label="Val")
plt.title("Accuracy")
plt.legend()
#plt.savefig("Acc_res5class_weightloss.png")


# ## Evaluate on the test set

# In[129]:


true_labels, predicted_labels, running_loss, num_correct = evaluate_model(model_conv)


# In[130]:


summarize_results(true_labels, predicted_labels, running_loss)


# In[131]:


sns_plot = sns.heatmap(confusion_matrix(true_labels, predicted_labels), 
            annot=True, fmt="d", cbar = False, cmap = plt.cm.Blues)


# In[ ]:





# ## Do some error analysis

# In[ ]:


testDataLoader = torch.utils.data.DataLoader(image_datasets[test_set], batch_size=128, shuffle=Faelse)


# In[132]:


inputs, classes = next(iter(testDataLoader))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


# In[135]:


nnz = np.nonzero(true_labels - predicted_labels)


# In[136]:


incorrect_indices = nnz[0]


# In[138]:


incorrect_indices


# In[139]:


testDir = "/home/trishajani/gcloud/www.kaggle.com/c/diabetic-retinopathy-detection/download/test/"
test_images = os.listdir(testDir)


# In[ ]:


img =  cv2.imread(imageFullPath)
inp = inp.numpy().transpose((1, 2, 0))
inp = np.clip(inp, 0, 1)
plt.imshow(inp)

