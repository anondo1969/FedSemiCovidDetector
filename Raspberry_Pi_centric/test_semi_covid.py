#!/usr/bin/env python
# coding: utf-8

# ## Setup
# 

#CUDA_VISIBLE_DEVICES=0 python3 test_semi_covid.py model.pth 45

import sys

checkpoint_path = sys.argv[1]
image_id = int(sys.argv[2])

print(checkpoint_path)



import torch
import pdb

if not torch.cuda.is_available():
  print("GPU not available. CPU training will be too slow.")
else:
    print("device name", torch.cuda.get_device_name(0))


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import random
import cv2
import copy
import os
import pdb
import time
import gc
from scipy.io import loadmat

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

from collections import namedtuple, defaultdict
from torch.jit.annotations import Optional
from copy import copy
from itertools import cycle

import torch
from torch import nn,optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split


starting_time = time.time()


# # Define Model

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(in_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.25)
    )   

def generate_saliency(inputs, encoder, optimizer):
  inputs2 = copy(inputs)
  inputs2.requires_grad = True
  encoder.eval()

  conv5, conv4, conv3, conv2, conv1, scores = encoder(inputs2)

  score_max, score_max_index = torch.max(scores, 1)
  score_max.backward(torch.FloatTensor([1.0]*score_max.shape[0]).to(device))
  saliency, _ = torch.max(inputs2.grad.data.abs(),dim=1)
  saliency = inputs2.grad.data.abs()
  optimizer.zero_grad()
  encoder.train()

  return saliency

class CovidMix(nn.Module):

    def __init__(self, n_class = 1):
        super().__init__()

        self.encoder = Encoder(1)
        self.decoder = Decoder(1)
        self.generate_saliency = generate_saliency
        

    def forward(self, x, optimizer):
        
        saliency = self.generate_saliency(x, self.encoder, optimizer)
        conv5, conv4, conv3, conv2, conv1, outC = self.encoder(x)
        outSeg = self.decoder(x, conv5, conv4, conv3, conv2, conv1, saliency)

        # return outSeg, outC, saliency
        return outSeg, outC

class Encoder(nn.Module):

    def __init__(self, n_class = 1):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)
        self.dconv_down5 = double_conv(128, 256)      
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))       
        self.fc = nn.Linear(256, 2) 

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x1 = self.maxpool(conv5)
        
        avgpool = self.avgpool(x1)
        avgpool = avgpool.view(avgpool.size(0), -1)
        outC = self.fc(avgpool)
        
        return conv5, conv4, conv3, conv2, conv1, outC

class Decoder(nn.Module):

    def __init__(self, n_class = 1, nonlocal_mode='concatenation', attention_dsample = (2,2)):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(256 + 128 + 2, 128)
        self.dconv_up3 = double_conv(128 + 64, 64)
        self.dconv_up2 = double_conv(64 + 32, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)
        self.conv_last = nn.Conv2d(16, n_class, 1)

        self.conv_last_saliency = nn.Conv2d(17, n_class, 1)
        
        
    def forward(self, input, conv5, conv4, conv3, conv2, conv1, saliency):
  
        bridge = torch.cat([input, saliency], dim = 1)
        bridge = nn.functional.interpolate(bridge, scale_factor=0.125, mode='bilinear', align_corners=True, recompute_scale_factor=True)

        x = self.upsample(conv5)
        
        '''
        print(input.shape)
        print(x.shape)
        print(bridge.shape)
        print(conv5.shape)
        print(conv4.shape)
        print(conv3.shape)
        print(conv2.shape)
        print(conv1.shape)
        print(saliency.shape)
        '''
        
        x = torch.cat([x, conv4, bridge], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)       

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        # pdb.set_trace()
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1) 

        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    

# # Testing
# To test the code, load the provided pth file after instantiating the model. Then, run the test code and receive a prediction.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidMix(1).to(device)
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))


# ## Load Data

# add the provided classification image to a folder and load it 

data_dir = 'covid_xray'

transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=1),
  transforms.Resize((256, 256)), 
  transforms.ToTensor()
])

#print(len(os.listdir(data_dir + "/test/COVID")) + len(os.listdir(data_dir + "/test/NORMAL")))

dataset = ImageFolder(data_dir+'/test', transform = transform)

batch_size = 1

test_loader_class = DataLoader(dataset, batch_size, num_workers=2, pin_memory=True, shuffle = False)

print("Classification test data loading completed")


# for classification predictions, run this
model.eval()
correct = 0
total = 0
predictions = np.array([])
with torch.set_grad_enabled(True):
  
  for i, data in enumerate(test_loader_class):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      _, outC = model(inputs, optimizer_ft)
      _, predicted = torch.max(outC.data, 1)
      predictions = np.append(predictions, predicted.data.cpu().numpy())
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100 

print('Test accuracy: %d %%' % (100 * correct / total))
#print(predictions)



# load segmentation dataset
datamat = loadmat('jsrt.mat')

#datamat.keys()

x_train = datamat["x_train"]
y_train = datamat["y_train"]
x_val = datamat["x_val"]
y_val = datamat["y_val"]
x_test = datamat["x_test"]
y_test = datamat["y_test"]


x_train = np.array(x_train).reshape(len(x_train),256, 256)
y_train = y_train[:,:,:,0] + y_train[:,:,:,1]
y_train = np.array(y_train).reshape(len(y_train),1, 256, 256)

x_val = np.array(x_val).reshape(len(x_val),256, 256)
y_val = y_val[:,:,:,0] + y_val[:,:,:,1]
y_val = np.array(y_val).reshape(len(y_val),1, 256, 256)

x_test = np.array(x_test).reshape(len(x_test),256, 256)
y_test = y_test[:,:,:,0] + y_test[:,:,:,1]
y_test = np.array(y_test).reshape(len(y_test),1, 256, 256)

class Dataset(Dataset):
  def __init__(self, x, y, transform=None):
    self.input_images = x
    self.target_masks = y
    self.transform = transform

  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    image = self.input_images[idx]
    mask = self.target_masks[idx]
    if self.transform:
      image = self.transform(image)

    return [image, mask]

# use the same transformations for train/val in this example
trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5))
])


# to view the predicted mask, run the following
# no need to save it every time if it is saved once
try:
    pred_masks = np.load('pred_masks_'+checkpoint_path)
    
except:
    
    # to get a segmentation prediction, load the image and run the following code
    model.eval() 

    test_dataset = Dataset(x_test, y_test, transform = trans)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print("Segmentation test data loading completed")
    pred_masks = []
    count=0
    for inputs, labels in test_loader:
      count += 1
      gc.collect()
      torch.cuda.empty_cache()

      inputs = inputs.to(device=device, dtype=torch.float)
      labels = labels.to(device=device, dtype=torch.float)
      pred, _, = model(inputs, optimizer_ft)
      pred = torch.sigmoid(pred)
      pred = pred.data.cpu().numpy()
      for i in range (len(pred)):
        pred_masks.append(pred[i])

    pred_masks = np.reshape(pred_masks, [-1, 256, 256, 1])
    np.save('pred_masks_'+checkpoint_path, pred_masks)

print("Segmentation prediction completed")
    
plt.imshow(np.squeeze(pred_masks[image_id]))
plt.savefig('segmentation_prediction_image_id_'+str(image_id)+ '_' + checkpoint_path + '.png', bbox_inches='tight')

time_elapsed = time.time() - starting_time
print('Test completed, time taken: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




