#!/usr/bin/env python
# coding: utf-8

#CUDA_VISIBLE_DEVICES=4 python3 run_clients.py covid_xray jsrt.mat CovidMix_fd_client_zeus.py 1 1 15 .6 .1 10 1 0 accuracy weighted&>> log_user_1_weighted.txt &

import sys
import subprocess
import os
import shutil
import time
import copy

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
import shutil
import gc
from scipy.io import loadmat

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

from collections import namedtuple, defaultdict
from torch.jit.annotations import Optional
#from copy import copy
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

#will clean later

import os
#import h5py

import socket
import struct
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


start_time = time.time()    # store start time

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

def average_weights_simple(w):
    """
    Returns the average of the weights.
    """
    
    
    w_avg = copy.deepcopy(w[0])
    
    

# when client use only one kinds of device

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]

        w_avg[key] = torch.div(w_avg[key], len(w))

# when client use various devices (cpu, gpu) you need to use it instead
#
#     for key, val in w_avg.items():
#         common_device = val.device
#         break
#     for key in w_avg.keys():
#         for i in range(1, len(w)):
#             if common_device == 'cpu':
#                 w_avg[key] += w[i][key].cpu()
#             else:
#                 w_avg[key] += w[i][key].cuda()
#         w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg
    
    
def average_weights_weighted_sample(w, datasize):
    """
    Returns the average of the weights.
    """
        
    for i, data in enumerate(datasize):
        for key in w[i].keys():
            w[i][key] *= float(data)
    
    w_avg = copy.deepcopy(w[0])
    

# when client use only one kinds of device

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

# when client use various devices (cpu, gpu) you need to use it instead
#
#     for key, val in w_avg.items():
#         common_device = val.device
#         break
#     for key in w_avg.keys():
#         for i in range(1, len(w)):
#             if common_device == 'cpu':
#                 w_avg[key] += w[i][key].cpu()
#             else:
#                 w_avg[key] += w[i][key].cuda()
#         w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg
    
def average_weights_std_dev(w, metrics, metric_type):
    
    """
    if the metric is greater than (average of metrics - std dev), then
            the weights are used for averaging, else the weights are discarded.
    """
    
    metrics = np.asarray(metrics)
    std_dev = metrics.std()
    avg = metrics.mean()

    selected_weights = []
		

    for i, weight in enumerate(w):
        
        curr_metric = metrics[i]
                
        if metric_type == "loss":
            criteria = curr_metric <= (avg+std_dev)
        else:
            criteria = curr_metric >= (avg-std_dev)

        if criteria:
            selected_weights.append(weight)
                    
        else:
            if metric_type == "loss":
                print(f"Client {i}: {curr_metric} > {avg+std_dev}")
            else:
                print(f"Client {i}: {curr_metric} < {avg-std_dev}")
                
            
            
    w_avg = copy.deepcopy(selected_weights[0])
    
    
    # when client use only one kinds of device

    for key in w_avg.keys():
        for i in range(1, len(selected_weights)):
            w_avg[key] += selected_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(selected_weights))
            
    return w_avg
    
def average_weights_weighted_metric(w, metrics, metric_type):
       
    """
    for all weights to be averaged based on the user's
    performance metric on their own model with their own validation data by
    multiplying the metric by the weights and averaging using the
    sum of the metrics as the divisor
    """
    
    metrics = np.asarray(metrics)
        
    metric_sum = 0
        
    for i, weight in enumerate(w):
        
        curr_metric = metrics[i]
            
        if metric_type == "loss":
            
            if curr_metric == 0:
                curr_metric = 10**(-6)
            
            curr_metric = 1/curr_metric
                
        metric_sum += curr_metric
            
        for key in w[i].keys():
            w[i][key] *= float(curr_metric)
    
        
    w_avg = copy.deepcopy(w[0])
        
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        
        w_avg[key] = torch.div(w_avg[key], float(metric_sum))
        
            
    return w_avg
    


#---------------------------------------------------------------------------------------------------

#CUDA_VISIBLE_DEVICES=3 python3 run_clients_transfer_fl.py covid_xray_transfer jsrt.mat CovidMix_fd_client_zeus_transfer_fl.py 5 10 5 .6 .1 10 1 0 accuracy std_dev experiment_dir_transfer_users_1_rounds_1_local_epochs_10_accuracy_std_dev_0.6_0.1_15_1_0/best_model_0.pth global_epoch_10&>> log_transfer_fl_10.txt &

#CUDA_VISIBLE_DEVICES=2 python3 run_clients_transfer_fl.py covid_xray_transfer jsrt.mat CovidMix_fd_client_zeus_transfer_fl.py 5 10 5 .6 .1 10 1 0 accuracy std_dev experiment_dir_transfer_users_1_rounds_1_local_epochs_15_accuracy_std_dev_0.6_0.1_15_1_0/best_model_0.pth global_epoch_15&>> log_transfer_fl_15.txt &

#CUDA_VISIBLE_DEVICES=0 python3 run_clients_transfer_fl.py covid_xray_transfer jsrt.mat CovidMix_fd_client_zeus_transfer_fl.py 5 10 5 .6 .1 10 1 0 accuracy std_dev experiment_dir_transfer_users_1_rounds_1_local_epochs_20_accuracy_std_dev_0.6_0.1_15_1_0/best_model_0.pth global_epoch_20&>> log_transfer_fl_20.txt &


all_classification_data_dir = sys.argv[1]#'covid_xray'
all_segmentation_data_dir = sys.argv[2]#'jsrt.mat'
client_run_file_name = sys.argv[3]
users = int(sys.argv[4])
rounds = int(sys.argv[5])
local_epochs = int(sys.argv[6])
classification_label_portion = float(sys.argv[7])
segmentation_label_portion = float(sys.argv[8])
classification_batch_size = int(sys.argv[9])
segmentation_batch_size = int(sys.argv[10])
total_classification_data= int(sys.argv[11])*users
performance_metric = sys.argv[12] # loss or accuracy
weight_averaging = sys.argv[13]  # simple, weighted_sample, std_dev, weighted_metric
transfer_model_path = sys.argv[14]
global_model_source = sys.argv[15]

print('\nHyper-parameters:')
print('-' * 20)
print('users = '+str(users))
print('rounds = ' + str(rounds))
print('local_epochs = ' + str(local_epochs))
print('classification_label_portion = ' + str(classification_label_portion))
print('segmentation_label_portion = ' + str(segmentation_label_portion))
print('classification_batch_size = ' + str(classification_batch_size))
print('segmentation_batch_size = ' + str(segmentation_batch_size))
print('total_classification_data (0 means all the data) = ' + str(total_classification_data))
print('all_classification_data_dir = ' + all_classification_data_dir)
print('all_segmentation_data_dir = ' + all_segmentation_data_dir)
print('client_run_file_name = ' + client_run_file_name)
print('performance_metric = ' + performance_metric)
print('weight_averaging = ' + weight_averaging)
print('transfer_model_path = ' + transfer_model_path)
print('global_model_source = ' + global_model_source)
print('-' * 40)
#print()

exp_dir = 'exp_transfer_fl_dir_'+'_users_'+str(users)+'_'+'rounds_' + str(rounds)+'_'+'local_epochs_' + str(local_epochs)+'_' +performance_metric+'_'+weight_averaging

#'''
if os.path.isdir(exp_dir):
    shutil.rmtree(exp_dir)

os.mkdir(exp_dir)
#'''
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidMix(1).to(device)


for round in range(rounds):
    
    checkpoint_path = exp_dir+'/average_weights_round_'+str(round)+'.pth'
    
    
    #weights = [0]*users
    weights = []
    metrics = []
    
    for client in range(users):
        
        
        
        client_checkpoint_path = exp_dir+'/model_client_'+str(client)+'_round_'+str(round)+'.pth'
        #'''
        client_code_parameters = [
    
        "python3", 
        client_run_file_name, 
        str(users),
        str(local_epochs),
        str(classification_label_portion),
        str(segmentation_label_portion),
        str(classification_batch_size),
        str(segmentation_batch_size),
        str(total_classification_data),
        all_classification_data_dir,
        all_segmentation_data_dir,
        exp_dir,
        str(client),
        str(round),
        performance_metric,
        transfer_model_path
        
        ]

        #print(client_code_parameters)

        run_train_file = subprocess.run(client_code_parameters)

        print()
        print('-' * 70)
        #print("The client training subprocess exit code was (0 means fine, 1 means error): %d" % run_train_file.returncode)
    
        print('Client '+str(client)+ ' of Round '+str(round)+  ' training is finished.')
        print('-' * 70)
        
        #'''
        model.load_state_dict(torch.load(client_checkpoint_path, map_location=device))
        '''
        #do the accuracy here
        
        list_files = subprocess.run(["python3", "test_semi_covid.py", client_checkpoint_path, 'client: '+str(client), str(round)])

        print()
        print('-' * 70)
        #print("The client test subprocess exit code was (0 means fine, 1 means error): %d" % list_files.returncode)
        #print('-' * 70)
        '''
        
        weights.append(model.state_dict())
        #weights[client]=model.state_dict()
        
    #do the averaging here
    
    print('Round '+str(round)+' training for all clients is finished.')
    print('-' * 70)
    
    
    f = open(exp_dir+'/metrics.txt', "r")
    for x in f:
        metrics.append((float(x)))
    f.close()
    print('Performance: '+str(metrics))
    os.remove(exp_dir+'/metrics.txt')

    if weight_averaging == 'simple':
        average_model = average_weights_simple(weights)
        
    elif weight_averaging == 'weighted_sample':
        datasize = [total_classification_data/users for i in range (users)]
        average_model = average_weights_weighted_sample(weights, datasize)
        
    elif weight_averaging == 'std_dev':
        average_model = average_weights_std_dev(weights, metrics, performance_metric)
        
    elif weight_averaging == 'weighted_metric':
        average_model = average_weights_weighted_metric(weights, metrics, performance_metric)
    
    torch.save(average_model, checkpoint_path) #only state dictionary
    
    list_files = subprocess.run(["python3", "test_semi_covid.py", checkpoint_path, 'average after', str(round)])

    #print()
    #print('-' * 30)
    #print("The round end test subprocess exit code was: %d" % list_files.returncode)
    


print('Finished Training, Total time taken: {:.0f}m {:.0f}s'.format((time.time() - start_time) // 60, (time.time() - start_time) % 60))
print('-' * 70)
print()