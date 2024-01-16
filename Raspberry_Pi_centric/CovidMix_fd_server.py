#!/usr/bin/env python
# coding: utf-8

import sys

#python3 CovidMix_fd_server.py 1900 9 2 1

port = int(sys.argv[1]) #1969#10080
users = int(sys.argv[2]) #1 # number of clients
rounds = int(sys.argv[3]) #1
local_epoch = int(sys.argv[4]) #2

print()
print('Server address or host address: 10.200.49.191')
print('Server port: '+str(port))
print('Total clients: '+str(users))
print('Total rounds: '+str(rounds))
print('Total client epochs: '+str(local_epoch))

print()
print('Waiting for all '+str(users)+' clients to join, after all '+str(users)+' clients are joined, we will start...')
print()

import time
start_time = time.time() 



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import random
#import cv2
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

import copy

# will clean later

import os
#import h5py

import socket
import struct
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from threading import Thread
from threading import Lock


import time

#from tqdm import tqdm

import copy


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    sock.sendall(msg)
    return l_send

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg, msglen

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    
    #packet = sock.recv(min(n - len(data), 2048))
    #if packet == b'':
        #raise RuntimeError("socket connection broken")
    
    
    while len(data) < n:
        packet = sock.recv(n - len(data))
        
        if not packet:
            return None
        data += packet
    return data



def average_weights(w, datasize):
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



def run_thread(func, num_user):
    global clientsoclist
    global start_time
    
    thrs = []
    for i in range(num_user):
        conn, addr = s.accept()
        print('Conntected with', addr)
        # append client socket on list
        clientsoclist[i] = conn
        #args = (i, num_user, conn)
        #thread = Thread(target=func, args=args)
        #thrs.append(thread)
        #thread.start()
        thrs.append(Thread(target=func, args=(i, num_user, conn)))
    
    print()
    print("All clients have connected! The process starts now!")
    print()
    
    start_time = time.time()    # store start time
    
    for thread in thrs:
        thread.start()
        
    for thread in thrs:
        thread.join()
    #end_time = time.time()  # store end time
    #print("TrainingTime: {} sec".format(end_time - start_time))


def receive(userid, num_users, conn): #thread for receive clients
    global weight_count
    
    global datasetsize

    #change here
    msg = {
        'rounds': rounds,
        'client_id': userid,
        'local_epoch': local_epoch
    }

    datasize = send_msg(conn, msg)    #send epoch
    
    print("Sent info about round, id, and epoch to client " + str(userid))

    
    total_sendsize_list.append(datasize)
    client_sendsize_list[userid].append(datasize)

    train_dataset_size, datasize = recv_msg(conn)    # get total_batch of train dataset
    
    print("received info about dataset_size and train dataset size from client "+ str(userid) + " , " + str(datasize) + " " + str(train_dataset_size))
    
    total_receivesize_list.append(datasize)
    client_receivesize_list[userid].append(datasize)
    
    
    with lock:
        datasetsize[userid] = train_dataset_size
        weight_count += 1
    
    print("weight_count: " + str(weight_count))
    train(userid, train_dataset_size, num_users, conn)



def train(userid, train_dataset_size, num_users, client_conn):
    global weights_list
    global global_weights
    global weight_count
    global ecg_net
    global val_acc
    
    for r in range(rounds):
        with lock:
            if weight_count == num_users:
                for i, conn in enumerate(clientsoclist):
                    datasize = send_msg(conn, global_weights)
                    print("sending average weight to client "+ str(i))
                    total_sendsize_list.append(datasize)
                    client_sendsize_list[i].append(datasize)
                    train_sendsize_list.append(datasize)
                    weight_count = 0

        print("receiving model weight from client "+ str(userid))
        client_weights, datasize = recv_msg(client_conn)
        print("received model weight from client "+ str(userid))
        total_receivesize_list.append(datasize)
        client_receivesize_list[userid].append(datasize)
        train_receivesize_list.append(datasize)

        weights_list[userid] = client_weights
        print("User" + str(userid) + "'s Round " + str(r + 1) +  " is done")
        with lock:
            weight_count += 1
            if weight_count == num_users:
                #average
                global_weights = average_weights(weights_list, datasetsize)


# # Define Model


import copy
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

#'''
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
#'''

'''
def generate_saliency(inputs, encoder, optimizer):
  #inputs2 = copy(inputs)
  inputs.requires_grad = True
  encoder.eval()

  conv5, conv4, conv3, conv2, conv1, scores = encoder(inputs)

  score_max, score_max_index = torch.max(scores, 1)
  score_max.backward(torch.FloatTensor([1.0]*score_max.shape[0]).to(device))
  saliency, _ = torch.max(inputs.grad.data.abs(),dim=1)
  saliency = inputs.grad.data.abs()
  optimizer.zero_grad()
  encoder.train()

  return saliency
  
'''

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
        bridge = nn.functional.interpolate(bridge, scale_factor=0.125, mode='bilinear', align_corners=True)

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
    
#print("done")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidMix(1).to(device)



## variables


# In[108]:


clientsoclist = [0]*users

start_time = 0
weight_count = 0

global_weights = copy.deepcopy(model.state_dict())

datasetsize = [0]*users
weights_list = [0]*users

lock = Lock()



total_sendsize_list = []
total_receivesize_list = []

client_sendsize_list = [[] for i in range(users)]
client_receivesize_list = [[] for i in range(users)]

train_sendsize_list = [] 
train_receivesize_list = []


#host = socket.gethostbyname(socket.gethostname())
host = socket.gethostbyname("")
#print(host)



s = socket.socket()
s.bind((host, port))
s.listen(128)


run_thread(receive, users)

checkpoint_path = 'model_users_'+str(users)+'_rounds_'+str(rounds)+'_epoch_'+str(local_epoch)+'.pth'

torch.save(global_weights, checkpoint_path)

print('Finished training, Total time taken: {:.0f}m {:.0f}s'.format((time.time() - start_time) // 60, (time.time() - start_time) % 60))




