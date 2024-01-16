#!/usr/bin/env python
# coding: utf-8

import sys

users = int(sys.argv[1])
client_order = int(sys.argv[2])
port = int(sys.argv[3])
host = str(sys.argv[4])
all_data_use = bool(int(sys.argv[5]))
classification_batch_size = int(sys.argv[6])
segmentation_batch_size = int(sys.argv[7])
classification_label_portion = float(sys.argv[8])
segmentation_label_portion = float(sys.argv[9])

all_classification_data_dir = 'covid_xray'
all_segmentation_data_dir = 'jsrt.mat'

print('users = '+str(users))
print('client_order = ' + str(client_order))
print('port = ' + str(port))
print('host = ' + host)
print('all_data_use = ' + str(all_data_use))
print('classification_label_portion = ' + str(classification_label_portion))
print('segmentation_label_portion = ' + str(segmentation_label_portion))
print('classification_batch_size = ' + str(classification_batch_size))
print('segmentation_batch_size = ' + str(segmentation_batch_size))
print('all_classification_data_dir = ' + all_classification_data_dir)
print('all_segmentation_data_dir = ' + all_segmentation_data_dir)

max_recv = 100000
PARAMETER_MAX = 10 #augmentation parameter


import time

start_time = time.time()    # store start time

#print()
#print("timer starts!")
#print()


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

# for image
# import matplotlib.pyplot as plt
# import numpy as np

import time

#from tqdm import tqdm

from gpiozero import CPUTemperature


def getFreeDescription():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 1:
            return (line.split()[0:7])


def getFree():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 2:
            return (line.split()[0:7])



def printPerformance():
    cpu = CPUTemperature()
    
    print()
    print('Current system status:')
    print()
    print("temperature: " + str(cpu.temperature))

    description = getFreeDescription()
    mem = getFree()

    print(description[0] + " : " + mem[1])
    print(description[1] + " : " + mem[2])
    print(description[2] + " : " + mem[3])
    print(description[3] + " : " + mem[4])
    print(description[4] + " : " + mem[5])
    print(description[5] + " : " + mem[6])

    print()



printPerformance()



def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)



def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg



def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data



# Prepare Dataset and DataLoader

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


# Augmentations


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    color = (0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def augment_pool():
    augs = [
            (AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)
            ]
    return augs

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        # img = CutoutAbs(img, 128) 
        return img


# Define Model


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



def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def kl_divergence_class(outC, outStrong):
  p = F.softmax(outC, dim = 1)
  log_p = F.log_softmax(outC, dim = 1)
  log_q = F.log_softmax(outStrong, dim = 1)
  kl = p * (log_p - log_q)
  
  return kl.mean()

def kl_divergence_seg(outSeg, outSegUnlabeled):
  p = F.softmax(outSeg, dim = 1)
  log_p = F.log_softmax(outSeg, dim = 1)
  log_q = F.log_softmax(outSegUnlabeled, dim = 1)
  kl = p * (log_p - log_q)
  
  return kl.mean()


def calc_loss(outSeg, target, outSegUnlabeled, outC, labels, outWeak, outStrong, metrics, ssl_weight = 0.25, threshold = 0.7, kl_weight = 0.01, dice_weight = 5):

    criterion = nn.CrossEntropyLoss()
    
    
    predSeg = torch.sigmoid(outSeg)

    dice = dice_loss(predSeg, target)

    lossClassifier = criterion(outC, labels)

    probsWeak = torch.softmax(outWeak, dim=1)
    max_probs, psuedoLabels = torch.max(probsWeak, dim=1)
    mask = max_probs.ge(threshold).float()

    lossUnLabeled = (F.cross_entropy(outStrong, psuedoLabels,
                              reduction='none') * mask).mean()

    kl_class = kl_divergence_class(outC, outStrong)
    kl_seg = kl_divergence_seg(outSeg, outSegUnlabeled)

    # do KL only with segmentation for now
    loss = lossClassifier + dice * dice_weight + (lossUnLabeled * ssl_weight) + (kl_seg * kl_weight)

    metrics['lossClassifier'] += lossClassifier.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss, metrics

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))



def train_model(model, optimizer, scheduler, dataloaders, dataloadersClassifier, num_epochs=25, checkpoint_path='model.pth'):
    best_loss = 1e10
    model.train()
    best_valid_accuracy = -1#0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        metrics = defaultdict(float)
        epoch_samples = 0
        total_train = 0
        correct_train = 0
        trainloader = zip(cycle(dataloaders["train"]), cycle(dataloaders["unlabeled"]), cycle(dataloadersClassifier["train"]), dataloadersClassifier["weak"], dataloadersClassifier["strong"]) # added cycling
            
        for i, (dataSeg, dataSegUnlabeled, data, dataWeak, dataStrong) in enumerate(trainloader):
            gc.collect()
            torch.cuda.empty_cache()

            inputs, masks = dataSeg
            inputs, masks = inputs.to(device=device, dtype=torch.float), masks.to(device=device, dtype=torch.float)

            inputsUnlabeled, masksUnlabeled = dataSegUnlabeled
            inputsUnlabeled, masksUnlabeled = inputsUnlabeled.to(device=device, dtype=torch.float), masksUnlabeled.to(device=device, dtype=torch.float)

            inputsClass, labels = data
            inputsClass, labels = inputsClass.to(device), labels.to(device)

            inputsWeak, weakLabelUnused = dataWeak
            inputsWeak, weakLabelUnused = inputsWeak.to(device), weakLabelUnused.to(device)

            inputsStrong, strongLabelUnused = dataStrong
            inputsStrong, strongLabelUnused = inputsStrong.to(device), strongLabelUnused.to(device)
                
            inputsAll = torch.cat((inputs, inputsUnlabeled, inputsClass, inputsWeak, inputsStrong))
            batch_size_seg = inputs.shape[0]
            batch_size_seg_unlabeled = inputsUnlabeled.shape[0] + batch_size_seg
            batch_size_class = inputsClass.shape[0] + batch_size_seg_unlabeled
            batch_size_weak = inputsWeak.shape[0] + batch_size_class
            batch_size_strong = inputsStrong.shape[0] + batch_size_weak

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):

            # backward + optimize only if in training phase
            
                outSegAll, outClassAll = model(inputsAll, optimizer)

                outSeg = outSegAll[:batch_size_seg]
                outSegUnlabeled = outSegAll[batch_size_seg:batch_size_seg_unlabeled]
                outC = outClassAll[batch_size_seg_unlabeled:batch_size_class]
                outWeak = outClassAll[batch_size_class:batch_size_weak]
                outStrong = outClassAll[batch_size_weak:batch_size_strong]

                loss, metrics = calc_loss(outSeg, masks, outSegUnlabeled, outC, labels, outWeak, outStrong, metrics)

                loss.backward()
                optimizer.step()
                
                '''
                if ((i+1) % 100 == 0):
                          
                    
                    model.eval()
                    # accuracy
                    _, predicted = torch.max(outC, 1)
                    total_train += labels.size(0)
                    correct_train += predicted.eq(labels.data).sum().item()
                    train_accuracy = 100 * correct_train / total_train
                    model.train()

                    print(str(total_train)+' Images trained, Training accuracy: '+ str(train_accuracy))
                '''

        # statistics
        epoch_samples += inputs.size(0)

        print_metrics(metrics, epoch_samples, 'train')
        epoch_loss = metrics['loss'] / epoch_samples

        scheduler.step()
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])

            
            
        model.eval()  
        # validation accuracy
        valid_correct = 0
        valid_total = 0
        valid_predictions = np.array([])
        with torch.set_grad_enabled(True):
          for i, valid_data in enumerate(dataloadersClassifier['val']):
              valid_inputs, valid_labels = valid_data
              valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
              _, valid_outC = model(valid_inputs, optimizer_ft)
              _, valid_predicted = torch.max(valid_outC.data, 1)
              valid_predictions = np.append(valid_predictions, valid_predicted.data.cpu().numpy())
              valid_total += valid_labels.size(0)
              valid_correct += (valid_predicted == valid_labels).sum().item()

        valid_accuracy = (valid_correct / valid_total)  

        print('Classification validation accuracy: %d %%' % (100 * valid_correct / valid_total))
            
        # save the model weights
        if valid_accuracy > best_valid_accuracy:
        #if epoch_loss < best_loss:
            #print(f"saving best model to {checkpoint_path}")
            print("saving best model....")
            #best_loss = epoch_loss
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), checkpoint_path)
        
        model.train()
        
        time_elapsed = time.time() - since
        print('Epoch finished, time taken: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()
    #print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model


# load classification dataset

def load_classification_dataset(data_dir, batch_size = 1, label_size = 2):

    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ])

    transform_weak = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.Resize((256, 256)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(size=256, padding=int(256*0.125), padding_mode='reflect'),
      transforms.ToTensor()
    ])

    transform_strong = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.Resize((256, 256)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(size=256, padding=int(32*0.125), padding_mode='reflect'),
      RandAugmentMC(n=2, m=10),
      transforms.ToTensor()
    ])

    #based on client id and total clients, do it later
    dataset = ImageFolder(data_dir, transform = transform)
    
    #print(len(dataset))
    
    
    # create subset

    val_size = round(len(dataset) * 0.1)
    unlabeled_size = len(dataset) - label_size - val_size
    labeled_ds, val_ds, unlabeled_ds = random_split(dataset, [label_size, val_size, unlabeled_size])

    # apply augmentations

    labeled_ds = copy(labeled_ds)
    labeled_ds.dataset = copy(dataset)

    unlabeled_ds_weak = copy(unlabeled_ds)
    # unlabeled_ds_weak = copy(labeled_ds)
    unlabeled_ds_weak.dataset = copy(dataset)

    unlabeled_ds_strong = copy(unlabeled_ds)
    # unlabeled_ds_strong = copy(labeled_ds)
    unlabeled_ds_strong.dataset = copy(dataset)

    #create augmentations
    labeled_ds.dataset.transform = transform_weak
    unlabeled_ds_weak.dataset.transform = transform_weak
    unlabeled_ds_strong.dataset.transform = transform_strong

    #shuffle=True?
    dataloadersClassifier = {
      'train': DataLoader(labeled_ds, batch_size, shuffle=False, num_workers=2, drop_last = True, pin_memory=True),
      'val': DataLoader(val_ds, 1, num_workers=2, drop_last = True, pin_memory=True),
      'weak': DataLoader(unlabeled_ds_weak, batch_size, shuffle=False, num_workers=2, drop_last = True, pin_memory=True),
      'strong': DataLoader(unlabeled_ds_strong, batch_size, shuffle=False, num_workers=2, drop_last = True, pin_memory=True)
    }
    
    return dataloadersClassifier


def create_client_classification_images(data_dir, users=1, client_id=0, all_data_use=False, classification_label_portion=0.6):

    client_train_dir = data_dir+'/client_train/'
    
    if all_data_use:
        total_train_dir = data_dir+'/train/'
    else:
        #print('enter here 1')
        total_train_dir = data_dir+'/train_some/'

    if os.path.isdir(client_train_dir):
        shutil.rmtree(client_train_dir)
    
    os.mkdir(client_train_dir)
    for class_name in ['COVID', 'Normal']:
        
        os.mkdir(client_train_dir+class_name)
        
    total_count = 0
        
    for class_name in ['COVID', 'Normal']:
    
    
        dirListing = os.listdir(total_train_dir+class_name)
    
        file_names = []
        file_numbers = []
        
        for item in dirListing:
            if ".png" in item:
                file_names.append(item)
                #print(item)
                file_numbers.append(int(item.strip(class_name+'-.png')) -1)
           
    
        num_traindata = len(file_numbers) // users
        
        
    
        
    
        for file_index in range(len(file_numbers)):
        
            if file_numbers[file_index] >= num_traindata * client_id and file_numbers[file_index] < num_traindata * (client_id + 1):
            
                selected_file_location = file_names[file_index]
            
                shutil.copyfile(total_train_dir+class_name+'/'+file_names[file_index], client_train_dir+class_name+'/'+file_names[file_index])
            
                total_count+=1
            
    
        #print(num_traindata)
        #print(total_count)
    print('total classification train data: ' + str(total_count))
    
    classification_label_size = int((num_traindata*2)*classification_label_portion)
        
    #print(classification_label_size)
        
    return client_train_dir, classification_label_size
        

# load segmentation dataset

def generate_segmentation_dataset(data_dir, client_id=0, users=1, all_data_use=False, segmentation_label_portion=0.10):

    datamat = loadmat(data_dir)

    #datamat.keys()

    x_train = datamat["x_train"]
    y_train = datamat["y_train"]
    x_val = datamat["x_val"]
    y_val = datamat["y_val"]
    x_test = datamat["x_test"]
    y_test = datamat["y_test"]
    
    #print(len(x_train))
    
    x_train = np.concatenate((x_train, x_val), axis=0) 
    y_train = np.concatenate((y_train, y_val), axis=0)
    
    num_traindata = len(x_train) // users
    
    x_train = x_train[num_traindata * client_id : num_traindata * (client_id + 1)]
    y_train = y_train[num_traindata * client_id : num_traindata * (client_id + 1)]
    
    #print(num_traindata)
    #print(len(x_train))
    #print(len(y_train))
    

    x_train = np.array(x_train).reshape(len(x_train),256, 256)
    y_train = y_train[:,:,:,0] + y_train[:,:,:,1]
    y_train = np.array(y_train).reshape(len(y_train),1, 256, 256)

    #x_val = np.array(x_val).reshape(len(x_val),256, 256)
    #y_val = y_val[:,:,:,0] + y_val[:,:,:,1]
    #y_val = np.array(y_val).reshape(len(y_val),1, 256, 256)

    x_test = np.array(x_test).reshape(len(x_test),256, 256)
    y_test = y_test[:,:,:,0] + y_test[:,:,:,1]
    y_test = np.array(y_test).reshape(len(y_test),1, 256, 256)


    #temp, for testing quickly
    if all_data_use==False:
        
        x_train = x_train[0:40]
        y_train = y_train[0:40]
        x_val = x_val[0:5]
        y_val = y_val[0:5]
        
        num_traindata = len(x_train) // users
        
    print()
    print('total segmentation train data: ' + str(num_traindata))
    

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5), (0.5))
    ])

    transformed_dataset = {
        'train_set': Dataset(x_train, y_train, transform = trans),
        #'val_set': Dataset(x_val, y_val, transform = trans),
        'test_set': Dataset(x_test, y_test, transform = trans)
    }
    
    segmentation_label_size = int(num_traindata*segmentation_label_portion)

    #print(segmentation_label_size)

    return transformed_dataset, segmentation_label_size


def load_segmentation_dataset(train_set, batch_size = 1, label_size = 2):
    
    # print(len(train_set))
    #based on client id and total clients, do it later
    unlabeled_size = len(train_set) - label_size
    labeled_ds, unlabeled_ds = random_split(train_set, [label_size, unlabeled_size])

    dataloaders = {
      'train': DataLoader(labeled_ds, batch_size=batch_size, shuffle=True, drop_last = True, num_workers=2),
      'unlabeled': DataLoader(unlabeled_ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 2)
      #'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last = True, num_workers=2)
    }
    return dataloaders
    
#---------------------------------------------------------------------------------------------------------------------------------------------------

s = socket.socket()
s.connect((host, port))


# Training


        

print()
print('Receiving instruction from the server....')
timestamp = time.time()

msg = recv_msg(s)

print('Received, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))
print()

rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidMix(1).to(device)


# update weights from server
# train

client_specific_classification_data_dir, classification_label_size = create_client_classification_images(all_classification_data_dir, users=users, client_id=client_id, all_data_use=all_data_use, classification_label_portion=classification_label_portion)

print()
print('Sending the total number of classification data to the server....')
timestamp = time.time()

send_msg(s, len(ImageFolder(client_specific_classification_data_dir)))

print('Sent, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))


client_specific_transformed_dataset, segmentation_label_size = generate_segmentation_dataset(all_segmentation_data_dir, client_id=client_id, users=users, all_data_use=all_data_use, segmentation_label_portion=segmentation_label_portion)

checkpoint_path = 'model_epoch_'+str(local_epochs)+'_seg_label_'+str(segmentation_label_size)+'_seg_batch_'+str(segmentation_batch_size)+'_class_label_'+str(classification_label_size)+'_class_batch_'+str(classification_batch_size)+'.pth'

print()
print('Model path: ' + checkpoint_path)

dataloaders = load_segmentation_dataset(
        client_specific_transformed_dataset['train_set'], 
        batch_size = segmentation_batch_size, label_size = segmentation_label_size)

dataloadersClassifier = load_classification_dataset(
        client_specific_classification_data_dir, 
        batch_size = classification_batch_size, label_size = classification_label_size)

for r in range(rounds):  # loop over the dataset multiple times

    print()
    print('Rounds {}/{}'.format(r+1, rounds))
    print('=' * 10)
    print()
    
    print('Receiving average weights from the server....')
    timestamp = time.time()
    
    weights = recv_msg(s)
    
    print('Received, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))
    print()
    
    model.load_state_dict(weights)
    model.eval()

    
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, dataloadersClassifier, num_epochs = local_epochs, checkpoint_path=checkpoint_path)
    
    model.eval()
    msg = model.state_dict()
    
    #print('trying to send the weights')
    
    print('Sending best model weights to the server....')
    timestamp = time.time()
    
    send_msg(s, msg)
    

    print('Sent, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))
    print()
    
    #print('round ' + str(r+1) + ' ended')
    
print()
print('Finished Training, Total time taken: {:.0f}m {:.0f}s'.format((time.time() - start_time) // 60, (time.time() - start_time) % 60))

printPerformance()




