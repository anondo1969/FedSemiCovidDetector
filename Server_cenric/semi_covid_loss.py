#!/usr/bin/env python
# coding: utf-8

# ## Setup
# 

#CUDA_VISIBLE_DEVICES=3 python3 semi_covid_loss.py 20 1000 10 210 10 22 model_seg_label_210.pth&>> logs/log_seg_label_210.txt &

#CUDA_VISIBLE_DEVICES=0 python3 semi_covid_loss.py 20 1000 10 200 10 22 model_seg_label_200.pth&>> logs/log_seg_label_200.txt &

import sys
import numpy

epochs = int(sys.argv[1]) #3
class_label = int(sys.argv[2]) #1000
class_batch = int(sys.argv[3]) #10
seg_label = int(sys.argv[4]) #10
seg_batch = int(sys.argv[5]) #10
image_id = int(sys.argv[6]) #76

checkpoint_path = sys.argv[7]

checkpoint_path = 'experiments/'+checkpoint_path

print(checkpoint_path)

print('total classification labeled train data: ' + str(class_label))
print('total segmentation labeled train data: ' + str(seg_label))
print('total classification batch: ' + str(class_batch))
print('total segmentation batch: ' + str(seg_batch))


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

def dice(result, reference):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    
    intersection = numpy.count_nonzero(result & reference)
    
    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc


starting_time = time.time()

# ## Load Data


# load segmentation dataset
datamat = loadmat('data/jsrt.mat')

#datamat.keys()


x_all = np.concatenate((datamat["x_train"], datamat["x_test"], datamat["x_val"]), axis=0)
y_all = np.concatenate((datamat["y_train"], datamat["y_test"], datamat["y_val"]), axis=0)


train_data_length = int(len(x_all) * 0.9)

x_train = x_all[:train_data_length]
y_train = y_all[:train_data_length]
x_test = x_all[train_data_length:]
y_test = y_all[train_data_length:]


print('total segmentation train data: ' + str(len(x_train)))
print('total segmentation test data: ' + str(len(x_test)))


x_train = np.array(x_train).reshape(len(x_train),256, 256)
y_train = y_train[:,:,:,0] + y_train[:,:,:,1]
y_train = np.array(y_train).reshape(len(y_train),1, 256, 256)

#x_val = np.array(x_val).reshape(len(x_val),256, 256)
#y_val = y_val[:,:,:,0] + y_val[:,:,:,1]
#y_val = np.array(y_val).reshape(len(y_val),1, 256, 256)

x_test = np.array(x_test).reshape(len(x_test),256, 256)
y_test = y_test[:,:,:,0] + y_test[:,:,:,1]
y_test = np.array(y_test).reshape(len(y_test),1, 256, 256)


# ## Prepare Dataset and DataLoader


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

train_set = Dataset(x_train, y_train, transform = trans)
#val_set = Dataset(x_val, y_val, transform = trans)

# print(len(train_set))

labeled_size = seg_label
unlabeled_size = len(train_set) - labeled_size
labeled_ds, unlabeled_ds = random_split(train_set, [labeled_size, unlabeled_size])

batch_size = seg_batch

dataloaders = {
  'train': DataLoader(labeled_ds, batch_size=batch_size, shuffle=True, drop_last = True, num_workers=2),
  'unlabeled': DataLoader(unlabeled_ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 2)
  #'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last = True, num_workers=2)
}

print("Segmentation data loading completed")

# Augmentations
PARAMETER_MAX = 10


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
    

# load classification dataset
data_dir = 'data/covid_xray_old'

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

dataset = ImageFolder(data_dir+'/train', 
                      transform = transform)
                      
                      
print('total data: ' + str(len(dataset)))
#print(len(dataset))
# create subset
labeled_size = class_label
#val_size = round(len(dataset) * 0.1)
unlabeled_size = len(dataset) - labeled_size# - val_size
#labeled_ds, val_ds, unlabeled_ds = random_split(dataset, [labeled_size, val_size, unlabeled_size])
labeled_ds, unlabeled_ds = random_split(dataset, [labeled_size, unlabeled_size])
# apply augmentations



print('total classification train data (labeled+unlabeled): ' + str(len(dataset)))
print('total classification unlabeled train data: ' + str(unlabeled_size))
#print('total classification validation data: ' + str(val_size))

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

batch_size = class_batch

dataloadersClassifier = {
  'train': DataLoader(labeled_ds, batch_size, shuffle=False, num_workers=2, drop_last = True, pin_memory=True),
  #'val': DataLoader(val_ds, batch_size, num_workers=2, drop_last = True, pin_memory=True),
  'weak': DataLoader(unlabeled_ds_weak, batch_size, shuffle=False, num_workers=2, drop_last = True, pin_memory=True),
  'strong': DataLoader(unlabeled_ds_strong, batch_size, shuffle=False, num_workers=2, drop_last = True, pin_memory=True)
}

print("Classification data loading completed")

data_dir = 'data/covid_xray_old'

transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=1),
  transforms.Resize((256, 256)), 
  transforms.ToTensor()
])

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
    


# # Define the main training loop


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

criterion = nn.CrossEntropyLoss()

def calc_loss(outSeg, target, outSegUnlabeled, outC, labels, outWeak, outStrong, metrics, ssl_weight = 0.25, threshold = 0.7, kl_weight = 0.01, dice_weight = 5):

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

def train_model(model, optimizer, scheduler, num_epochs=25, checkpoint_path='model.pth'):
    best_loss = 1e10
    model.train()
    best_valid_accuracy = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        metrics = defaultdict(float)
        epoch_samples = 0
        total_train = 0
        correct_train = 0
        trainloader = zip(cycle(dataloaders["train"]), cycle(dataloaders["unlabeled"]), cycle(dataloadersClassifier["train"]), dataloadersClassifier["weak"], dataloadersClassifier["strong"]) # added cycling
        
        countt = 0
            
        for i, (dataSeg, dataSegUnlabeled, data, dataWeak, dataStrong) in enumerate(trainloader):
            
            countt+=1
            
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
            
            '''
            if countt == 1:
                print('i: '+str(i))
                print('inputs: ' + str(len(inputs)))
                print('inputsUnlabeled: ' + str(len(inputsUnlabeled)))
                print('inputsClass: ' + str(len(inputsClass)))
                print('inputsWeak: ' + str(len(inputsWeak)))
                print('inputsStrong: ' + str(len(inputsStrong)))
                print('inputsAll: ' + str(len(inputsAll)))
            '''

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
        ''' 
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
            print(f"saving best model to {checkpoint_path}")
            #best_loss = epoch_loss
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), checkpoint_path)
        '''
        if epoch_loss < best_loss:
            print(f"saving best model to {checkpoint_path}")
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
                          
        model.train()
        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #print("Count: "+ str(countt))
        #print("Counti: "+ str(i))
    #print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model


# # Training


# gc.collect()
# torch.cuda.empty_cache()

# uncomment the following code for training
#'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidMix(1).to(device)
  
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs = epochs, checkpoint_path=checkpoint_path)
#'''


# # Testing
# To test the code, load the provided pth file after instantiating the model. Then, run the test code and receive a prediction.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidMix(1).to(device)
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))


# add the provided classification image to a folder and load it 

data_dir = 'data/covid_xray_old'

transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=1),
  transforms.Resize((256, 256)), 
  transforms.ToTensor()
])

#print(len(os.listdir(data_dir + "/test/COVID")) + len(os.listdir(data_dir + "/test/NORMAL")))

dataset = ImageFolder(data_dir+'/test', 
                      transform = transform)

batch_size = 1

test_loader_class = DataLoader(dataset, batch_size, num_workers=2, pin_memory=True, shuffle = False)

print("Classification test data loading completed")


# for classification predictions, run this cell
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

print('Classification test accuracy: %d %%' % (100 * correct / total))
#print(predictions)

#'''
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
y_test = np.reshape(y_test, [-1, 256, 256, 1])
#np.save('pred_masks_'+checkpoint_path, pred_masks)


# to view the predicted mask, run the following
print("Segmentation prediction completed")
# to view the predicted mask, run the following
plt.imshow(np.squeeze(pred_masks[image_id]))
plt.savefig('images/segmentation_prediction_image_id_'+str(image_id) +'_seg_label_'+str(seg_label)+'.png', bbox_inches='tight')

plt.imshow(np.squeeze(y_test[image_id]))
plt.savefig('images/segmentation_original_image_id_'+str(image_id) +'_seg_label_'+str(seg_label)+'.png', bbox_inches='tight')

dice_score=dice(pred_masks[image_id], y_test[image_id])

print(dice_score)

dice_score = 0
for image_id in range(x_test.shape[0]):
    #print(dice_score)
    dice_score += dice(pred_masks[image_id], y_test[image_id])
    
print(dice_score)
print (float(dice_score/x_test.shape[0]))

time_elapsed = time.time() - starting_time
print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#'''



