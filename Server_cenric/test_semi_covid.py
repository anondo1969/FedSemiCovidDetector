#!/usr/bin/env python
# coding: utf-8

# ## Setup
# 

#CUDA_VISIBLE_DEVICES=0 python3 test_semi_covid.py model.pth 45 log.txt

import sys

checkpoint_path = sys.argv[1]
#image_id = int(sys.argv[2])
client_or_average=sys.argv[2]
round=sys.argv[3]
#log_file_name = sys.argv[4]

print()
print('-' * 70)
print('Test starts: '+client_or_average+' round: '+round+' ...')




import numpy as np
import copy
from scipy.io import loadmat
from copy import copy
import torch
from torch import nn,optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


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
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidMix(1).to(device)
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))


# ## Load Data

# add the provided classification image to a folder and load it 

data_dir = 'data/covid_xray'

transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=1),
  transforms.Resize((256, 256)), 
  transforms.ToTensor()
])



dataset = ImageFolder(data_dir+'/test', transform = transform)

batch_size = 1

test_loader_class = DataLoader(dataset, batch_size, num_workers=2, pin_memory=True, shuffle = False)



# for classification predictions, run this
model.eval()
correct = 0
total = 0
predictions = np.array([])
y_true = np.array([])
with torch.set_grad_enabled(True):
  
  for i, data in enumerate(test_loader_class):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      _, outC = model(inputs, optimizer_ft)
      _, predicted = torch.max(outC.data, 1)
      predictions = np.append(predictions, predicted.data.cpu().numpy())
      y_true= np.append(y_true, labels.data.cpu().numpy())
      


accuracy = accuracy_score(y_true, predictions)
precision, recall, fscore, support = precision_recall_fscore_support(y_true, predictions, average='binary')

print('accuracy: '+ str(accuracy)+'\nprecision: '+str(precision)+'\nrecall: '+str(recall)+ '\nfscore: ' +str(fscore))#+'\nsupport: '+str(support))







