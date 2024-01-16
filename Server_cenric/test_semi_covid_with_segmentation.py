#!/usr/bin/env python
# coding: utf-8

# ## Setup
# 

#CUDA_VISIBLE_DEVICES=0 python3 test_semi_covid.py model.pth 45 log.txt

import sys

checkpoint_path = sys.argv[1]
seg_label = sys.argv[2]
image_id = int(sys.argv[3])


import torch
import numpy as np 
import matplotlib.pyplot as plt
import copy
import gc
from scipy.io import loadmat
from copy import copy
from torch import nn,optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader


def dice_coeff(pred, target):
    smooth = 1.
    #num = pred.size(0)
    #m1 = pred.view(num, -1)  # Flatten
    #m2 = target.view(num, -1)  # Flatten 65536
    #m1 = pred.view(65536)  # Flatten
    #m2 = target.view(65536)  # Flatten 65536
    m1 = pred.reshape(65536)  # Flatten
    m2 = target.reshape(65536)  # Flatten 65536
    #print(m1.shape)
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)



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

print(checkpoint_path)

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

x_test = datamat["x_test"]
y_test = datamat["y_test"]

print('total segmentation train data: ' + str(len(x_train)))
print('total segmentation test data: ' + str(len(x_test)))


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


# to get a segmentation prediction, load the image and run the following code
model.eval() 

test_dataset = Dataset(x_test, y_test, transform = trans)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

pred_masks = []
#pred_masks_torch = []
#labels_torch = []

for inputs, labels in test_loader:
        
     gc.collect()
     torch.cuda.empty_cache()

     inputs = inputs.to(device=device, dtype=torch.float)
     labels = labels.to(device=device, dtype=torch.float)
     pred, _, = model(inputs, optimizer_ft)
     pred = torch.sigmoid(pred)
     pred_numpy = pred.data.cpu().numpy()
      
     #inputs = inputs.data.cpu().numpy()
     for i in range (len(pred_numpy)):
         pred_masks.append(pred_numpy[i])
         #pred_masks_torch.append(pred[i])
         #labels_torch.append(labels[i])


pred_masks = np.reshape(pred_masks, [-1, 256, 256, 1])
x = np.reshape(x_test, [-1, 256, 256, 1])
y = np.reshape(y_test, [-1, 256, 256, 1])

plt.imshow(np.squeeze(y[image_id]))
plt.savefig('images/y_'+str(image_id)+'.png', bbox_inches='tight')

plt.imshow(np.squeeze(x[image_id]))
plt.savefig('images/x_'+str(image_id)+'.png', bbox_inches='tight')
    
plt.imshow(np.squeeze(pred_masks[image_id]))
plt.savefig('images/y_p_'+str(image_id)+'_seg_label_'+seg_label+'.png', bbox_inches='tight')


score = dice_coeff(pred_masks[image_id], y[image_id])

print('dice score of images/y_p_'+str(image_id)+'_seg_label_'+seg_label+'.png: ' + str(score))


dice_score = 0
for image_id in range(x_test.shape[0]):
    
    dice_score += (dice_coeff(pred_masks[image_id], y[image_id]))
   
print('Average dice score: '+ str(float(dice_score/x_test.shape[0])))






