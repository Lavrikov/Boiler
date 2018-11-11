import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model_CONV3D_VRNN_CUDA import VRNN
import os
from frames_dataset import FramesDataset_Conv3D
from matplotlib import animation
import numpy as np
import visualize


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform(m.weight)
    if type(m) == nn.ConvTranspose3d:
        torch.nn.init.xavier_uniform(m.weight)


# hyperparameters
torch.cuda.set_device(1)
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_name(1))
    print(torch.cuda.get_device_capability(0))
    print(torch.cuda.get_device_capability(1))

conv_filters=10
h_dim = 32*conv_filters
z_dim = 16
n_layers = 1
n_epochs = 10
clip = 3
learning_rate = 5e-4
batch_size = 40
seed = 128
print_every = 100
save_every = 10
frame_x=85
frame_y=48
batch_number=500

# manual seed
torch.manual_seed(seed)

#plt.ion()

basePath = os.path.dirname(os.path.abspath(__file__))
face_dataset = FramesDataset_Conv3D(basePath + '/train/annotations_single_bubble_new_dark.csv', basePath + '/train')
train_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size)

x_dim = face_dataset[0]['frame'].shape[0]

model = VRNN(x_dim, h_dim, z_dim, n_layers, conv_filters, frame_x, frame_y)
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(torch.load('23_10_101.pth'))


#looking for 1 layer weights
visualize.show_weights3d(model.phi_x.state_dict()['0.weight'])

#looking for weights results
conv1=nn.Conv3d(1, conv_filters, 3).cuda()
nn.ReLU()
#load weight of first layer to temporary layer
pretrained_dict=model.phi_x.state_dict()
convstate=conv1.state_dict()
convstate['weight']=model.phi_x.state_dict()['0.weight']
convstate['bias']=model.phi_x.state_dict()['0.bias']
conv1.load_state_dict(convstate)

for batch_idx, data in enumerate(train_loader):
    data = Variable(torch.unsqueeze(data['frame'], 1)).float().cuda()
    data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])
    output=conv1(data)

    print(output[0,0,:,:,0])
    for i in range(0, 10):
        # here i show results
        ax = plt.subplot(3, 4, i + 1)  # coordinates
        plt.tight_layout()
        ax.set_title(i)
        ax.axis('off')
        # print(SummResult)
        # show the statistic matrix
        plt.imshow(output.data[0,i,:,:,0].cpu().numpy(), 'gray')

    plt.show()
    plt.clf()



