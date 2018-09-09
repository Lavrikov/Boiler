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


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def train(epoch):
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        #transforming data
        #data = Variable(data)
        #to remove eventually
        data = Variable(torch.unsqueeze(data['frame'],1)).float().cuda()
        data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])
        print('data  ' + str(data.shape))

        conv_1=torch.nn.Conv3d(1, conv_filters, 3).cuda()
        max_pool1=torch.nn.MaxPool3d((2, 2, 1)).cuda()
        conv_2=torch.nn.Conv3d(conv_filters, conv_filters * 4, (3,3,1)).cuda()
        max_pool2= torch.nn.MaxPool3d((2, 2, 1)).cuda()
        conv_3=torch.nn.Conv3d(conv_filters * 4, conv_filters * 8, (3,3,1)).cuda()
        max_pool3=torch.nn.MaxPool3d((1, 2, 1)).cuda()
        conv_4 =torch.nn.Conv3d(conv_filters * 8, conv_filters * 16, (3,3,1)).cuda()
        max_pool4=torch.nn.MaxPool3d((2, 2, 1)).cuda()
        conv_5 =torch.nn.Conv3d(conv_filters * 16, conv_filters * 32, (3,3,1)).cuda()
        max_pool5=torch.nn.MaxPool3d((2, 2, 1)).cuda()
        output=conv_1(data)
        print('1_conv  ' + str(output.shape))
        output=max_pool1(output)
        print('max_poll2  ' + str(output.shape))
        output=conv_2(output)
        print('2_conv  ' + str(output.shape))
        output=max_pool2(output)
        print('max_poll2  ' + str(output.shape))
        output=conv_3(output)
        print('3_conv  ' + str(output.shape))
        output=max_pool3(output)
        print('max_poll3  ' + str(output.shape))
        output=conv_4(output)
        print('4_conv  ' + str(output.shape))
        output=max_pool4(output)
        print('max_poll4  ' + str(output.shape))
        output=conv_5(output)
        print('5_conv  ' + str(output.shape))
        output=max_pool5(output)
        print('max_poll5  ' + str(output.shape))

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        #printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                kld_loss.data[0] / batch_size,
                nll_loss.data[0] / batch_size))

        train_loss += loss.data[0]

    print('====> Epoch: {} Average loss: {:.4f}'.format(
         epoch, train_loss / len(train_loader.dataset)))

    return




# hyperparameters
torch.cuda.set_device(1)
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_name(1))
    print(torch.cuda.get_device_capability(0))
    print(torch.cuda.get_device_capability(1))

h_dim = 100
z_dim = 16
n_layers = 1
n_epochs = 50
clip = 30
learning_rate = 5e-4
batch_size = 120
seed = 128
print_every = 100
save_every = 10
conv_filters=30

# manual seed
torch.manual_seed(seed)

#plt.ion()

basePath = os.path.dirname(os.path.abspath(__file__))
face_dataset = FramesDataset_Conv3D(basePath + '/train/annotations_single_bubble.csv', basePath + '/train')
train_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size)

x_dim = face_dataset[0]['frame'].shape[0]

model = VRNN(x_dim, h_dim, z_dim, n_layers,conv_filters)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#model.load_state_dict(torch.load('20_05.pth'))

generate_epoch=1
data = np.empty(batch_size*n_epochs*generate_epoch, dtype=object)
statistic_generated = np.empty((n_epochs, batch_size*generate_epoch), dtype=float)
statistic_original = np.empty((n_epochs, batch_size*generate_epoch), dtype=float)

for epoch in range(1, n_epochs + 1):

    # training + testing
    train(epoch)

    # saving model
    if epoch % save_every == 1:
        fn = 'vrnn_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)

    # save generated video to memory
    output = model.sample2_reverse(batch_size*generate_epoch)

    # concatenate generated and original video
    for k in range(batch_size*generate_epoch):
        generated = np.resize(output[k].cpu().numpy(),(48,85))
        statistic_generated[epoch-1,k] = generated[10,50]
        original = np.resize(train_loader.dataset[k+(epoch-1)*batch_size*generate_epoch]['frame'] / 255,(48,85))
        statistic_original[epoch-1,k] = original[10,50]
        data[k+batch_size*(epoch-1)] = np.vstack((generated / (np.max(generated) - np.min(generated)), original))


# show generated video
print('show generated video')
fig = plt.figure()
plot = plt.matshow(data[0], cmap='gray', fignum=0)
anim = animation.FuncAnimation(fig, update, init_func=init, frames=batch_size*n_epochs*generate_epoch, interval=30,
                                         blit=True)
plt.show()
