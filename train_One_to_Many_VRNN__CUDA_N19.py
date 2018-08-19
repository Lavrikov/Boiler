import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model_VRNN_CUDA import VRNN
import os
from frames_dataset import FramesDataset
from matplotlib import animation
import numpy as np

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
        data = Variable(data['frame'].squeeze().transpose(0, 1)).float().cuda()
        data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        #sample = model.sample(batch_size, 14)
        #print('sample')
        #print(sample)
        #plt.imshow(sample.numpy())
        #plt.pause(1e-6)

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


def init():
    plot.set_data(data[0])
    return [plot]


def update(j):
    plot.set_data(data[j])
    return [plot]


# hyperparameters

h_dim = 100
z_dim = 16
n_layers = 1
n_epochs = 1
clip = 1
learning_rate = 1e-3
batch_size = 128
seed = 128
print_every = 100
save_every = 10

# manual seed
torch.manual_seed(seed)
#plt.ion()

basePath = os.path.dirname(os.path.abspath(__file__))
face_dataset = FramesDataset(basePath + '/train/annotations_single_bubble.csv', basePath + '/train')
x_dim = face_dataset[0]['frame'].shape[1]

print('init model + optimizer + datasets')
train_loader = torch.utils.data.DataLoader(
    face_dataset,
    batch_size=batch_size, pin_memory=True)

print(train_loader.dataset[0])


model = VRNN(x_dim, h_dim, z_dim, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):

    # training + testing
    train(epoch)

    # saving model
    if epoch % save_every == 1:
        fn = 'vrnn_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)


# show generated video
for batch_idx, data in enumerate(train_loader):

    #transforming data
    #data = Variable(data)
    #to remove eventually
    data = Variable(data['frame'].squeeze().transpose(0, 1)).float().cuda()
    data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])

    #output = model.sample(batch_size, face_dataset[0]['frame'].shape[0] )
    output = model.sample_reconstruction(face_dataset[0]['frame'].shape[0], data, 15)
    print('show generated video')
    data = np.empty(batch_size, dtype=object)
    for k in range(batch_size):
        reg = output[k].cpu().numpy()
        reg_original = train_loader.dataset[batch_idx*batch_size + k]['frame'] / 255
        data[k] = np.vstack((reg / (np.max(reg) - np.min(reg)), reg_original))

    fig = plt.figure()
    plot = plt.matshow(data[0], cmap='gray', fignum=0)
    #plt.title(' W/m2' + str(100000 * train_loader.dataset[k]['heat_transfer'] / 255))

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=batch_size, interval=120,
                                       blit=True)
    plt.show()