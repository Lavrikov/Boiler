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
from frames_dataset import FramesDataset_Mono
from matplotlib import animation
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def train_at_all(epoch, data_all):
    train_loss = 0

    #forward + backward + optimize
    optimizer.zero_grad()
    kld_loss, nll_loss, _, _ = model(data_all)
    loss = kld_loss + nll_loss
    loss.backward()
    optimizer.step()

    #grad norm clipping, only in pytorch version >= 1.10
    nn.utils.clip_grad_norm(model.parameters(), clip)

    train_loss += loss.data[0]

    print('====> Epoch: {} Average loss: {:.4f}'.format(
         epoch, train_loss / len(train_loader.dataset)))

    return

def train(epoch):
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        #transforming data
        #data = Variable(data)
        #to remove eventually
        data = Variable(torch.unsqueeze(data['frame'],1)).float().cuda()
        data = (data - data.min().item()) / (data.max().item() - data.min().item())

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

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
                kld_loss.item() / batch_size,
                nll_loss.item() / batch_size))

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
         epoch, train_loss / len(train_loader.dataset)))

    return


def init():
    plot.set_data(data[0])
    return [plot]


def update(j):
    plt.title(int(j/batch_size))
    plot.set_data(data[j])
    return [plot]


# hyperparameters

h_dim = 1000
z_dim = 32
n_layers = 1
n_epochs = 10
clip = 30
learning_rate = 1e-4
batch_size = 120
seed = 128
print_every = 100
save_every = 10
cross_section_value=30

# manual seed
torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.set_device(0)
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
    print(torch.cuda.get_device_capability(0))
#plt.ion()

basePath = os.path.dirname(os.path.abspath(__file__))
face_dataset = FramesDataset_Mono(basePath + '/train/annotations_X1_bubble.csv', basePath + '/train')
train_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size)

x_dim = face_dataset[0]['frame'].shape[0]

model = VRNN(x_dim, h_dim, z_dim, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#model.load_state_dict(torch.load('21_06.pth'))

generate_epoch=10
data = np.empty(batch_size*n_epochs*generate_epoch, dtype=object)
data_X_long = np.empty(batch_size*n_epochs*generate_epoch, dtype=object)
generated_full=np.empty(( batch_size*generate_epoch,48,85), dtype=float)
generated_shifted=np.empty((batch_size*generate_epoch,48,85), dtype=float)
generated_shifted_mask=np.ones((48,85), dtype=float)
generated_shifted_floor=np.zeros((batch_size*generate_epoch,48*85), dtype=float)
generated_shifted_floor_mask=np.zeros((48*85), dtype=float)


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

    # changing part of following steps generation picture with part of old picture
    for k in range(batch_size*generate_epoch):
        generated = np.resize(output[k].cpu().numpy(),(48,85))
        generated_full[k]=generated
        generated_shifted[k, :, 0:cross_section_value] = generated_full[k, :, -cross_section_value - 1:-1]
        generated_shifted_floor[k] = np.resize(generated_shifted[k], 48 * 85)

    generated_shifted_mask[:, 0:cross_section_value] = 0
    generated_shifted_floor_mask = np.resize(generated_shifted_mask, 48 * 85)

    output = model.sample3(batch_size*generate_epoch, Variable(torch.from_numpy(generated_shifted_floor).cuda().float()),Variable(torch.from_numpy(generated_shifted_floor_mask).cuda().float()))

    # concatenate generated and original video
    for k in range(batch_size*generate_epoch):
        generated2 = np.resize(output[k].cpu().numpy(),(48,85))
        concatenated_generated=np.hstack((generated_full[k][:,0:85-cross_section_value],generated2))
        data[k+batch_size*(epoch-1)] = concatenated_generated


# show generated video
print('show generated video')
fig = plt.figure()
plot = plt.matshow(data[0], cmap='gray', fignum=0)
anim = animation.FuncAnimation(fig, update, init_func=init, frames=batch_size*n_epochs*generate_epoch, interval=30,
                                         blit=True)
plt.show()

