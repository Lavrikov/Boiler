import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, conv_filters, frame_x, frame_y, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.frame_x=frame_x
        self.frame_y=frame_y

        #feature-extracting transformations, h_dim must compare this output size to work correctly
        self.phi_x = nn.Sequential(
            nn.Conv3d(1, conv_filters, 3),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1)),
            nn.Conv3d(conv_filters, conv_filters * 4, (3,3,1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1)),
            nn.Conv3d(conv_filters * 4, conv_filters * 8, (3,3,1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 1)),
            nn.Conv3d(conv_filters * 8, conv_filters * 16, (3,3,1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1)),
            nn.Conv3d(conv_filters * 16, conv_filters * 32, (3,3,1)),
            nn.ReLU()).cuda()
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()).cuda()

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()).cuda()
        self.enc_mean = nn.Linear(h_dim, z_dim).cuda()
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()).cuda()

        #prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()).cuda()
        self.prior_mean = nn.Linear(h_dim, z_dim).cuda()
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()).cuda()

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim), #create the one dimensional tensor with lengh equal the frame size
            nn.ReLU()).cuda()
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()).cuda()
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential( #create 3 frame from 1
            nn.ConvTranspose3d(in_channels=conv_filters * 32, out_channels=conv_filters * 25, kernel_size=(5, 10, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=conv_filters * 25, out_channels=conv_filters * 16, kernel_size=(5, 10, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=conv_filters * 16, out_channels=conv_filters * 8, kernel_size=(5, 10, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=conv_filters * 8, out_channels=conv_filters * 4, kernel_size=(5, 10, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=conv_filters * 4, out_channels=conv_filters, kernel_size=(5, 10, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=conv_filters, out_channels=int(conv_filters/2), kernel_size=(6, 10, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=int(conv_filters/2), out_channels=int(conv_filters/4), kernel_size=(8, 10, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=int(conv_filters/4), out_channels=int(conv_filters/8), kernel_size=(9, 12, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=int(conv_filters/8), out_channels=1, kernel_size=(8, 11, 3)),
            nn.ReLU()).cuda()


        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias).cuda()


    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0
        sample = torch.zeros(x.size(0), self.frame_y, self.frame_x, 3).cuda()
        phi = self.phi_x(x).squeeze().unsqueeze(1)# calculate conv for whole batch and conjugate dimesions with the h-variable
        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).cuda()

        for t in range(x.size(0)-1,0,-1):

            phi_x_t = phi[t]

            #encoder eq.9 p(z|x)
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            #prior eq.5 p(z)
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder eq.6 p(x|z)
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t.unsqueeze(2).unsqueeze(2).unsqueeze(2))
            dec_std_t = self.dec_std(dec_t)
            sample[t] = dec_mean_t.data

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)


        return kld_loss, nll_loss, sample, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)



    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.frame_y, self.frame_x, 3).cuda()
        h = Variable(torch.randn(self.n_layers, 1, self.h_dim)).cuda()

        for t in range(seq_len - 1, 0, -1):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t.unsqueeze(2).unsqueeze(2).unsqueeze(2))
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t).squeeze().unsqueeze(0)
            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data


        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):# an arbitrary source
        """"using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):

        return  torch.sum((x-theta)**2)


    def _nll_gauss(self, mean, std, x):
        pass