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
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
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
            nn.Linear(h_dim, h_dim),
            nn.ReLU()).cuda()
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()).cuda()
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()).cuda()

        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias).cuda()


    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).cuda()
        for t in range(x.size(0)-1,0,-1):

            phi_x_t = self.phi_x(x[t])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

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

        return kld_loss, nll_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)


    def sample(self, seq_len, batch_size):

        sample = torch.zeros(seq_len,batch_size, self.x_dim).cuda()
        h_by_row = Variable(torch.zeros(seq_len,self.n_layers, 1, self.h_dim)).cuda()
        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).cuda()

        for t in range(batch_size-1,1,-1):#row number

            for y in range(seq_len):# for video generation, frame number

                h = h_by_row[y]

                #prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                #sampling and reparameterization
                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
                phi_z_t = self.phi_z(z_t)

                #decoder
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean(dec_t)
                #dec_std_t = self.dec_std(dec_t)

                phi_x_t = self.phi_x(dec_mean_t)

                #recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

                sample[y,t] = dec_mean_t.data

                h_by_row[y] = h


        return sample

    def sample2(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim).cuda()
        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).cuda()

        for t in range(seq_len-1,1,-1):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data


        return sample


    def sample2_reverse(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim).cuda()
        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).cuda()

        for t in range(0,seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data


        return sample

    def sample_reconstruction(self, Y_size, x, x_prior):
        # there is the reconstruction a part of image
        # x_prior number of prior rows
        seq_len=x.size(1)
        sample = torch.zeros(seq_len,Y_size, self.x_dim).cuda()

        # first rows initialization
        h_rec = Variable(torch.zeros(self.n_layers, seq_len, self.h_dim)).cuda()
        for t in range(Y_size - 1, Y_size - x_prior - 1, -1):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h_rec[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h_rec[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_rec[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h_rec = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h_rec)
            # sample[:, t] = dec_mean_t.data[:]
            sample[:, t] = x.data[t,:]

        h_by_frame = Variable(torch.zeros(seq_len, self.n_layers, 1, self.h_dim)).cuda()

        h_by_frame[:,0,0]=h_rec[0,:] # initialization of the prior rows state

        for t in range(Y_size-1-x_prior,1,-1):
            # row numbers excluding prior rows

            for i in range(seq_len):# for video generation, frame number

                h = h_by_frame[i]

                #prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                #sampling and reparameterization
                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
                phi_z_t = self.phi_z(z_t)

                #decoder
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean(dec_t)
                #dec_std_t = self.dec_std(dec_t)

                phi_x_t = self.phi_x(dec_mean_t)

                #recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

                sample[i,t] = dec_mean_t.data

                h_by_frame[i] = h


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
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


    def _nll_gauss(self, mean, std, x):
        pass