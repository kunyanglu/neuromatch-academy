import torch
import random

import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from pytorch_pretrained_biggan import BigGAN
from pytorch_pretrained_biggan import one_hot_from_names
from pytorch_pretrained_biggan import truncated_noise_sample

from tqdm.notebook import tqdm, trange

SEED = 2021

class BiasLayer(nn.Module):

    def __init__(self, shape):
        super(BiasLayer, self).__init__()
        init_bias = torch.zeros(shape)
        self.bias = nn.Parameter(init_bias, requires_grad=True)

    def forward(self, x):
        return x + self.bias

class ConvAutoEncoder(nn.Module):

    def __init__(self, x_dim, h_dim, n_filters=32, filter_size=5):
        super().__init__()
        channels, height, widths = x_dim

        self.enc_bias = BiasLayer(x_dim)
        self.enc_conv_1 = nn.Conv2d(channels, n_filters, filter_size)
        conv_1_shape = cout(x_dim, self.enc_conv_1)
        self.enc_conv_2 = nn.Conv2d(n_filters, n_filters, filter_size)
        conv_2_shape = cout(conv_1_shape, self.enc_conv_2)

        self.enc_flatten = nn.Flatten()
        flat_after_conv = conv_2_shape[0] * conv_2_shape[1] * conv_2_shape[2]
        self.enc_lin = nn.Linear(flat_after_conv, h_dim)

        self.dec_lin = nn.Linear(h_dim, flat_after_conv)
        self.dec_unflatten = nn.Unflatten(dim=-1, unflattened_size=conv_2_shape)
        self.dec_deconv_1 = nn.ConvTranspose2d(n_filters, n_filters, filter_size)
        self.dec_deconv_2 = nn.ConvTranspose2d(n_filters, channels, filter_size)

        self.dec_bias = BiasLayer(x_dim)

    def encode(self, x):
        s = self.enc_bias(x)
        s = F.relu(self.enc_conv_1(s))
        s = F.relu(self.enc_conv_2(s))
        s = self.enc_flatten(s)
        h = self.enc_lin(s)
        return h

    def decode(self, h):
        s = F.relu(self.dec_lin(h))
        s = self.dec_unflatten(s)
        s = F.relu(self.dec_deconv_1(s))
        s = self.dec_deconv_2(s)
        x_prime = self.dec_bias(s)
        return x_prime
    
    def forward(self, x):
        return self.decode(self.encode(x))

def train_autoencoder(autoencoder, dataset, device, epochs=20, batch_size=250,
                      seed=0):
    autoencoder.to(device)
    optim = torch.optim.Adam(autoencoder.parameters(),
                           lr=1e-3,
                           weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)
    loader = DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      num_workers=2,
                      worker_init_fn=seed_worker,
                      generator=g_seed)

    mse_loss = torch.zeros(epochs * len(dataset) // batch_size, device=device)
    i = 0

    for epoch in trange(epochs, desc='Epoch'):
        
        for im_batch, _ in loader:
            
            im_batch = im_batch.to(device)
            optim.zero_grad()
            reconstruction = autoencoder(im_batch)
            # write the loss calculation
            loss = loss_fn(reconstruction.view(batch_size, -1),
                           target=im_batch.view(batch_size, -1))
            loss.backward()
            optim.step()

            mse_loss[i] = loss.detach()
            i += 1
    # After training completes, make sure the model is on CPU so we can easily
    # do more visualizations and demos.
    autoencoder.to('cpu')
    
    return mse_loss.cpu()

def load_data():
    ### Load MNIST ###
    mnist = datasets.MNIST('./mnist/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    mnist_val = datasets.MNIST('./mnist/',
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True)
    return mnist, mnist_val

def get_data(mnist, mnist_val):

    my_dataset_name = "MNIST"
    my_dataset = mnist
    my_valset = mnist_val
    my_dataset_shape = (1, 28, 28)
    my_dataset_size = 28 * 28
    
    return my_dataset, my_dataset_name, my_dataset_shape, my_dataset_size, my_valset

if __name__ == "__main__":
    
    #####################################
    ### Test Section 3 - Autoencoders ###
    #####################################
    
    mnist, mnist_val, cifar10, cifar10_val = load_data()
    train_set, dataset_name, data_shape, data_size, valid_set = get_data(mnist, mnist_val)

    ### Test Nonlinear Autoencoder ###
    K = 20
    trained_conv_AE = ConvAutoEncoder(data_shape, K)
    assert trained_conv_AE.encode(train_set[0][0].unsqueeze(0)).numel() == K, "Encoder output size should be K!"
    conv_losses = train_autoencoder(trained_conv_AE, train_set, device=DEVICE, seed=SEED)
    plt.plot(conv_losses)
    plt.show()
