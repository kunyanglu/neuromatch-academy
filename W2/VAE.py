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

def load_data():
    ### Load MNIST ###
    mnist = datasets.MNIST('./mnist/',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=False)
    mnist_val = datasets.MNIST('./mnist/',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=False)

    ### CIFAR 10 ###
    cifar10 = datasets.CIFAR10('./cifar10/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)
    cifar10_val = datasets.CIFAR10('./cifar10/',
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=False)
def get_data(name='mnist'):

    if name == 'mnist':
        my_dataset_name = "MNIST"
        my_dataset = mnist
        my_valset = mnist_val
        my_dataset_shape = (1, 28, 28)
        my_dataset_size = 28 * 28
      
    elif name == 'cifar10':
        my_dataset_name = "CIFAR10"
        my_dataset = cifar10
        my_valset = cifar10_val
        my_dataset_shape = (3, 32, 32)
        my_dataset_size = 3 * 32 * 32

    return my_dataset, my_dataset_name, my_dataset_shape, my_dataset_size, my_valset

if __name__ == "__main__":
    
    #####################################
    ### Test Section 3 - Autoencoders ###
    #####################################

    train_set, dataset_name, data_shape, data_size, valid_set = get_data(name='mnist')


