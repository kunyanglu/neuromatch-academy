import time
import torch
import pathlib
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import zipfile, gzip, shutil, tarfile
import requests, os

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm, trange
from PIL import Image
from scipy.signal import convolve2d
from scipy.signal import correlate2d
from IPython.display import IFrame

from helper import set_seed, seed_worker

SEED = 2021

############################################
### Coding Exerise 2.2 - ConvOutput Size ###
############################################

def calculate_output_shape(image_shape, kernel_shape):
    image_height, image_width = image_shape
    kernel_height, kernel_width = kernel_shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernal_width + 1
    
    return output_height, output_width

def convolution2d(image, kernel):
    im_h, im_w = image.shape
    ker_h, ker_w = kernel.shape
    out_h = im_h - ker_h + 1
    out_w = im_w - ker_w + 1
    # Empty matrix to store the output
    output = np.zeros((out_h, out_w))

    for out_row in range(out_h):
        
        for out_col in range(out_w):
            
            current_product = 0
        
            for i in range(ker_h):
            
                for j in range(ker_w):
                
                    current_product += image[out_row+i, out_col+j] * kernel[i,j] 

        output[out_row, out_col] = current_product

    return output

######################################
### Section 3.1 - Multiple Filters ###
######################################

### Build a simple ConvNet with Pytorch
class Net(nn.Module):

    def __init__(self, kernel=None, padding=0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, \
                                padding=padding)
        # Set up a default kernel
        if kernel is not None:
            dim1, dim2 = kernel.shape[0], kernel.shape[1]
            kernel = kernel.reshape(1, 1, dim1, dim2)
            
            self.conv1.weight = torch.nn.Parameter(kernel)
            self.conv1.bias = torch.nn.Parameter(torch.zeros_like(self.conv1.bias))

    def forward(self, x):
        
        return self.conv1(x)

class Net2(nn.Module):

    def __init__(self, padding=0):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5,
                                padding=padding)

        # first kernel - leading diagonal
        kernel_1 = torch.Tensor([[[ 1.,  1., -1., -1., -1.],
                                  [ 1.,  1.,  1., -1., -1.],
                                  [-1.,  1.,  1.,  1., -1.],
                                  [-1., -1.,  1.,  1.,  1.],
                                  [-1., -1., -1.,  1.,  1.]]])

        # second kernel - other diagonal
        kernel_2 = torch.Tensor([[[-1., -1., -1.,  1.,  1.],
                                  [-1., -1.,  1.,  1.,  1.],
                                  [-1.,  1.,  1.,  1., -1.],
                                  [ 1.,  1.,  1., -1., -1.],
                                  [ 1.,  1., -1., -1., -1.]]])

        # third kernel - checkerboard pattern
        kernel_3 = torch.Tensor([[[ 1.,  1., -1.,  1.,  1.],
                                  [ 1.,  1.,  1.,  1.,  1.],
                                  [-1.,  1.,  1.,  1., -1.],
                                  [ 1.,  1.,  1.,  1.,  1.],
                                  [ 1.,  1., -1.,  1.,  1.]]])


        # Stack all kernels in one tensor with (3, 1, 5, 5) dimensions
        multiple_kernels = torch.stack([kernel_1, kernel_2, kernel_3], dim=0)

        self.conv1.weight = torch.nn.Parameter(multiple_kernels)
        # Negative bias to give a threshold to select the high output value
        self.conv1.bias = torch.nn.Parameter(torch.Tensor([-4, -4, -12]))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

################################################
### Coding Exercise 4 - Implement My Own CNN ###
################################################

class myCNN(nn.Module):

    def __init__(self, padding=0, stride=0):
        
        super(myCNN, self).__init__()
        
        self.layer = nn.Sequential(
                
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
                nn.ReLU(),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),

                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),

                nn.Linear(in_features=9216, out_features=128),
                nn.ReLU(),

                nn.Linear(in_features=128, out_features=2)
        )

    def forward(self, x):
        
        return self.layer(x)

def train(model, device, train_loader, epochs):
    model.train()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        with tqdm(train_loader, unit='batch') as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                time.sleep(0.1)

def test(model, device, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    for data in data_loader:
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total

    return acc

def get_Xvs0_dataset(normalize=False, download=False):
    
    if normalize:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
    else:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
    
    emnist_train = datasets.EMNIST(root='.',
                                split='letters',
                                download=download,
                                train=True,
                                transform=transform)
    emnist_test = datasets.EMNIST(root='.',
                                split='letters',
                                download=download,
                                train=False,
                                transform=transform)

    # Get only X (15) or O (24) labels
    train_idx = (emnist_train.targets == 15) | (emnist_train.targets == 24)
    emnist_train.targets = emnist_train.targets[train_idx]
    emnist_train.data = emnist_train.data[train_idx]

    # Convert Xs predictions to 1, Os predictions to 0
    emnist_train.targets = (emnist_train.targets == 24).type(torch.int64)

    # Repeat for the test set
    test_idx = (emnist_test.targets == 15) | (emnist_test.targets == 24)
    emnist_test.targets = emnist_test.targets[test_idx]
    emnist_test.data = emnist_test.data[test_idx]

    emnist_test.targets = (emnist_test.targets == 24).type(torch.int64)
    
    return emnist_train, emnist_test


def get_data_loaders(train_dataset, test_dataset, batch_size=32, seed=0):

    g_seed = torch.Generator()
    g_seed.manual_seed(seed)

    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            worker_init_fn=seed_worker,
                            generator=g_seed)
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            worker_init_fn=seed_worker,
                            generator=g_seed)
    
    return train_loader, test_loader
    
def display_after_filter(emnist_train, emnist_test):
    
    # Index of an image in the dataset that corresponds to an X and O
    x_img_idx = 4
    o_img_idx = 15

    net2 = Net2().to('cpu')
    x_img = emnist_train[x_img_idx][0].unsqueeze(dim=0).to('cpu')
    output_x = net2(x_img)
    output_x = output_x.squeeze(dim=0).detach().cpu().numpy()

    o_img = emnist_train[o_img_idx][0].unsqueeze(dim=0).to('cpu')
    print(f'Input shape Before Downsampling {o_img.shape}')
    output_o = net2(o_img)
    print(f'Output shape After Downsampling {output_o.shape}')
    output_o = output_o.squeeze(dim=0).detach().cpu().numpy()
    print(f'Output shape After Downsampling {output_o.shape}')

    fig, ((ax11, ax12, ax13, ax14),
          (ax21, ax22, ax23, ax24),
          (ax31, ax32, ax33, ax34)) = plt.subplots(3, 4)

    # show the filters
    ax11.axis("off")
    ax12.set_title("filter 1")
    ax12.imshow(net2.conv1.weight[0, 0].detach().cpu().numpy(), cmap="gray")
    ax13.set_title("filter 2")
    ax13.imshow(net2.conv1.weight[1, 0].detach().cpu().numpy(), cmap="gray")
    ax14.set_title("filter 3")
    ax14.imshow(net2.conv1.weight[2, 0].detach().cpu().numpy(), cmap="gray")

    vmin, vmax = -6, 10
    # show x and the filters applied to x
    ax21.set_title("image x")
    ax21.imshow(emnist_train[x_img_idx][0].reshape(28, 28), cmap='gray')
    ax22.set_title("output filter 1")
    ax22.imshow(output_x[0], cmap='gray', vmin=vmin, vmax=vmax)
    ax23.set_title("output filter 2")
    ax23.imshow(output_x[1], cmap='gray', vmin=vmin, vmax=vmax)
    ax24.set_title("output filter 3")
    ax24.imshow(output_x[2], cmap='gray', vmin=vmin, vmax=vmax)

    # show o and the filters applied to o
    ax31.set_title("image o")
    ax31.imshow(emnist_train[o_img_idx][0].reshape(28, 28), cmap='gray')
    ax32.set_title("output filter 1")
    ax32.imshow(output_o[0], cmap='gray', vmin=vmin, vmax=vmax)
    ax33.set_title("output filter 2")
    ax33.imshow(output_o[1], cmap='gray', vmin=vmin, vmax=vmax)
    ax34.set_title("output filter 3")
    ax34.imshow(output_o[2], cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

if __name__ == "__main__":
    
    ### Test ConvNet built w Pytorch
    '''
    # Format a default 2x2 kernel of numbers from 0 through 3
    kernel = torch.Tensor(np.arange(4).reshape(2, 2))
    # Prepare the network with that default kernel
    net = Net(kernel=kernel, padding=0).to('cpu')

    # set up a 3x3 image matrix of numbers from 0 through 8
    image = torch.Tensor(np.arange(9).reshape(3, 3))
    image = image.reshape(1, 1, 3, 3).to('cpu') # BatchSizeXChannelsXHeightXWidth
    
    ### Without padding
    print("Image:\n" + str(image))
    print("Kernel:\n" + str(kernel))
    output = net(image) # Apply the convolution
    print("Output:\n" + str(output))
    
    ### Add padding
    print("Image (before padding):\n" + str(image))
    print("Kernel:\n" + str(kernel))
    net = Net(kernel=kernel, padding=1).to('cpu')
    output = net(image)
    print("Output:\n" + str(output))
    '''
    emnist_train, emnist_test = get_Xvs0_dataset(normalize=True)
   
    train_loader, test_loader = get_data_loaders(emnist_train, emnist_test,
                                                    seed=SEED)

    ######################
    ### Test Section 3 ###
    ######################
    
    # display_after_filter(emnist_train, emnist_test)

    ##############################
    ### Test Coding Exercise 4 ###
    ##############################

    # Train the network
    emnist_net = myCNN().to('cpu')
    print("Total Parameters in Network {:10d}".format( \
        sum(p.numel() for p in emnist_net.parameters())))
    
    train(emnist_net, 'cpu', train_loader, 1)
    test_accuracy = test(emnist_net, 'cpu', test_loader)
    print(f'Test Accuracy: {test_accuracy}')
