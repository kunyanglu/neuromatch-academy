import copy
import torch
import random
import pathlib
import numpy as np
import requests, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from MLP import set_seed, seed_worker
from deep_MLPs import imshow
from regularization_1 import plot_weights, AnimalNet, train, test, main
from regularization_1 import calculate_frobenius_norm, early_stopping_main

SEED = 2021
### Plotting Function(s)
def visualize_data(dataloader):

    for idx, (data,label) in enumerate(dataloader):
        plt.figure(idx)
        # Choose the datapoint you would like to visualize
        index = 22

        # Choose that datapoint using index and permute the dimensions
        # and bring the pixel values between [0,1]
        data = data[index].permute(1, 2, 0) * \
               torch.tensor([0.5, 0.5, 0.5]) + \
               torch.tensor([0.5, 0.5, 0.5])

        # Convert the torch tensor into numpy
        data = data.numpy()

        plt.imshow(data)
        plt.axis(False)
        image_class = classes[label[index].item()]
        print(f'The image belongs to : {image_class}')

    plt.show()

# Simple Net
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1, 300)
        self.fc2 = nn.Linear(300, 500)
        self.fc3 = nn.Linear(500, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        output = self.fc3(x)
        return output

# Network Class - Animal Faces
class BigAnimalNet(nn.Module):
      
    def __init__(self):
        super(BigAnimalNet, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 124)
        self.fc2 = nn.Linear(124, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

### Implement Animal net with dropouts
class AnimalNetDropout(nn.Module):
    
    def __init__(self):
        super(AnimalNetDropout, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 248)
        self.fc2 = nn.Linear(248, 210)
        self.fc3 = nn.Linear(210, 3)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.dropout1(self.fc1(x)))
        x = F.leaky_relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        
        return output

### Create test dataset
def create_test_data():
    test_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    img_test_dataset = ImageFolder(data_path/'val', transform=test_transform)
    return img_test_dataset

### Load train and validation datasets (original,random, partically random)
def load_data(f_name, batch_size):

    batch_size = 128
    classes = ('cat', 'dog', 'wild')
    # Split into len_train, len_val, len_test
    train_val_test_split = [100, 100, 14430]
    
    # Transform input data
    train_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    g_seed = torch.Generator()
    g_seed.manual_seed(SEED)
    
    # Split data into training and validation data
    data_path = pathlib.Path('.')/f_name
    img_dataset = ImageFolder(data_path/'train', transform=train_transform)
    img_train_data, img_val_data, _ = random_split(img_dataset, train_val_test_split)

    # Create train and validation dataloader for random dataset
    train_loader = DataLoader(
                                img_train_data,
                                batch_size=batch_size,
                                num_workers=2,
                                worker_init_fn=seed_worker,
                                generator=g_seed
                            )
    
    val_loader = DataLoader(
                               img_val_data,
                               batch_size=1000,
                               num_workers=2,
                               worker_init_fn=seed_worker,
                               generator=g_seed
                            )
    # Create test dataset
    test_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    img_test_dataset = ImageFolder(data_path/'val', transform=test_transform)
    
    return img_test_dataset, train_loader, val_loader

###############################################
### Coding Exercise 1.1 - L1 Regularization ###
###############################################

def l1_reg(model):
    l1 = 0.0
    for param in model.parameters():
        l1 += torch.sum(abs(param.data))
    print(f'l1 is {l1}')
    return l1

def l2_reg(model):
    l2 = 0.0
    for param in model.parameters():
        l2 += torch.sum(param.data**2)
    
    return l2

###############################################
### Coding Exercise 1.2 - L2 Regularization ###
###############################################

### Train a classifier using L1 regularizer 
def train(img_test_dataset, train_loader, val_loader):
    # Set the arguments
    args = {
        'test_batch_size': 1000,
        'epochs': 200,
        'lr': 5e-3,
        'batch_size': 32,
        'momentum': 0.9,
        'device': 'cpu',
        'lambda1': 0.001,
        'lambda2': 0.001,
        'log_interval': 100
    }
    # Intialize the model
    set_seed(seed=SEED)
    model = AnimalNetDropout()

    # Train the model
    val_acc_dropout, train_acc_dropout, _, model_dq = main(args,
                                                        model,
                                                        train_loader,
                                                        val_loader,
                                                        img_test_dataset,
                                                        reg_function1=l1_reg)
    set_seed(seed=SEED)
    model = BigAnimalNet()
    
    # Train the model
    val_acc_big, train_acc_big, _, model_big = main(args,
                                                    model,
                                                    train_loader,
                                                    val_loader,
                                                    img_test_dataset,
                                                    reg_function1=l1_reg)
    # Train and Test accuracy plot
    plt.figure()
    plt.plot(val_acc_big, label='Val - Big', c='blue', ls='dashed')
    plt.plot(train_acc_big, label='Train - Big', c='blue', ls='solid')
    plt.plot(val_acc_dropout, label='Val - DP', c='magenta', ls='dashed')
    plt.plot(train_acc_dropout, label='Train - DP', c='magenta', ls='solid')
    plt.title('Dropout')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    ################################
    ### Test Coding Exercise 1.1 ###
    ################################

    net = nn.Linear(20, 20)
    print(f"L1 norm of the model: {l1_reg(net)}")

    # Load test, train, val dataset
    f_name = 'afhq'
    batch_size = 128
    img_test_dataset, train_loader, val_loader = load_data(f_name, batch_size)
    # Train
    train(img_test_dataset, train_loader, val_loader)


