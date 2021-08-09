import pathlib
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import requests, os
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from IPython.display import display

from MLP import Net, train_test_classification 
from MLP import shuffle_and_split_data, set_seed, seed_worker

### Ploting Functions ###
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis(False)
    plt.show()

def sample_grid(M=500, x_max=2.0):
    ii, jj = torch.meshgrid(torch.linspace(-x_max, x_max,M),
                          torch.linspace(-x_max, x_max, M))
    X_all = torch.cat([ii.unsqueeze(-1),
                        jj.unsqueeze(-1)],
                        dim=-1).view(-1, 2)
    return X_all

def plot_decision_map(X_all, y_pred, X_test, y_test,
                      M=500, x_max=2.0, eps=1e-3):
    decision_map = torch.argmax(y_pred, dim=1)
    for i in range(len(X_test)):
        indeces = (X_all[:, 0] - X_test[i, 0])**2 + (X_all[:, 1] - X_test[i, 1])**2 < eps
        decision_map[indeces] = (K + y_test[i]).long()

    decision_map = decision_map.view(M, M).cpu()
    plt.imshow(decision_map, extent=[-x_max, x_max, -x_max, x_max], cmap='jet')
    plt.plot()

def get_data_loaders(batch_size, seed):
    
    # Define transforms done during training
    augmentation_transforms = [transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip()] 
    # Define transforms done during training and testing
    preprocessing_transforms = [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] 

    train_transform = transforms.Compose(augmentation_transforms + \
            preprocessing_transforms)
    test_transform = transforms.Compose(preprocessing_transforms)

    data_path = pathlib.Path('.')/'afhq'
    img_train_dataset = ImageFolder(data_path/'train', transform=train_transform)
    img_test_dataset = ImageFolder(data_path/'val', transform=test_transform)
    
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)
    img_train_loader = DataLoader(
                                img_train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g_seed
                                )
    img_test_loader = DataLoader(
                                img_test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                worker_init_fn=seed_worker,
                                generator=g_seed
                                )
    print('Successfully loaded train and test dataset')
    return img_train_loader, img_test_loader

if __name__ ==  "__main__":
    
    ##############################
    ### Test Coding Exercise 2 ###
    ##############################
    
    SEED = 2021
    batch_size = 64
    set_seed(seed=SEED)
    img_train_loader, img_test_loader = get_data_loaders(batch_size, SEED)
    dataiter = iter(img_train_loader)
    print(f'dataiter has Type {dataiter}')
    images, labels = dataiter.next()
    # imshow(make_grid(images, nrow=8))

    # Train the network
    net = Net('ReLU()', 3*32*32, [64, 64, 64], 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    train_test_classification(net, criterion, optimizer,
                            img_train_loader, img_test_loader,
                            num_epochs=10)
    # Visualize the feature map
    fc1_weights = net.mlp[0].weight.view(64, 3, 32, 32).detach().cpu()
    fc1_weights /= torch.max(torch.abs(fc1_weights))
    imshow(make_grid(fc1_weights, nrow=8))
