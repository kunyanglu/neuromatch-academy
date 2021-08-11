import os
import time
import tqdm
import random
import torch
import IPython
import torchvision
import requests, urllib
import tarfile

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import AlexNet
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

from PIL import Image
from io import BytesIO
from ipywidgets import widgets
from helper import seed_worker

SEED = 2021

def prepare_data(root_path):
    val_transform = transforms.Compose((transforms.Resize((256, 256)),
                                        transforms.ToTensor()))

    imagenette_val = ImageFolder(root_path + 'val', transform=val_transform)

    train_transform = transforms.Compose((transforms.Resize((256, 256)),
                                          transforms.ToTensor()))

    imagenette_train = ImageFolder(root_path + 'train',
                                   transform=train_transform)
    random.seed(SEED)
    random_indices = random.sample(range(len(imagenette_train)), 400)
    imagenette_train_subset = torch.utils.data.Subset(imagenette_train,
                                                      random_indices)

    # Subset to only one tenth of the data for faster runtime
    random_indices = random.sample(range(len(imagenette_val)), int(len(imagenette_val) * .1))
    imagenette_val = torch.utils.data.Subset(imagenette_val, random_indices)

    return imagenette_train_subset, imagenette_val

def load_data(imagenette_train_subset, imagenette_val):
    # To preserve reproducibility
    g_seed = torch.Generator()
    g_seed.manual_seed(SEED)

    imagenette_train_loader = torch.utils.data.DataLoader(imagenette_train_subset,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    worker_init_fn=seed_worker,
                                                    generator=g_seed
                                                    )

    imagenette_val_loader = torch.utils.data.DataLoader(imagenette_val,
                                                    batch_size=16,
                                                    shuffle=False,
                                                    num_workers=2,
                                                    worker_init_fn=seed_worker,
                                                    generator=g_seed
                                                    )
    dataiter = iter(imagenette_val_loader)
    images, labels = dataiter.next()

    # Show images
    plt.figure(figsize=(8, 8))
    plt.imshow(make_grid(images, nrow=4).permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    return imagenette_train_loader, imagenette_val_loader

def imagenette_train_loop(model, optimizer, train_loader, 
                            loss_fn, device='cpu'):
    for epoch in tqdm.notebook.tqdm(range(5)):
        model.train()
        # Train on a batch of images
        for imagenette_batch in train_loader:
            images, labels = imagenette_batch
            
            # Convert labels from imagenette indices to imagenet labels
            for i, label in enumerate(labels):
                @TODO:Set up dir_index_to_imagnel_label dictionary
                labels[i] = dir_index_to_imagenet_label[label.item()]
            
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            optimizer.zero_grad()
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
    
    return model

def main(imagenette_train_loader, imagenette_val_loader):
    # Instantiate a pretrained resnet model
    set_seed(seed=SEED)
    resnet = torchvision.models.resnet18(pretrained=True).to('cpu')
    resnet_opt = torch.optim.Adam(resnet.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    imagenette_train_loop(resnet,
                          resnet_opt,
                          imagenette_train_loader,
                          loss_fn,
                          device='cpu')

if __name__ == "__main__":
    root_path = 'imagenette2-320/'
    imagenette_train_subset, imagenette_val = prepare_data(root_path)
    imagenette_train_loader, imagenette_val_loader = load_data(imagenette_train_subset, imagenette_val)
    main(imagenette_train_loader, imagenette_val_loader)
