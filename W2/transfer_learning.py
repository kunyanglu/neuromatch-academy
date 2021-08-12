import time
import math
import torch
import string
import random
import unidecode
import zipfile, io, os, requests
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tqdm.notebook import tqdm
from helper import read_file, char_tensor, time_since, generate

def load_data(f_name):
    train_transform = transforms.Compose((transforms.Resize((256, 256)),
                                          transforms.ToTensor()))

    pokemon_dataset = ImageFolder(f_name, transform=train_transform)

    image_count = len(pokemon_dataset)
    
    train_indices = []
    test_indices = []
    for i in range(image_count):
      # Put ten percent of the images in the test set
      if random.random() < .1:
        test_indices.append(i)
      else:
        train_indices.append(i)

    pokemon_test_set = torch.utils.data.Subset(pokemon_dataset, test_indices)
    pokemon_train_set = torch.utils.data.Subset(pokemon_dataset, train_indices)

    pokemon_train_loader = torch.utils.data.DataLoader(pokemon_train_set,
                                                       batch_size=16,
                                                       shuffle=True,)
    pokemon_test_loader = torch.utils.data.DataLoader(pokemon_test_set,
                                                      batch_size=16)

    dataiter = iter(pokemon_train_loader)
    images, labels = dataiter.next()
    '''
    # show sample images
    plt.imshow(make_grid(images, nrow=4).permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    '''
    return pokemon_train_loader, pokemon_test_set, pokemon_test_loader

def train(num_classes, fine_tuning_all, pokemon_train_loader, 
            pokemon_test_set, pokemon_test_loader):
    
    resnet = initialize_resnet(fine_tuning_all=fine_tuning_all)

    num_ftrs = resnet.fc.in_features
    # Reset final fully connected layer, number of classes = types of Pokemon = 9
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    resnet.to('cpu')
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    pretrained_accs = []
    
    for epoch in range(10):

        for batch in pokemon_train_loader:
            images, labels = batch
            images = images.to('cpu')
            labels = labels.to('cpu')

            optimizer.zero_grad()
            output = resnet(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_sum = 0
            total_correct = 0
            total = len(pokemon_test_set)
            for batch in pokemon_test_loader:
                images, labels = batch
                images = images.to('cpu')
                labels = labels.to('cpu')

                output = resnet(images)
                loss = loss_fn(output, labels)
                loss_sum += loss.item()

                predictions = torch.argmax(output, dim=1)
                num_correct = torch.sum(predictions == labels)
                total_correct += num_correct
        
            # Plot accuracy
            pretrained_accs.append(total_correct / total)
            plt.plot(pretrained_accs)
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('Pokemon prediction accuracy')
            plt.show()
            # IPython.display.clear_output(wait=True)
            # IPython.display.display(plt.gcf())

def initialize_resnet(fine_tuning_all):
   
    ##########################################
    ### Section 7.2 - Fine-tuning a ResNet ###
    ##########################################

    ### Replace the classification layer 
    ### and fine-tune the entire network to perform a different task

    if fine_tuning_all:
        resnet = torchvision.models.resnet18(pretrained=True)
    
    else:    
    #########################################################
    ### Section 7.3 - Train only the Classification Layer ###
    #########################################################
        
        resnet = torchvision.models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

    return resnet

if __name__ == "__main__":
    
    f_name = 'small_pokemon_dataset'
    num_classes = 9
    fine_tuning_all = False
    # Dataloaders
    pokemon_train_loader, pokemon_test_set, pokemon_test_loader = load_data(f_name)
    # Training
    train(num_classes, fine_tuning_all,
            pokemon_train_loader, pokemon_test_set, pokemon_test_loader)



