import os
import time
import tqdm
import torch
import IPython
import torchvision
import requests, urllib

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


### Visisualize intermediate filters
def display_filters():
    
    url, filename = ("https://raw.githubusercontent.com/NeuromatchAcademy/course-content-dl/main/tutorials/W2D2_ModernConvnets/static/dog.jpg", "dog.jpg")
    try: urllib.request.urlopen(url).retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                 ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    plt.imshow(input_image)
    plt.show()
    
    return input_batch, input_image

def alexnet_intermediate_output(net, image):

    return F.relu(net.features[0](image))

def browse_images(input_batch, input_image):
    state_dict = torch.hub.load_state_dict_from_url("https://osf.io/9dzeu/download")
    
    alexnet = AlexNet()
    alexnet.load_state_dict(state_dict=state_dict)

    intermediate_output = alexnet_intermediate_output(alexnet, input_batch)
    print(f'intermediate_output shape: {intermediate_output.shape}')
    n = intermediate_output.shape[1]
    
    def view_image(i):
        with torch.no_grad():
            channel = intermediate_output[0, i, :].squeeze()
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].imshow(input_image)
            ax[1].imshow(filters[i])
            ax[1].set_xlim([-22, 33])
            ax[2].imshow(channel.cpu())
            ax[0].set_title('Input image')
            ax[1].set_title(f"Filter {i}")
            ax[2].set_title(f"Filter {i} on input image")
            [axi.set_axis_off() for axi in ax.ravel()]

    widgets.interact(view_image, i=(0, n-1))

if __name__ == "__main__":
    input_batch, input_image = display_filters()
    browse_images(input_batch, input_image)
