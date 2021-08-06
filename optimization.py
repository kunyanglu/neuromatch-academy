import time
import copy
import torch
import torchvision

import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets

from tqdm.auto import tqdm

from MLP import Net, train_test_classification                                  
from MLP import shuffle_and_split_data, set_seed, seed_worker
from deep_MLPs import imshow, sample_grid, plot_decision_map

class MLP(nn.Module):
    
    def __init__(self, in_dim=784, out_dim=10, hidden_dims=[], use_bias=True):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        if len(hidden_dims) == 0:
            layers = [nn.Linear(in_dim, out_dim, bias=use_bias)]
        else:
            # Initialize
            layers = [nn.Linear(in_dim, hidden_dims[0], bias=use_bias), nn.ReLU()]
            for i, hidden_dim in enumerate(hidden_dims[:-1]):
                layers += [nn.Linear(hidden_dim, hidden_dims[i + 1], bias=use_bias),
                        nn.ReLU()]
            # Add the final layer
            layers += [nn.Linear(hidden_dims[-1], out_dim, bias=use_bias)]
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the images into 'vectors'
        transformed_x = x.view(-1, self.in_dim)
        hidden_output = self.main(transformed_x)
        output = F.log_softmax(hidden_output, dim=1)
        return output

def load_mnist_data(change_tensors=False, download=False):
    
    train_set = datasets.MNIST(root='.', train=True, download=download,
                             transform=torchvision.transforms.ToTensor())
    
    test_set = datasets.MNIST(root='.', train=False, download=download,
                            transform=torchvision.transforms.ToTensor())

    mean = train_set.data.float().mean()
    std = train_set.data.float().std()

    if change_tensors:
        # Normalize the dataset that has already been converted to tensor
        train_set.data = (train_set.data.float() - mean) / std
        test_set.data = (test_set.data.float() - mean) / std
    else:
        # Otherwise, convert to tensor and then normalize
        tform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[mean / 255.], std=[std / 255.])
            ])
        train_set = datasets.MNIST(root='.', train=True, download=download,
                                    transform=tform)
        test_set = datasets.MNIST(root='.', train=False, download=download,
                                    transform=tform)

    return train_set, test_set

###############################################################
### Coding Exercise 3 - Manually Implement Gradient Descent ###
###############################################################


if __name__ == "__main__":
    train_set, test_set = load_mnist_data(change_tensors=True)

    # Randomly select 500 samples out of 60000
    subset_index = np.random.choice(len(train_set.data), 500)
    X, y = train_set.data[subset_index, :], train_set.targets[subset_index]
    # Simple test
    model = MLP(in_dim=784, out_dim=10, hidden_dims=[])
    print(model)

    loss_fn = F.nll_loss
    partial_trained_model = MLP(in_dim=784, out_dim=10, hidden_dims=[])
    optimizer = optim.Adam(partial_trained_model.parameters(), lr=7e-4)
    
    for _ in range(200):
      loss = loss_fn(partial_trained_model(X), y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Show class filters of a trained model
    W = partial_trained_model.main[0].weight.data.numpy()
    fig, axs = plt.subplots(1, 10, figsize=(15, 4))
    for class_id in range(10):
      axs[class_id].imshow(W[class_id, :].reshape(28, 28), cmap='gray_r')
      axs[class_id].axis('off')
      axs[class_id].set_title('Class ' + str(class_id) )
    plt.show()
