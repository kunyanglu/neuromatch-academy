import time
import copy
import torch
import torchvision
import random
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

def zero_grad(params):
    for par in params:
        if not(par.grad is None):
            par.grad.data.zero_()

def random_update(model, noise_scale=0.1, normalized=False):
    for par in model.parameters():
        noise = torch.randn_like(par)
        if normalized:
            noise /= torch.norm(noise)
        par.data +=  noise_scale * noise

def gradient_update(loss, params, lr=1e-3):
    zero_grad(params)
    loss.backward()
    with torch.no_grad():
        for par in params:
            #par.data -= lr * torch.gradient(par.data)
            pr.data -= lr * par.grad.data

def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

def loss_2d(model, u, v, mask_idx=(0, 378), bias_id=7):
    # Defines a 2-dim function by freezing all but two parameters of a linear model
    mask = torch.ones_like(model.main[0].weight)
    mask[mask_idx[0], mask_idx[1]] = 0
    masked_weights = model.main[0].weight * mask
    
    masked_weights[mask_idx[0], mask_idx[1]] = u
    res = X.reshape(-1, 784) @ masked_weights.T + model.main[0].bias

    res[:, 7] += v - model.main[0].bias[7]
    res =  F.log_softmax(res, dim=1)

    return loss_fn(res, y)

##############################################
### Coding Exercise 4 - Implement Momentum ###
##############################################

def momentum_update(loss, params, grad_vel, lr=1e-3, beta=0.8):
    zero_grad(params)
    loss.backward()

    with torch.no_grad():
        for (par, vel) in zip(params, grad_vel):
            #vel.data = beta * vel.data
            vel.data = -lr * par.grad.data + beta * vel.data
            par.data += vel.data
            #par.data += -lr * par.grad.data + vel.data

##############################################
### Coding Exercise 6 - Minibatch Sampling ###
##############################################

def sample_minibatch(input_data, target_data, num_points=100):
    batch_indices = random.sample(range(len(input_data)), num_points)
    batch_inputs = input_data[batch_indices, :]
    batch_targets = target_data[batch_indices]

    return batch_inputs, batch_targets

#############################################
### Coding Exercise 7 - Implement RMSprop ###
#############################################

def rmsprop_update(loss, params, grad_sq, lr=1e-3, alpha=0.8, epsilon=1e-8):
    zero_grad(params)
    loss.backward()

    with torch.no_grad():
        for (par, gsq) in zip(params, grad_sq):
            gsq.data = alpha * gsq.data + (1-alpha) * (par.grad.data**2) 
            par.data -=  lr * par.grad.data / math.sqrt(gsq.data + epsilon)

if __name__ == "__main__":
    SEED = 2021
    set_seed(SEED)
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
    
    ##############################
    ### Test Coding Exercise 3 ###
    ##############################
    
    model1 = MLP(in_dim=784, out_dim=10, hidden_dims=[])
    print('\n The model1 parameters before the update are: \n')
    print_params(model1)
    loss = loss_fn(model1(X), y)
    
    ##############################
    ### Test Coding Exercise 4 ###
    ##############################
    
    model2 = MLP(in_dim=784, out_dim=10, hidden_dims=[])
    print('\n The model2 parameters before the update are: \n')
    print_params(model2)
    loss = loss_fn(model2(X), y)
    initial_vel = [torch.randn_like(p) for p in model2.parameters()]
    
    momentum_update(loss, list(model2.parameters()), grad_vel=initial_vel, lr=1e-1, beta=0.9)
    print('\n The model2 parameters after the update are: \n')
    print_params(model2)
    
    ##############################
    ### Test Coding Exercise 6 ###
    ##############################
        
    x_batch, y_batch = sample_minibatch(X, y, num_points=100)
    print(f"The input shape is {x_batch.shape} and the target shape is: {y_batch.shape}")
    
    ##############################
    ### Test Coding Exercise 7 ###
    ##############################
    
    model3 = MLP(in_dim=784, out_dim=10, hidden_dims=[])
    print('\n The model3 parameters before the update are: \n')
    print_params(model3)
    loss = loss_fn(model3(X), y)
    # Intialize the moving average of squared gradients
    grad_sq = [1e-6*i for i in list(model3.parameters())]
