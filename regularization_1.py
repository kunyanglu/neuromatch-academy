import time
import copy
import torch
import pathlib
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm.auto import tqdm
from IPython.display import HTML
from torchvision import transforms
from torchvision.datasets import ImageFolder
from deep_MLPs import imshow
from MLP import set_seed, seed_worker

SEED = 2021

### Plotting Functions ###
def plot_weights(norm, labels, ws, title='Weight Size Measurement'):
    plt.figure(figsize=[8, 6])
    plt.title(title)
    plt.ylabel('Frobenius Norm Value')
    plt.xlabel('Model Layers')
    plt.bar(labels, ws)
    plt.axhline(y=norm,
                linewidth=1,
                color='r',
                ls='--',
                label='Total Model F-Norm')
    plt.legend()
    plt.show()

def early_stop_plot(train_acc_earlystop, val_acc_earlystop, best_epoch):
    plt.figure(figsize=(8, 6))
    plt.plot(val_acc_earlystop,label='Val - Early',c='red',ls = 'dashed')
    plt.plot(train_acc_earlystop,label='Train - Early',c='red',ls = 'solid')
    plt.axvline(x=best_epoch, c='green', ls='dashed',
            label='Epoch for Max Val Accuracy')
    plt.title('Early Stopping')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

class AnimalNet(nn.Module):
    
    def __init__(self):
        super(AnimalNet, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def train(args, model, train_loader, optimizer,
        reg_function1=None, reg_function2=None, criterion=F.nll_loss):
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        output = model(data)
        if reg_function1 is None:
            loss = criterion(output, target)
        elif reg_function2 is None:
            loss = criterion(output, target) + \
            args['lambda']*reg_function1(model)
        else:
            loss = criterion(output, target) + \
            args['lambda1']*reg_function1(model) + \
            args['lambda2']*reg_function2(model)
        
        loss.backward()
        optimizer.step()
    
    return model

def test(model, test_loader, criterion=F.nll_loss, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            # Get the index of the max log likelihood probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    return 100. * correct / len(test_loader.dataset)

def main(args, model, train_loader, val_loader,
         reg_function1=None, reg_function2=None):
    
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                        momentum=args['momentum'])
    val_acc_list, train_acc_list,param_norm_list = [], [], []
    for epoch in tqdm(range(args['epochs'])):
        trained_model = train(args, model, train_loader, optimizer,
                            reg_function1=reg_function1,
                            reg_function2=reg_function2)
        train_acc = test(trained_model, train_loader)
        val_acc = test(trained_model, val_loader)
        param_norm = calculate_frobenius_norm(trained_model)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        param_norm_list.append(param_norm)

    return val_acc_list, train_acc_list, param_norm_list, trained_model

##########################################
### Coding Exercise 1 - Frobenius Norm ###
##########################################

def calculate_frobenius_norm(model):
    # Calculate the weight size of the entire model
    norm = 0.0
    for param in model.parameters():
        norm += torch.sum(param.data**2)
    norm = torch.sqrt(norm)  
    return norm

### Get weight size in each layer of the model
def calculate_frobenius_norm_layer(model):
    norm, ws, labels = 0.0, [], []
    # Sum all the weights
    for name, parameters in model.named_parameters():
        p = torch.sum(parameters**2)
        norm += p
        ws.append((p**0.5).cpu().detach().numpy())
        labels.append(name)
    norm = (norm**0.5).cpu().detach().numpy()

    return norm, ws, labels

##########################################
### Coding Exercise 4 - Early Stopping ###
##########################################

def early_stopping_main(args, model, train_loader, val_loader):
    optimizer = optim.SGD(model.parameters(),
                        lr=args['lr'],
                        momentum=args['momentum'])
    best_acc = 0.0
    best_epoch = 0
    # Number of successive epochs to wait before stopping training process
    patience = 20
    # Keeps track of epoch account during which the val_acc was less than best_acc
    wait = 0
    val_acc_list, train_acc_list = [], []
    
    for epoch in tqdm(range(args['epochs'])):
        # Train the model
        trained_model = train(args, model, train_loader, optimizer)
        # Calculate training accuracy
        train_acc = test(trained_model, train_loader)
        # Calculate validation accuracy
        val_acc = test(trained_model, val_loader)
    
        if (val_acc > best_acc):
            best_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(trained_model)
            wait = 0
        else:
            wait += 1

        if (wait > patience):
            print(f'early stopped on epoch: {epoch}')
            break

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    return val_acc_list, train_acc_list, best_model, best_epoch

def load_data():
    batch_size = 128
    classes = ('cat', 'dog', 'wild')
    # Split data into sets for training, validating, testing
    len_train, len_val, len_test = 100, 100, 14430

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    # Using pathlib to be compatible with all OS's
    data_path = pathlib.Path('.')/'afhq'
    img_dataset = ImageFolder(data_path/'train', transform=train_transform)

    g_seed = torch.Generator()
    g_seed.manual_seed(SEED)

    img_train_data, img_val_data,_ = torch.utils.data.random_split(img_dataset,
                                                                   [len_train,
                                                                    len_val,
                                                                    len_test])
    train_loader = torch.utils.data.DataLoader(img_train_data,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               worker_init_fn=seed_worker,
                                               generator=g_seed) 
    val_loader = torch.utils.data.DataLoader(img_val_data,
                                             batch_size=1000,
                                             num_workers=2,
                                             worker_init_fn=seed_worker,
                                             generator=g_seed)
    return train_loader, val_loader

if __name__ == "__main__":
    set_seed(seed=SEED)
    '''
    net = nn.Linear(10, 1)
    print(f'Frobenius Norm of Single Linear Layer: {calculate_frobenius_norm(net)}')
    
    norm, ws, labels = calculate_frobenius_norm_layer(net)
    print(f'Frobenius Norm of Single Linear Layer: {norm:.4f}')
    # Plot the weights
    plot_weights(norm, labels, ws)

    model = AnimalNet()
    norm, ws, labels = calculate_frobenius_norm_layer(model)
    print(f'Frobenius Norm of Models weights: {norm:.4f}')
    plot_weights(norm, labels, ws)
    '''
    ##############################
    ### Test Coding Exercise 4 ###
    ##############################

    # Set the arguments
    args = {
        'epochs': 200,
        'lr': 5e-4,
        'momentum': 0.99,
        'device': 'cpu'
    }
    model = AnimalNet()
    train_loader, val_loader = load_data()
    val_acc_earlystop, train_acc_earlystop, best_model, best_epoch = \
    early_stopping_main(args, model, train_loader, val_loader)
    print(f'Maximum Validation Accuracy is reached at epoch: {best_epoch:2d}')

