import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Grayscale
from sklearn.datasets import make_moons

### Coding Exercises 2.3 ###

def functionC(my_tensor1, my_tensor2):
    if torch.numel(my_tensor1) == torch.numel(my_tensor2):
        my_tensor2 = my_tensor2.reshape(list(my_tensor1.shape))
        return torch.add(my_tensor1, my_tensor2)
    else:
        my_tensor1_flattened = torch.flatten(my_tensor1)
        print(my_tensor1_flattened, my_tensor1_flattened.shape)
        print(my_tensor2, my_tensor2.shape)
        return torch.cat((torch.flatten(my_tensor1), my_tensor2))

### Section 2.5: Datasets and Dataloaders ###

def load_data():
    # download dataset
    cifar10_data = datasets.CIFAR10(
        root='data',
        download=True,
        transform=ToTensor()
        )
    # print the number of samples in the loaded dataset
    print(f"Number of samples: {len(cifar10_data)}")
    print(f"Class names: {cifar10_data.classes}") 

    # load the training samples
    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
        )
    testing_data = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
        )
    # create dataloaders
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=64 ,shufffle=True)
    return train_dataloader, test_dataloader

### Coding Exercise 2.6 ###

def load_data_as_gray_images():
    data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=Compose([
            Grayscale(num_output_channels=1),
            ToTensor(),
        ])
    )
    # display_image(data)
    return data	

# ensure reproducibility
def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
  
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')

### Coding Exercise 2.5 ###

def display_image(data):
    # load a random image
    set_seed(2021)
    image, _ = data[random.randint(0,len(data))]
    image = image.permute(1, 2, 0)
    # plt.imshow(image)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()

### Section 3: Neural Networks ###

def create_csv():
    # Create a dataset of 256 points with a little noise
    X, y = make_moons(256, noise=0.1)

    # Store the data as a Pandas data frame and save it to a CSV file
    df = pd.DataFrame(dict(x0=X[:,0], x1=X[:,1], y=y))
    df.to_csv('sample_data.csv')

# build a sample neural net
class naiveNet(nn.Module):
    def __init__(self):
        super(naiveNet, self).__init__()
        self.layers = nn.Sequential(
            # input layer of size 2 -> hidden layer of size 16
            nn.Linear(2, 16),
            nn.ReLU(),
            # hidden layer of size 16 -> output size of 2
            nn.Linear(16, 2)
        )
    # specifies the computation of the net
    def forward(self, x):
        return self.layers(x)

    def predict(self, output):
        # after forward pass
        # then choose the label with highest likelihood
        return torch.argmax(output, 1)

# DIY dataloader
class ToyDataset():
    
    def __init__(self, csv_file, transform=None):
        self.toy_data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.toy_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = torch.tensor(self.toy_data.iloc[idx, 0:2].to_numpy()).float()
        y = torch.tensor(self.toy_data.iloc[idx, -1]).float()
        sample = {'X': X, 'y':y}

        return sample

### Coding Exercise 3.2 ###

def dumb_classifier(model, input_data):
    data = pd.read_csv("sample_data.csv")
    # forward pass to the net
    output = model(input_data)
    print(f'Network Output: {output}')
    # predict the label of each point
    y_predicted = model.predict(output)
    print(f'Predicted Labels: {y_predicted}')

def train(model, train_loader):
    # define the loss function:
    # here we chose the Cross Entropy for the classification task
    loss_function = nn.CrossEntropyLossI()
    
    # define an optimizer for the learning rate (here we chose SGD)
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # number of epochs
    epochs = 1000
    losses = []

    # load data
    toy_dataset = ToyDataset(csv_file='sample_data.csv') 
    dataloader = Dataloader(toy_dataset, batch_size=4, shuffle=True) 
    
    for batch_idx, data in enumerate(dataloader):
        y_logits = model(data)
        print(f'data from train_loader: {data.shape}')
        return
    # loss = loss_function(y_logits, y)

if __name__ == "__main__":
    D = torch.tensor([[1, -1], [-1,3]])
    E = torch.tensor([2,3,0,2])
    F = torch.tensor([2,3,0])

    # print(functionC(D, E), functionC(D, F))
    # cifar10_data = load_data()
    # display_image(cifar10_data)
    # load_data_as_gray_images()
    # create_csv()
    
    ### Test Section 3: Neural Networks ###
    model = naiveNet()
    input_data = torch.tensor([
        [0.9066, 0.5052],
        [-0.2024, 1.1226],
        [1.0685, 0.2809],
        [0.6720, 0.5097],
        [0.8548, 0.5122]
    ])
    # dumb_classifier(model, input_data)
    load_csv()
