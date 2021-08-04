import torch
import random
import numpy as np
from torch import nn
from math import pi
import matplotlib.pyplot as plt

def set_seed(seed=None, seed_torch=True):
    
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

# In case that `DataLoader` is used
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#############################################
### Coding Exercise 1.1 - Gradient Vector ###
#############################################

def fun_z(x, y):
    
    z = np.sin(x**2 + y**2)
    return z

def fun_dz(x, y):
    
    dz_dx = np.cos(x**2 + y**2) * 2*x
    dz_dy = np.cos(x**2 + y**2) * 2*y

    return (dz_dx, dz_dy)

############################################################
### Coding Exercise 2.1 - Building a Computational Graph ###
############################################################

class SimpleGraph:
  def __init__(self, w, b):
    """Initializing the SimpleGraph

    Args:
      w (float): initial value for weight
      b (float): initial value for bias
    """
    assert isinstance(w, float)
    assert isinstance(b, float)
    self.w = torch.tensor([w], requires_grad=True)
    self.b = torch.tensor([b], requires_grad=True)
  
  def forward(self, x):
    """Forward pass

    Args:
      x (torch.Tensor): 1D tensor of features

    Returns:
      torch.Tensor: model predictions
    """
    assert isinstance(x, torch.Tensor)
    prediction = torch.tanh(self.w * x + self.b)

    return prediction

def sq_loss(y_true, y_prediction):
    
    assert isinstance(y_true, torch.Tensor)
    assert isinstance(y_prediction, torch.Tensor)
    loss = (y_true - y_prediction)**2

    return loss

###########################################
### Coding Exercise 3.1 - Training Loop ###
###########################################

class WideNet(nn.Module):
    
    def __init__(self):
        n_cells = 512
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(1, n_cells),
                nn.Tanh(),
                nn.Linear(n_cells, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def train(features, labels, model, loss_fun, optimizer, n_epochs):
    
    loss_record = []  # keeping recods of loss
    for i in range(1, n_epochs+1):
        print(f'Process Training Epoch {i}')
        optimizer.zero_grad()
        prediction = model(features)
        loss = loss_func(prediction, labels)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    
    return loss_record

if __name__ == "__main__":

    # set device (not necessary since 'cuda' is not available)
    SEED = 2021
    set_seed(seed=SEED)
    # DEVICE = set_device()

    ################################
    ### Test Coding Exercise 2.1 ###
    ################################
    feature = torch.tensor([1])  # input tensor
    target = torch.tensor([7])  # target tensor
    simple_graph = SimpleGraph(-0.5, 0.5)
    print(f"initial weight = {simple_graph.w.item()}, "
            f"\ninitial bias = {simple_graph.b.item()}")
    prediction = simple_graph.forward(feature)
    square_loss = sq_loss(target, prediction)
    print(f"for x={feature.item()} and y={target.item()}, "
            f"prediction={prediction.item()}, and L2 Loss = {square_loss.item()}")
    
    ################################
    ### Test Coding Exercise 3.1 ###
    ################################
    
    # Create sample dataset
    n_samples = 32
    inputs = torch.linspace(-1.0, 1.0, n_samples).reshape(n_samples, 1)
    noise = torch.randn(n_samples, 1) / 4
    targets = torch.sin(pi * inputs) + noise
    plt.figure(figsize=(8, 5))
    plt.scatter(inputs, targets, c='c')
    plt.xlabel('x (inputs)')
    plt.ylabel('y (targets)')
    plt.show()

    model = WideNet()
    loss_func = nn.MSELoss()
    lr = 0.003
    n_epochs = 1847

    sgd_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_record = train(inputs, targets, model, loss_func, sgd_optimizer, n_epochs)
    print(len(loss_record))


