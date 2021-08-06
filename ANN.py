import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import ipywidgets as widgets
from tqdm.auto import tqdm
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset

#plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/\
#                content-creation/main/nma.mplstyle")

### Plotting Functions ###
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis(False)
    plt.show()

def plot_function_approximation(x, relu_acts, y_hat):
    fig, axes = plt.subplots(2, 1)
    # Plot ReLU Activations
    axes[0].plot(x, relu_acts.T)
    axes[0].set(xlabel='x',
                ylabel='Activation',
                title='ReLU Activations - Basis Functions')
    labels = [f"ReLU {i + 1}" for i in range(relu_acts.shape[0])]
    axes[0].legend(labels, ncol = 2)

    # Plot function approximation
    axes[1].plot(x, torch.sin(x), label='truth')
    axes[1].plot(x, y_hat, label='estimated')
    axes[1].legend()
    axes[1].set(xlabel='x',
                ylabel='y(x)',
                title='Function Approximation')
    plt.tight_layout()
    plt.show()

### Set Random Seed ###
def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

###########################################################
### Coding Exercise 1 - Function Approximation with ReLU ###
###########################################################

def approximate_function(x_train, y_train):
    # Number of relus
    n_relus = x_train.shape[0] - 1

    # x axis points (more than x train)
    x = torch.linspace(torch.min(x_train), torch.max(x_train), 1000)
    # bias term
    b = x_train[:-1]

    # Compute ReLU activation
    relu_acts = torch.zeros((n_relus, x.shape[0]))
    for i_relu in range(n_relus):
        relu_acts[i_relu, :] = torch.relu(x + b[i_relu])
    
    # Initialize weights for weighted sum of ReLUs
    combination_weights = torch.zeros((n_relus, ))
    prev_slope = 0
    for i in range(n_relus):
        delta_x = x_train[i+1] - x_train[i]
        slope = (y_train[i+1] - y_train[i]) / delta_x
        combination_weights[i] = slope - prev_slope
        prev_slope = slope
    
    y_hat = combination_weights @ relu_acts

    return y_hat, relu_acts, x

#################################################
### Coding Exercise 2 - A General-Purpose MLP ### 
#################################################

class Net(nn.Module):

    def __init__(self, actv, input_feature_num, hidden_unit_nums, 
            output_feature_num):

        super(Net, self).__init__()
        self.input_feature_num = input_feature_num
        self.mlp = nn.Sequential()

        in_num = input_feature_num
        for i in range(len(hidden_unit_nums)):

            out_num = hidden_unit_nums[i]
            layer = nn.Linear(in_num, out_num)
            in_num = out_num
            self.mlp.add_module('Linear_%d'%i, layer)

            actv_layer = eval('nn.%s'%actv)
            self.mlp.add_module('Activation_%d'%i, actv_layer)
        
        out_layer = nn.Linear(hidden_unit_nums[-1], output_feature_num)
        self.mlp.add_module('Output_Linear', out_layer)

    def forward(self, x):
        x = x.view(-1, self.input_feature_num)
        logits = self.mlp(x)
        return logits

######################################################
### Coding Exercise 2.1 - Batch Cross Entropy Loss ###
######################################################

def cross_entropy_loss(x, labels):
    # Initialize, later fill
    x_of_labels = torch.zeros(len(labels))
    for i, label in enumerate(labels):
        x_of_labels[i] = x[i, label]

    losses = -x_of_labels + torch.log(torch.sum(torch.exp(x), axis=1))
    av_loss = losses.mean()
    print(losses, av_loss)    
    return av_loss

def create_spiral_dataset(K, sigma, N):
    # Initialize t, X, y
    t = torch.linspace(0, 1, N)
    X = torch.zeros(K*N, 2)
    y = torch.zeros(K*N)

    # Create data
    for k in range(K):
        X[k*N:(k+1)*N, 0] = t*(torch.sin(2*np.pi/K*(2*t+k)) + sigma*torch.randn(N))
        X[k*N:(k+1)*N, 1] = t*(torch.cos(2*np.pi/K*(2*t+k)) + sigma*torch.randn(N))
        y[k*N:(k+1)*N] = k

    return X, y

def shuffle_and_split_data(X, y, seed):

    torch.manual_seed(seed)
    # Number of samples
    N = X.shape[0]
    # Shuffle data
    shuffled_indices = torch.randperm(N)
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    # Split data into train/test
    test_size = int(N * 0.2)
    X_test = X[:test_size]
    y_test = y[:test_size]
    X_train = X[test_size:]
    y_train = y[test_size:]
    
    return X_test, y_test, X_train, y_train

def train_test_classification(net, criterion, optimizer, train_loader,
                              test_loader, num_epochs=1, verbose=True,
                              training_plot=False, device='cpu'):
    net.train()
    training_losses = []
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.long()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # apply the defined loss function
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if verbose:
                training_losses += [loss.item()]
        
        net.eval()

    def test(data_loader):
        correct = 0
        total = 0
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.long()

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        return total, acc

    train_total, train_acc = test(train_loader)
    test_total, test_acc = test(test_loader)
    if verbose:
        print(f"Accuracy on the {train_total} training samples: {train_acc:0.2f}")
        print(f"Accuracy on the {test_total} testing samples: {test_acc:0.2f}")
    if training_plot:
        plt.plot(training_losses)
        plt.xlabel('Batch')
        plt.ylabel('Training loss')
        plt.show()

    return train_acc, test_acc

if __name__ == "__main__":
    SEED = 2021
    #################################
    ### Testing Coding Exercise 2 ###
    #################################
    '''
    input = torch.zeros((100, 2))
    net = Net(actv='LeakyReLU(0.1)', input_feature_num=2, 
            hidden_unit_nums=[100, 10, 5], output_feature_num=1)
    y = net(input)
    print(f'The output shape is {y.shape} for an input of shape {input.shape}')
    '''
    ###################################
    ### Testing Coding Exercise 2.1 ###
    ###################################
    '''
    labels = torch.tensor([0, 1])
    x = torch.tensor([[10.0, 1.0, -1.0, -20.0],  # correctly classified
                  [10.0, 10.0, 2.0, -10.0]])  # Not correctly classified
    CE = nn.CrossEntropyLoss()
    pytorch_loss = CE(x, labels).item()
    our_loss = cross_entropy_loss(x, labels).item()
    print(f'Our CE loss: {our_loss:0.8f}, Pytorch CE loss: {pytorch_loss:0.8f}')
    print(f'Difference: {np.abs(our_loss - pytorch_loss):0.8f}')
    '''
    ###################################
    ### Testing Coding Exercise 2.2 ###
    ###################################
    
    # Visualize our artificial dataset
    K = 4
    sigma = 0.16
    N = 1000
    set_seed(seed=SEED)
    X, y = create_spiral_dataset(K, sigma, N)
    #plt.scatter(X[:, 0], X[:, 1], c = y)
    #plt.show()

    ###########################
    ### Testing Section 2.3 ###
    ###########################

    X_test, y_test, X_train, y_train = shuffle_and_split_data(X, y, seed=SEED)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.title('Test data')
    plt.show()

    g_seed = torch.Generator()
    g_seed.manual_seed(SEED)

    batch_size = 128
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                            shuffle=False, num_workers=2,
                            worker_init_fn=seed_worker)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=2,
                            worker_init_fn=seed_worker)

    net = Net('ReLU()', X_train.shape[1], [128], K)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    num_epochs = 100

    train_test_classification(net, criterion, optimizer, train_loader,
                test_loader, num_epochs=num_epochs, training_plot=True)
