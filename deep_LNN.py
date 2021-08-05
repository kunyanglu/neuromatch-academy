import math
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

class LNNet(nn.Module):

  def __init__(self, in_dim, hid_dim, out_dim):
    super().__init__()
    self.in_hid = nn.Linear(in_dim, hid_dim, bias=False)
    self.hid_out = nn.Linear(hid_dim, out_dim, bias=False)

  def forward(self, x):
    hid = self.in_hid(x)  # hidden activity
    out = self.hid_out(hid)  # output (prediction)
    return out, hid

def train(model, inputs, targets, n_epochs, lr, illusory_i=0):
    in_dim = inputs.size(1)
    
    losses = np.zeros(n_epochs)  # loss records
    modes = np.zeros((n_epochs, in_dim))  # singular values (modes) records
    rs_mats = []  # representational similarity matrices
    illusions = np.zeros(n_epochs)  # prediction for the given feature

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for i in range(n_epochs):
        optimizer.zero_grad()
        predictions, hiddens = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        # Singular Value Decomposition
        U, Σ, V = net_svd(model, in_dim)
        
        # Representational Similarity Matrix
        RSM = net_rsm(hiddens.detach())

        # Network prediction of illusory_i inputs for the last feature:
        # have bones or not have bones
        pred_ij = predictions.detach()[illusory_i, -1]

        # Logging (recordings)
        losses[i] = loss.item()
        modes[i] = Σ.detach().numpy()
        rs_mats.append(RSM.numpy())
        illusions[i] = pred_ij.numpy()

    return losses, modes, np.array(rs_mats), illusions

def initializer_(model, gamma=1e-12):

  for weight in model.parameters():
    n_out, n_in = weight.shape
    sigma = gamma / math.sqrt(n_out + n_in)
    nn.init.normal_(weight, mean=0.0, std=sigma)

def net_svd(model, in_dim):
    W_tot = torch.eye(in_dim)
    for weight in model.parameters():
        W_tot = W_tot @ weight
    
    U, Σ, V = torch.svd(W_tot)
    return U, Σ, V

def net_rsm(h):
    rsm = h @ h.T
    return rsm


