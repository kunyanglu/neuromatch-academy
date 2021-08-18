import torch
import random

import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from pytorch_pretrained_biggan import BigGAN
from pytorch_pretrained_biggan import one_hot_from_names
from pytorch_pretrained_biggan import truncated_noise_sample

from tqdm.notebook import tqdm, trange

SEED = 2021

################################
### Coding Exercise 2 - pPCA ###
################################

### Generate random normally distributed data ###
def generate_data(n_samples, mean_of_temps, cov_of_temps, seed):
    np.random.seed(seed)
    therm1, therm2 = np.random.multivariate_normal(mean_of_temps,
                                    cov_of_temps,n_samples).T
    return therm1, therm2

### Calculate parameters of pPCA model ###
def get_pPCA_parameters(pc_axes, therm1, therm2):

    # thermometers data
    therm_data = np.array([therm1, therm2])

    # Zero center the data
    therm_data_mean = np.mean(therm_data, 1)
    therm_data_center = np.outer(therm_data_mean, np.ones(therm_data.shape[1]))
    therm_data_zero_centered = therm_data - therm_data_center

    # Calculate the variance of the projection on the PC axes
    pc_projection = np.matmul(pc_axes, therm_data_zero_centered);
    pc_axes_variance = np.var(pc_projection)

    # Calculate the residual variance (variance not accounted for by projection on the PC axes)
    sensor_noise_std = np.mean(np.linalg.norm(therm_data_zero_centered - np.outer(pc_axes, pc_projection), axis=0, ord=2))
    sensor_noise_var = sensor_noise_std **2
    
    return sensor_noise_var, therm_data_mean, pc_axes_variance

def gen_from_pPCA(noise_var, data_mean, pc_axes, pc_variance):

    n_samples = 1000

    # Randomly sample from z (latent space value)
    # `z` has shape (1000,)
    z = np.random.normal(0.0, np.sqrt(pc_variance), n_samples)

    # sensor noise covariance matrix (∑)
    epsilon_cov = [[noise_var, 0.0], [0.0, noise_var]]

    # data mean reshaped for the generation
    # `sim_mean` has shape (2,1000)
    sim_mean = np.outer(data_mean, np.ones(n_samples))
    
    rand_eps = np.random.multivariate_normal([0.0, 0.0], epsilon_cov, n_samples)
    rand_eps = rand_eps.T

    therm_data_sim = sim_mean + np.outer(pc_axes, z) + rand_eps
    return therm_data_sim

if __name__ == "__main__":

    # biggan_model = torch.load(fname)

    ##############################
    ### Test Coding Exercise 2 ###
    ##############################
    
    ### Plot randomly generated data
    n_samples = 2000
    mean_of_temps = np.array([25, 25])
    cov_of_temps = np.array([[10, 5], [5, 10]])
    therm1, therm2 = generate_data(n_samples, mean_of_temps, cov_of_temps, seed=SEED)

    ### Add 1st PC axis to the plot
    plt.plot(therm1, therm2, '.')
    plt.axis('equal')
    plt.xlabel('Thermometer 1 ($^\circ$C)')
    plt.ylabel('Thermometer 2 ($^\circ$C)')
    plt.plot([plt.axis()[0], plt.axis()[1]],
             [plt.axis()[0], plt.axis()[1]])
    plt.show()

    pc_axes = np.array([1.0, 1.0]) / np.sqrt(2.0)
    
    sensor_noise_var, therm_data_mean, pc_axes_variance = get_pPCA_parameters(pc_axes, therm1, therm2)
    therm_data_sim = gen_from_pPCA(sensor_noise_var, therm_data_mean, pc_axes, pc_axes_variance)
    print(therm_data_sim[0:2])
