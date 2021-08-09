import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ipywidgets import interact, IntSlider, FloatSlider, fixed
from ipywidgets import HBox, interactive_output, ToggleButton, Layout
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/content-creation/main/nma.mplstyle")

###############################################
### Coding Exercise 1.1 - Simple Narrow LNN ###
###############################################

class ShallowNarrowExercise:
    
    def __init__(self, init_weights):
        assert isinstance(init_weights, (list, np.ndarray, tuple))
        assert len(init_weights) == 2
        self.w1 = init_weights[0]
        self.w2 = init_weights[1]
    
    def forward(self, x):
        y = self.w2 * self.w1 * x
        return y
    
    def dloss_dw(self, x, y_true):
        assert x.shape == y_true.shape
        dloss_dw1 = -(2 * self.w2 * x * (y-self.w1*self.w2*x)).mean()
        dloss_dw2 = -(2 * self.w1 * x * (y-self.w1*self.w2*x)).mean()

        return dloss_dw1, dloss_dw2

   def train(self, x, y_true, lr, n_ep):
       assert x.shape == y_true.shape
       loss_records = np.empty(n_ep)  # pre allocation of loss records
       weight_records = np.empty((n_ep, 2))  # pre allocation of weight records

       for i in range(n_ep):
           y_prediction = self.forward(x)
           loss_records[i] = loss(y_prediction, y_true)
           dloss_dw1, dloss_dw2 = self.dloss_dw(x, y_true)
           # Update weights
           self.w1 -= dloss_dw1 * lr
           self.w2 -= dloss_dw2 * lr
           weight_records[i] = [self.w1, self.w2]
        return loss_records, weight_records

def loss(y_prediction, y_true):
    assert y_prediction.shape == y_true.shape
    mse = ((y_prediction - y_true)**2).mean()
    return mse

