import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tensorflow as tf

class fullnn:
    """
    This class is used to generate a fully L-layer and W-node-per-layer
    connected neural network that can generate and predict binary labels
    and be trained using SGD with minibatch.
    The nonlinear node simply use ReLU. And the loss function uses log
    loss.
    """
    def __init__(self,dim,width,)
