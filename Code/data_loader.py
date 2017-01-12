# Created by Mårten Nilsson 2017-01-09.
# Holds functions for loading data
import numpy as np 
from scipy.io import loadmat, savemat

data_folder = "../Data/"

def load_data(name):
    """
    Load the training data from one of the datasets.
    """
    mat = loadmat(data_folder + name)
    X = mat["X_tr"]
    T = mat["T_tr"]
    return X, T


def hallucinate_data(dim, n):
    """
    Create simple data to work with where X is a [n x dim] matrix and Y is a [n x 1] matrix.
    """
    X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)
    Y = np.sin(X.sum(1)*2) + np.random.normal(0, 0.8, n)
    Y = np.reshape(Y, [Y.size, 1])
    return X, Y

