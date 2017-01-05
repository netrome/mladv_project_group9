# Created by Mårten 2017-01-5 for the purpose of getting to know the data
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

data_folder = "../Data/"

def load_data(name):
    # Load the kin40k training data
    mat = loadmat(data_folder + name)
    X = mat["X_tr"]
    T = mat["T_tr"]
    return X, T


# Plot the first dimension of the data
# X, T = load_data("kin40k.mat")
X, T = load_data("pumadyn32nm.mat")
for i in range(X.shape[1]):
    plt.plot(X[:, i], T, '*')
    plt.show()

