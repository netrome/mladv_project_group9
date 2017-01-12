# Created by Mårten Nilsson 2017-01-09 to illustrate a toy example of the SPGP implementation.
import numpy as np
import matplotlib.pyplot as plt
from data_loader import hallucinate_data, load_data
from SPGP import SPGP

X, T = hallucinate_data(1, 100)  #1D-data is plottable

# Create the process
process = SPGP(X, T)
process.do_precomputations()

# Get the predictive mean
X2 = np.reshape(np.linspace(-8, 8, 200), (200, 1))
mean = process.get_predictive_mean(X2)
var = process.get_predictive_variance(X2)
std = np.sqrt(var)

# Plot first dimension of data
plt.plot(X[:, 0], T, 'r*')
plt.plot(X2, mean, 'g')
plt.plot(X2, mean + std, 'm')
plt.plot(X2, mean - std, 'm')
plt.plot(process.pseudo_inputs[:, 0], np.zeros(process.n) + np.min(T), 'b+')
plt.show()

# With real data, but no plot
X, T = load_data("kin40k")
process = SPGP(X, T)
process.do_precomputations()
print("Finished with kin40k")

X, T = load_data("pumadyn32nm")
process = SPGP(X, T)
process.do_precomputations()
print("Finished with pumadyn")
