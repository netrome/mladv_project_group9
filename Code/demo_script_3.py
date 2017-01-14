# Created by Mårten Nilsson 2017-01-09 to illustrate a toy example of the SPGP implementation.
import numpy as np
import matplotlib.pyplot as plt
from data_loader import hallucinate_data, load_data, load_original_toy_data
from SPGP_alt import SPGP_alt

X, T = load_original_toy_data() 
X_b = np.linspace(0, 2, 20)    # Set pseudo inputs
X_b = np.reshape(X_b, [20, 1])

print(X)


# Create the process
process = SPGP_alt(X, T)
process.pseudo_inputs = X_b
process.do_precomputations()
process.do_differential_precomputations()
print(process.log_likelihood())
print()
plt.plot(process.pseudo_inputs[:, 0], np.ones(process.M) * 0.1 + np.min(T), 'r+')
process.optimize_hyperparameters()
process.do_precomputations()
print()
print(process.log_likelihood())

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
plt.plot(process.pseudo_inputs[:, 0], np.zeros(process.M) + np.min(T), 'b+')
plt.show()

