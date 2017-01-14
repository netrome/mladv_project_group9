import numpy as np
import matplotlib.pyplot as plt
from data_loader import hallucinate_data, load_data, load_original_toy_data
from SPGP_alt import SPGP_alt

X, T = load_data("kin40k")
process = SPGP_alt(X, T)
plt.plot(process.pseudo_inputs[:, 0], np.ones(process.M) * 0.1 + np.min(T), 'r+')
process.optimize_hyperparameters()
process.set_kernel()
process.do_precomputations()
process.do_differential_precomputations()
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