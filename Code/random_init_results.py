# Created by Mårten Nilsson to provide MSE prediction results on the full data sets with random pseudo input points 
import numpy as np
import matplotlib.pyplot as plt
from data_loader import hallucinate_data, load_data, load_test_data
from SPGP_alt import SPGP_alt

M = 20

# Kin40k data
X, T = load_data("kin40k")
process = SPGP_alt(X, T)

# Select pseudo-inputs
process.M = M
process.update_random_pseudo_inputs(X)

# Get test data
X_tst, T_tst = load_test_data("kin40k")

process.do_precomputations()
print("Precomputations performed")
T_inferred = process.get_predictive_mean(X_tst)

# Calculate mean_square_error
error = T_tst - T_inferred

print("Mean square error: ", np.mean(error ** 2))
print("Finished")


# Do a loop and save values

Ms = np.array([20, 40, 60, 80, 100, 200, 300, 400])
vals = np.zeros(len(Ms))

for i, M in enumerate(Ms):
    process.M = M
    process.update_random_pseudo_inputs(X)
    
    process.do_precomputations()
    T_inferred = process.get_predictive_mean(X_tst)
    
    error = T_tst - T_inferred
    val = np.mean(error ** 2)
    vals[i] = val
    
    print("Finished with M = ", M)

print(Ms)
print(vals)

plt.plot(Ms, vals, 'rs')
plt.show()

plt.semilogy(Ms, vals, 'rs')
plt.show()

