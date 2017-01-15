# Created by Mårten Nilsson to provide MSE prediction results on the full data sets with random pseudo input points 
import numpy as np
import matplotlib.pyplot as plt
from data_loader import hallucinate_data, load_data, load_test_data
from SPGP_alt import SPGP_alt

M = 20

# Kin40k data
X, T = load_data("kin40k")
process = SPGP_alt(X, T)
print(X.shape)

# Select pseudo-inputs
#process.sigma_sq = 0.0047 ** 2
process.sigma_sq = 1
c = 0.4828
b = np.array([0.0111, 0.0085, 0.3714, 0.2144, 0.2346, 0.5437, 0.5547, 0.2769])
process.hyp = [c, b]
process.set_kernel()

# Get test data
X_tst, T_tst = load_test_data("kin40k")
print(X_tst.shape)

process.do_precomputations()
print("Precomputations performed")
T_inferred = process.get_predictive_mean(X_tst)

# Calculate mean_square_error
error = T_tst - T_inferred

# Compare to regular MSE
error_baseline = T_tst - np.mean(T_tst)

print("Mean square error: ", np.mean(error ** 2))
print("Compare to: ", np.mean(error_baseline ** 2))
print("Finished")


# Do a loop and save values

Ms = np.array([20, 40, 60, 80, 100, 200, 300, 400, 500, 750, 1000, 1250])
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

