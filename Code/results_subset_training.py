# Created by Mårten Nilsson to provide MSE prediction results on the full data sets with training on a subset of the original data
import numpy as np
import matplotlib.pyplot as plt
from data_loader import hallucinate_data, load_data, load_test_data, reduce_data
from SPGP_alt import SPGP_alt

M = 200

# Kin40k data
X, T = load_data("kin40k")
X_red = reduce_data(X, 800)
T_red = reduce_data(T, 800)
opt_gp = SPGP_alt(X_red, T_red)

# Optimeze pseudo-inputs and kernel/noise parameters
opt_gp.optimize_hyperparameters()

# Real process
process = SPGP_alt(X, T)
process.pseudo_inputs = opt_gp.pseudo_inputs
process.hyp = opt_gp.hyp
process.sigma_sq = opt_gp.sigma_sq

# Get test data
X_tst, T_tst = load_test_data("kin40k")

process.do_precomputations()
print("Precomputations performed")
T_inferred = process.get_predictive_mean(X_tst)

# Calculate mean_square_error
error = T_tst - T_inferred

print("Mean square error: ", np.mean(error ** 2))
print("Finished")
