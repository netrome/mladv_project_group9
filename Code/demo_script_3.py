# Created by Mårten Nilsson 2017-01-09 to illustrate a toy example of the SPGP implementation.
import numpy as np
import matplotlib.pyplot as plt
import cProfile as cp
from data_loader import hallucinate_data, load_data, load_original_toy_data
from SPGP_alt import SPGP_alt
import pstats

benchmark_file_name = "demo_script_3_benchmark"
#X, T = load_original_toy_data() 
X, T = hallucinate_data(1,1000)

X_b = np.linspace(0, 1, 9)   # Set pseudo inputs
X_b = np.reshape(X_b, [9, 1])

print(X)


# Create the process
process = SPGP_alt(X, T)
#process.pseudo_inputs = X_b
#rocess.M = 9
process.do_precomputations()
process.do_differential_precomputations()
print(process.log_likelihood())
print()
plt.plot(process.pseudo_inputs[:, 0], np.ones(process.M) * 0.1 + np.min(T), 'r+')
cp.run("process.optimize_hyperparameters()",filename=benchmark_file_name)
process.do_precomputations()
print(process.log_likelihood())

# Get the predictive mean
X2 = np.reshape(np.linspace(-2, 8, 200), (200, 1))
mean = process.get_predictive_mean(X2)
var = process.get_predictive_variance(X2)
std = np.sqrt(var)

# Plot first dimension of data
plt.plot(X[:, 0], T, 'r*')
plt.plot(X2, mean, 'g')
plt.plot(X2, mean + std, 'm')
plt.plot(X2, mean - std, 'm')
plt.plot(process.pseudo_inputs[:, 0], np.zeros(process.M) + np.min(T), 'b+')

plt.xlabel("input")
plt.ylabel("output")
plt.show()

p = pstats.Stats(benchmark_file_name)
p.strip_dirs().sort_stats("cumtime").print_stats()
