from SPGP_alt import SPGP_alt
from data_loader import hallucinate_data

X, T = hallucinate_data(1, 100)  #1D-data is plottable

# Create the process
process = SPGP_alt(X, T)
process.check_gradient()