import numpy as np
import matplotlib.pyplot as plt
from data_loader import hallucinate_data, load_data
from SPGP_alt import SPGP_alt

X, T = hallucinate_data(1, 100)  #1D-data is plottable
process = SPGP_alt(X, T)
process.do_precomputations()
process.do_differential_precomputations()
