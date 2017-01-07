# Created by Mårten 2017-01-5 for the purpose of getting to know the data and making some prototypes.
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

data_folder = "../Data/"


def load_data(name):
    """
    Load the training data from one of the datasets.
    """
    mat = loadmat(data_folder + name)
    X = mat["X_tr"]
    T = mat["T_tr"]
    return X, T


def hallucinate_data(dim, n):
    """
    Create simple data to work with where X is a [n x dim] matrix and Y is a [n x 1] matrix.
    """
    X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)
    Y = np.sin(X.sum(1)*2)
    Y = np.reshape(Y, [Y.size, 1])
    return X, Y


def get_kernel_function(hyp):
    """ 
    Returns a squared exponential covariance function based on the hyperparameters.
    hyp is assued to be a list [c, b] according to Haos notation in the Algoritms/math.pdf.
    """
    c = hyp[0]
    b = hyp[1]
    def kernel(x1, x2):
        """
        x1 is assumed to be of shape [n x i].
        x2 is assumed to be of shape [n x j].
        This function returns the matrix K so that K[i, j] = K(xi, xj).
        This can possibly be optimized by the use of matrix algebra instead of for loops.
        """
        K = np.zeros([x1.shape[0], x2.shape[0]])
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                K[i, j] = c * np.exp( (-1/2) * ( b @ ((x1[i] - x2[j])**2) ) )
        return K
    return kernel


class SPGP_blubb:
    """
    Suggested skeleton for object oriented implementation of the SPGP
    """

    def __init__(self, X_tr, Y_tr):
        """
        Sets arbitrary hyperparameters, noise and pseudo inputs so that the model works.
        Requires the data to have more than 20 data points
        """
        self.N, self.dim = X_tr.shape
        self.n = 20
        self.hyp = [1, np.ones(self.dim)]
        self.sigma = 1
        self.kernel = get_kernel_function(self.hyp)
        self.pseudo_inputs = X_tr[np.random.randint(0, X_tr.shape[0], self.n)] 
        self.X_tr = X_tr
        self.Y_tr = Y_tr


    def set_kernel(self, hyp):
        """
        Updates the kernel function with new hyperparameters
        """
        self.hyp = hyp
        self.kernel = get_kernel_function(hyp)


    def do_precomputations(self):
        K_M = self.kernel(self.pseudo_inputs, self.pseudo_inputs) + 1e-6*np.eye(self.n) # Added jitter
        #L = np.linalg.cholesky(K_M) Maybe not necessary
        K_MN = self.kernel(self.pseudo_inputs, self.X_tr)
        K_NM = np.transpose(K_MN)
        K_N = self.kernel(self.X_tr, self.X_tr)

        Q_N = K_NM.dot( np.linalg.inv(K_M).dot( K_MN ) )
        Lambda_sigma = np.diag(np.diag(K_N - Q_N) + self.sigma**2) 
        LS_inv = np.linalg.inv(Lambda_sigma)

        B = K_M + K_MN.dot( LS_inv ).dot(K_NM)

        # Save stuff to be used in predictions
        self.B_inv = np.linalg.inv(B)
        self.alpha = self.B_inv.dot( K_MN.dot( LS_inv.dot( self.Y_tr ) ) )

    def get_predictive_mean(self, x_input): 
        K = self.kernel(x_input, self.pseudo_inputs)
        return K.dot(self.alpha)


# Get the data
# X, T = load_data("kin40k.mat")
X, T = load_data("pumadyn32nm.mat")
X, T = hallucinate_data(1, 100)  #1D-data is plottable

# Create the process
process = SPGP_blubb(X, T)
process.do_precomputations()

# Get the predictive mean
X2 = np.linspace(-8, 8, 2000)
mean = process.get_predictive_mean(X2)

# Plot first dimension of data
plt.plot(X[:, 0], T, 'r*')
plt.plot(X2, mean, 'g')
plt.show()
