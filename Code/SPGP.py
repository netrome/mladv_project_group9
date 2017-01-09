# Created by Mårten Nilsson 2017-01-09. 
# This file contains an object oriented implementation of the sparse pseudo-input Gaussian process..
import numpy as np

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
                K[i, j] = c * np.exp( (-1/2) * ( b.dot( ( (x1[i] - x2[j])**2 ) ) ) )
        return K
    return kernel

class SPGP:
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
        self.sigma = 0.2
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
        """
        Do any calculation that can be performed without new data and save relevant results
        """
        K_M = self.kernel(self.pseudo_inputs, self.pseudo_inputs) + 1e-6*np.eye(self.n) # Added jitter
        self.K_M_inv = np.linalg.inv(K_M)
        #L = np.linalg.cholesky(K_M)  - Cholesky decomposition
        K_MN = self.kernel(self.pseudo_inputs, self.X_tr)
        K_NM = np.transpose(K_MN)
        K_N = self.kernel(self.X_tr, self.X_tr)

        Q_N = K_NM.dot( self.K_M_inv.dot( K_MN ) )
        Lambda_sigma = np.diag(np.diag(K_N - Q_N) + self.sigma**2) 
        LS_inv = np.linalg.inv(Lambda_sigma)

        B = K_M + K_MN.dot( LS_inv ).dot(K_NM)

        # Save stuff to be used in predictions
        self.B_inv = np.linalg.inv(B)
        self.alpha = self.B_inv.dot( K_MN.dot( LS_inv.dot( self.Y_tr ) ) )

    def get_predictive_mean(self, x_input): 
        """
        Returns the predictive mean for the inputs. Relies on the precomputed alpha values.
        """
        K = self.kernel(x_input, self.pseudo_inputs)
        return K.dot(self.alpha)

    def get_predictive_variance(self, x_input):
        """
        Returns the predictive variance for the inputs.
        """
        var = np.zeros([x_input.shape[0], 1])
        for i, x in enumerate(x_input):
            x = np.reshape(x, [1, x.size])
            K_star = self.kernel(x, x)
            K_starM = self.kernel(x, self.pseudo_inputs)
            K_Mstar = np.transpose(K_starM)

            var[i] = K_star - K_starM.dot(self.K_M_inv - self.B_inv).dot(K_Mstar) + self.sigma**2
        return var

    def optimize_hyperparameters(self):
        # TODO - gradient descent for pseudo inputs, noise parameter and kernel parameters
        return

