# Created by Mårten Nilsson 2017-01-09. 
# This file contains an object oriented implementation of the sparse pseudo-input Gaussian process..
import numpy as np
from numpy.linalg import det, inv, norm, cholesky
from scipy.optimize import check_grad, approx_fprime
from scipy.optimize import fmin_tnc, fmin_cg

def get_kernel_function(hyp):
    """ 
    Returns a squared exponential covariance function based on the hyperparameters.
    hyp is assued to be a list [c, b] according to Haos notation in the Algoritms/math.pdf.
    """
    c = hyp[0]
    b = hyp[1]
    def kernel(x1, x2):
        """
        x1 is assumed to be of shape [dim x i].
        x2 is assumed to be of shape [dim x j].
        b is assumed to be of shape [dim].
        This function returns the matrix K so that K[i, j] = K(xi, xj).
        This can possibly be optimized by the use of matrix algebra instead of for loops.
        """
        K = np.zeros([x1.shape[0], x2.shape[0]])
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                K[i, j] = c * np.exp( (-1/2) * ( b.dot( ( (x1[i] - x2[j])**2 ) ) ) )
        return K

    def diag_kernel(x1, x2):
        """
        x1 is assumed to be of shape [dim x i].
        x2 is assumed to be of shape [dim x i].
        b is assumed to be of shape [dim].
        This function returns the matrix K so that K[i, j] = K(xi, xj).
        This can possibly be optimized by the use of matrix algebra instead of for loops.
        """
        if (not (x1.shape == x2.shape)):
            raise ValueError("Input need to be symmetric in diag_kernel")

        K = np.zeros([x1.shape[0], x2.shape[0]])
        for i in range(x1.shape[0]):
                K[i, i] = c * np.exp( (-1/2) * ( b.dot( ( (x1[i] - x2[i])**2 ) ) ) )
        return K

    return kernel, diag_kernel

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
        self.M = 20
        #self.hyp = (np.random.rand(), np.random.rand(self.dim))
        self.hyp = (1,np.ones(self.dim))
        self.sigma = 0.2
        self.kernel, self.diag_kernel = get_kernel_function(self.hyp)
        self.pseudo_inputs = X_tr[np.random.randint(0, X_tr.shape[0], self.M)] 
        #self.pseudo_inputs = np.linspace(np.min(X_tr),np.max(X_tr),self.M)[:,np.newaxis]
        self.X_tr = X_tr
        self.Y_tr = Y_tr


    def set_kernel(self, hyp):
        """
        Updates the kernel function with new hyperparameters
        """
        self.hyp = hyp
        self.kernel, self.diag_kernel = get_kernel_function(hyp)


    def do_precomputations(self):
        """
        Do any calculation that can be performed without new data and save relevant results
        """
        K_M = self.kernel(self.pseudo_inputs, self.pseudo_inputs) + 1e-6*np.eye(self.M) # Added jitter
        K_M_inv = np.linalg.inv(K_M)
        #L = np.linalg.cholesky(K_M)  - Cholesky decomposition
        K_MN = self.kernel(self.pseudo_inputs, self.X_tr)
        K_NM = np.transpose(K_MN)
        K_N = self.diag_kernel(self.X_tr, self.X_tr) # Note, only copute the diagonal!

        Q_N = K_NM.dot(K_M_inv.dot( K_MN ) )
        Lambda_sigma = np.diag(np.diag(K_N - Q_N) + self.sigma**2) 
        Gamma = Lambda_sigma / (self.sigma ** 2)
        LS_inv = np.linalg.inv(Lambda_sigma)
        B = K_M + K_MN.dot( LS_inv ).dot(K_NM)

        # Save stuff to be used in predictions
        self.B_inv = np.linalg.inv(B)
        self.A = (self.sigma ** 2) * B
        self.A_sqrt = cholesky(self.A + np.eye(self.A.shape[0])*0.0000000001) 
        self.alpha = self.B_inv.dot( K_MN.dot( LS_inv.dot( self.Y_tr ) ) )
        self.K_M_inv = K_M_inv
        self.K_M = K_M
        self.K_N = K_N
        self.K_MN = K_MN
        self.K_NM = K_NM
        self.Gamma = Gamma
        self.Q_N = Q_N

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

    def optimize_hyperparameters_numerical(self):
        # TODO - gradient descent for pseudo inputs, noise parameter and kernel parameters
        foo = SPGP(self.X_tr, self.Y_tr)

        def f(X):
            X_bar = np.reshape(X[:self.M*self.dim], (self.M, self.dim))
            Theta = X[self.M*self.dim:]
            Theta = (Theta[0], np.array(Theta[1:]))
            foo.pseudo_inputs = X_bar
            foo.set_kernel(Theta)
            foo.do_precomputations()
            return foo.log_likelihood()

        param = fmin_cg(f, np.concatenate((np.reshape(self.pseudo_inputs, (self.M*self.dim)), [self.hyp[0]], self.hyp[1]), axis=0),epsilon = .000001,maxiter=35)

#        print("nfeval:", nfeval)

        X_bar = np.reshape(param[:self.M*self.dim], (self.M, self.dim))
        Theta = param[self.M*self.dim:]
        Theta = (Theta[0], np.array(Theta[1:]))

        return X_bar, Theta


    def optimize_hyperparameters(self):

        # Sigma optimization implemented
        l = 0.01 # Arbitrary value
        for i in range(150):
            self.do_precomputations()
            dds = self.derivate_sigma() 
            self.sigma -= np.sign(dds[0, 0])*l
            print(self.sigma)
            print(self.log_likelihood())
        return

    def derivate_log_likelihood(self):
        # TODO - return the gradient with respect to the hyperparameters
        return

    def derivate_log_likelihood_numerical(self,X):
        foo = SPGP(self.X_tr,self.Y_tr)
        def f(X):
            X_bar = X[:self.M]
            Theta = X[self.M:]
            foo.pseudo_inputs = X_bar
            foo.set_kernel(Theta)
            foo.do_precomputations()
            return foo.log_likelihood()
        return approx_fprime(X,f,.001)

    def derivate_sigma(self):
        Gamma = self.Gamma
        Gamma_sqrt = cholesky(Gamma)
        sigma_sq = self.sigma ** 2
        A_inv = np.linalg.inv(self.A)
        K_MN = self.K_MN.dot( inv(Gamma_sqrt) ) # Implicit underscore
        K_NM = np.transpose(K_MN)
        y = inv(Gamma_sqrt).dot(self.Y_tr)  

        dL1 = (1 / sigma_sq) * (np.trace(inv(Gamma)) - np.trace(K_NM.dot( 
                                                        A_inv ).dot( K_MN )))
        dL1 /= 2

        dL2 = norm(y) ** 2 + norm(K_NM.dot( A_inv ).dot( K_MN ).dot( y )) ** 2 
        dL2 -= 2 * y.transpose().dot( K_NM ).dot( A_inv ).dot( K_MN ).dot( y )
        dL2 *= -(1 /( sigma_sq ** 2) )
        dL2 /= 2
        return dL1 + dL2

    def log_likelihood(self):
        # returns the log likelihood of the marginal for y
        K_M = self.K_M
        sigma = self.sigma
        K_MN = self.K_MN
        K_NM = self.K_NM
        Gamma = self.Gamma 
        Gamma_sqrt = cholesky(Gamma)
        A = self.A
        A_sqrt = self.A_sqrt
        N = self.N
        M = self.M
        y = self.Y_tr
        y_under = inv(Gamma_sqrt).dot(y)  
        K_MN_under = K_MN.dot( inv(Gamma_sqrt) )

        L1 = np.log(det(A)) - np.log(det(K_M)) + np.log(det(Gamma)) + (N - M) * np.log(sigma ** 2)
        L1 /= 2
        L2 = (sigma ** -2) * (norm(y_under) ** 2 - norm(inv(A_sqrt).dot(K_MN_under).dot(y_under)) ** 2)
        L2 /= 2
        return L1 + L2 #+(N/2 * np.log(2 * np.pi))

