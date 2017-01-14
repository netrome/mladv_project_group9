# Alternative version of the SPGP class to avoid merge conflicts
# Created by Mårten Nilsson 2017-01-09. 
# This file contains an object oriented implementation of the sparse pseudo-input Gaussian process..
import functools as ft
import numpy as np
from numpy.linalg import det, inv, norm, cholesky
from scipy.optimize import check_grad, approx_fprime
from scipy.optimize import fmin_tnc, fmin_cg


np.seterr(all = "raise")

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

class SPGP_alt:
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
        self.hyp = [1.2, np.zeros(self.dim) + 1.001]
        self.sigma_sq = 5
        self.kernel, self.diag_kernel = get_kernel_function(self.hyp)
        self.pseudo_inputs = X_tr[np.random.randint(0, X_tr.shape[0], self.M)] 
        #self.pseudo_inputs = np.linspace(np.min(X_tr),np.max(X_tr),self.M)[:,np.newaxis]
        self.X_tr = X_tr
        self.Y_tr = Y_tr


    def set_kernel(self, hyp=None):
        """
        Updates the kernel function with new hyperparameters
        """
        if hyp is None:
            hyp = self.hyp
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
        Lambda_sigma = np.diag(np.diag(K_N - Q_N) + self.sigma_sq)
        Gamma = Lambda_sigma / (self.sigma_sq)
        LS_inv = np.diag(np.diag(Lambda_sigma) ** (-1))
        B = K_M + K_MN.dot( LS_inv ).dot(K_NM)

        # Save stuff to be used in predictions
        self.B_inv = np.linalg.inv(B)
        self.A = (self.sigma_sq) * B
        self.A_sqrt = cholesky(self.A + np.eye(self.A.shape[0])*0.000001) 
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

            var[i] = K_star - K_starM.dot(self.K_M_inv - self.B_inv).dot(K_Mstar) + self.sigma_sq
        return var

    def do_differential_precomputations(self):
        """
        Precomputations for the derivatives, used in the optimization
        """
        self.K_N = self.diag_kernel(self.X_tr, self.X_tr) # Note, only copute the diagonal!
        self.K_M = self.kernel(self.pseudo_inputs, self.pseudo_inputs) + 1e-6*np.eye(self.M)
        self.K_M_sqrt = cholesky(self.K_M)
        self.K_M_sqrt_inv = inv(self.K_M_sqrt)
        self.K_M_inv = inv(self.K_M)
        self.K_MN = self.kernel(self.pseudo_inputs, self.X_tr)
        self.K_NM = np.transpose(self.K_MN)
        self.Q_N = self.K_NM @ self.K_M_inv @ self.K_MN

        self.Gamma = np.diag(np.diag(self.K_N - self.Q_N)/self.sigma_sq + 1)
        self.Gamma_inv = np.diag(np.diag(self.Gamma) ** (-1))    # It's diagonal
        self.Gamma_sqrt = self.Gamma ** (1/2)   
        self.Gamma_sqrt_inv = np.diag(np.diag(self.Gamma) ** (-1/2))
        
        self.y_ = self.Gamma_sqrt_inv @ self.Y_tr
        self.K_NM_ = self.Gamma_sqrt_inv @ self.K_NM
        self.K_MN_ = self.K_NM_.transpose()

        self.A = self.sigma_sq * self.K_M + self.K_MN @ self.Gamma_inv @ self.K_NM 
        self.A_inv = inv(self.A)
        self.A_sqrt = cholesky(self.A)
        self.A_sqrt_inv = inv(self.A_sqrt)

    def optimize_hyperparameters(self):
        
        l = 0.01
        iters = 580
        for i in range(iters):
            self.do_differential_precomputations()      #PRECOMPUTATIONS - IMPORTAAAAANT
            
            dss, dhyp, dxb = self.derivate_log_likelihood(self.sigma_sq,self.hyp,self.pseudo_inputs)

            # Ugly hack
            dss = np.sign(dss) * 10 * (np.exp((iters-i)/2/iters))
            
            # Update sigma_square
            print("sigma_sq, ", self.sigma_sq)
            self.sigma_sq -= l*dss
            self.sigma_sq = np.abs(self.sigma_sq)
            
            # Update hyp
            self.hyp[0] -= l*dhyp[0]
#            self.hyp[1] += l*np.sign(dhyp[1]) * 10 * (np.exp((iters-i)/2/iters))
            self.hyp[1] -= l*dhyp[1] 
            #self.hyp[1] = np.abs(self.hyp[1])
            
            print("dxb, ", dxb[4])
            print("xb, ", self.pseudo_inputs[4])
                        
            self.pseudo_inputs -= l * dxb
            
            # Hack the b:s
            #self.hyp[1][self.hyp[1] < 0] = 0
            print("db", dhyp[1])
            print("b: ", self.hyp[1])
            print("c: ", self.hyp[0])
            self.set_kernel()
            
        return

    def derivate_log_likelihood_packed(self,X):
        sigma_sq, hyp, pseudo_inputs = unpack_params(X,self.dim,self.M)
        print(pseudo_inputs.shape)
        dsigma_sq,dhyp,dpseudo_inputs = self.derivate_log_likelihood(sigma_sq,hyp,pseudo_inputs)
        return pack_params(dsigma_sq,dhyp,dpseudo_inputs,self.M,self.dim)

    def derivate_log_likelihood(self,sigma_sq,hyp,pseudo_inputs):
        """
        Giant gradient calculations
        """

        dss = self.derivate_sigma(sigma_sq)
        dhyp = [0, np.zeros(self.dim)]
        dhyp[0] = self.derivate_nasty(self.derivate_c(hyp),sigma_sq)

        for i in range(len(dhyp[1])):
            val = self.derivate_nasty(self.derivate_b(i,pseudo_inputs),sigma_sq)
            dhyp[1][i] = val 
            
        dxb = np.zeros([self.M, self.dim])
        
        for m in range(self.M):
            for d in range(self.dim):
                val = self.derivate_nasty(self.derivate_kernel(m, d,hyp,pseudo_inputs),sigma_sq)
                dxb[m, d] = val
                
        return dss, dhyp, dxb

    def derivate_nasty(self, dKs,sigma_sq):
        """ Do nasty stuff """
        dK_M, dK_N, dK_NM = dKs
        dK_MN = dK_NM.T
        dK_MN_ = dK_MN @ self.Gamma_sqrt_inv
        


        dGamma = (1 / sigma_sq) * np.diag(np.diag( dK_N - 2*dK_NM.dot(self.K_M_inv).dot(self.K_MN) +
                            self.K_NM.dot(self.K_M_inv).dot(dK_M).dot(self.K_M_inv).dot(self.K_MN) ))
        dA = sigma_sq * dK_M + 2 * (dK_NM.transpose() @ (self.Gamma_inv) @ (self.K_NM)) - (
             self.K_MN @ ( self.Gamma_inv ) @ ( dGamma ) @ ( self.Gamma_inv ) @ ( self.K_NM ))
         
        dGamma_ = np.diag(np.diag(self.Gamma_sqrt_inv) * np.diag(dGamma) * np.diag(self.Gamma_sqrt_inv))
        
        dL1 = (
            np.trace(self.A_sqrt_inv @ dA @ self.A_sqrt_inv.T) -
		    np.trace(self.K_M_sqrt_inv @ dK_M @ self.K_M_sqrt_inv.T) +
            np.trace(dGamma_)
		) / 2
		
        dL2 = (
            -(1/2) * self.y_.T @ dGamma_ @ self.y_ +
            (self.A_sqrt_inv @ self.K_MN_ @ self.y_).T @
            (
                (1/2) * self.A_sqrt_inv @ dA @ self.A_sqrt_inv.T @ (self.A_sqrt_inv @ self.K_MN_ @ self.y_) -
                self.A_sqrt_inv @ dK_MN_ @ self.y_ +
                self.A_sqrt_inv @ self.K_MN_ @ dGamma_ @ self.y_
            )
        ) / sigma_sq
        
        #t1 =  -(1/2) * self.y_.T @ dGamma_ @ self.y_ 
        #t2 =   (self.A_sqrt_inv @ self.K_MN_ @ self.y_).T @
        #    (
        #        (1/2) * self.A_sqrt_inv @ dA @ self.A_sqrt_inv.T @ (self.A_sqrt_inv @ self.K_MN_ @ self.y_) -
        #        self.A_sqrt_inv @ dK_MN_ @ self.y_ +
        #        self.A_sqrt_inv @ self.K_MN_ @ dGamma_ @ self.y_
        #    )
        #dL2 = t1 + t2 *  sigma_sq
        
        return dL1 + dL2.squeeze()

    
    def derivate_c(self,hyp):
        """
        Returns the derivatives wrp c
        """
        c = hyp[0]

        dK_M = (1/c) * self.K_M
        #dK_N = (1/c) * self.K_N
        dK_N = np.eye(self.N)
        dK_NM = (1/c) * self.K_NM
        return dK_M, dK_N, dK_NM


    def derivate_b(self, k,pseudo_inputs):
        """
        Returns the derivatives wrp b_k
        """
        # k:th dimension of all data
        x_M = np.reshape(pseudo_inputs[:, k],  [self.M, 1])
        x_N = np.reshape(self.X_tr[:, k], [self.N, 1])
	    
        # Subtraction matrix
        M_M = -((x_M - x_M.T) ** 2) / 2
        dK_M = M_M * self.K_M # Elementwise multiplication
        	    
        # analogous
        M_NM = -((x_N - x_M.T) ** 2) / 2
        dK_NM = M_NM * self.K_NM
        
        # Possible error, but I think this one will be zero
        dK_N = np.zeros([self.N, self.N])    

        return dK_M, dK_N, dK_NM
        
        
    def derivate_kernel(self, m, k, hyp,pseudo_inputs):
        """ Returns the derivative wrp x_ik """
        # Get reshaped x:es
        x_M = np.reshape(pseudo_inputs[:, k],  [self.M, 1])
        x_N = np.reshape(self.X_tr[:, k], [self.N, 1])
        bk = hyp[1][k]
        
        # This one is zero
        dK_N = np.zeros([self.N, self.N])
        
        # K_M
        dK_M = np.zeros([self.M, self.M])
        
        for j in range(self.M):
            dK_M[m, j] += - bk * (x_M[m, 0] - x_M[j, 0]) * self.K_M[m, j]
        
        for i in range(self.M):
            dK_M[i, m] += - bk * (x_M[m, 0] - x_M[i, 0]) * self.K_M[m, i]
        
        # K_NM
        dK_NM = np.zeros([self.N, self.M])
        
        for j in range(self.N):
            dK_NM[j, m] += - bk * (x_M[m, 0] - x_N[j, 0]) * self.K_MN[m, j]   
        
        return dK_M, dK_N, dK_NM
        
        
    def derivate_sigma(self,sigma_sq):
        Gamma = self.Gamma
        Gamma_sqrt = self.Gamma_sqrt
        A_inv = self.A_inv
        K_MN = self.K_MN.dot( self.Gamma_sqrt_inv ) # Implicit underscore
        K_NM = np.transpose(K_MN)
        y = self.Gamma_sqrt_inv.dot(self.Y_tr)  

        dL1 = (1 / sigma_sq) * (np.trace(self.Gamma_inv) - np.trace(K_NM.dot(
                                                        A_inv ).dot( K_MN )))
        dL1 /= 2

        dL2 = norm(y) ** 2 + norm(K_NM.dot( A_inv ).dot( K_MN ).dot( y )) ** 2 
        dL2 -= 2 * y.transpose().dot( K_NM ).dot( A_inv ).dot( K_MN ).dot( y )
        dL2 *= -(1 /( sigma_sq ** 2) )
        dL2 /= 2
        return (dL1 + dL2)[0, 0] / 2


    def log_likelihood_packed(self,X):
        sigma_sq, hyp, pseudo_inputs = unpack_params(X,self.dim,self.M)
        return self.log_likelihood(sigma_sq,hyp,pseudo_inputs,recompute_stuffs = True)

    def log_likelihood(self,sigma_sq,hyp,pseudo_inputs,recompute_stuffs = False):
        # returns the log likelihood of the marginal for y
        if recompute_stuffs:
            #Modify object state for calculations
            old_sigma_sq = self.sigma_sq
            old_hyp = self.hyp
            old_pseudo_inputs = self.pseudo_inputs
            self.sigma_sq = sigma_sq
            self.hyp = hyp
            self.pseudo_inputs = pseudo_inputs
            self.set_kernel()
            self.do_precomputations()
            self.do_differential_precomputations()

        
        K_M = self.K_M
        K_MN = self.K_MN
        K_NM = self.K_NM
        Gamma = self.Gamma
        Gamma_sqrt = self.Gamma_sqrt
        A = self.A
        A_sqrt = self.A_sqrt
        A_sqrt_inv = self.A_sqrt_inv
        N = self.N
        M = self.M
        y_under = self.y_
        K_MN_under = self.K_MN_

        L1 = np.log(det(A)) - np.log(det(K_M)) + np.log(det(Gamma)) + (N - M) * np.log(sigma_sq)
        L1 /= 2
        L2 = (1/sigma_sq) * (norm(y_under) ** 2 - norm(A_sqrt_inv @ K_MN_under @ y_under) ** 2)
        L2 /= 2

        if recompute_stuffs:
            #reset original state
            self.sigma_sq = old_sigma_sq
            self.hyp = old_hyp
            self.pseudo_inputs = old_pseudo_inputs
            self.set_kernel()            
            self.do_precomputations()
            self.do_differential_precomputations()
        
        return L1 + L2 + (N/2 * np.log(2 * np.pi))
    
    def check_gradient(self):

        #print(check_grad(self.log_likelihood_packed
        #    ,self.derivate_log_likelihood_packed
        #    ,np.random.rand(get_packed_param_len(self.dim,self.M))))
        
        the_point = np.ones(get_packed_param_len(self.dim,self.M))
        self.sigma_sq, self.hyp, self.pseudo_inputs = unpack_params(the_point,self.dim,self.M)
        self.set_kernel()
        self.do_precomputations()
        self.do_differential_precomputations()
        dsigma_sq_num, dhyp_num, dpseudo_inputs_num = unpack_params(approx_fprime(the_point,self.log_likelihood_packed,epsilon = .00001),self.dim,self.M)
        dsigma_sq, dhy, dpseudo_inputs = unpack_params(self.derivate_log_likelihood_packed(the_point),self.dim,self.M)
        print("dsigma_sq - dsigma_sq_num = {0}, dsigma_sq = {1}, dpseudo_inputsnum = {2}".format(dsigma_sq - dsigma_sq_num,dsigma_sq ,dsigma_sq_num))
        print("dc - dc_num = {0}, dc = {1}, dc_num = {2}".format(dhy[0] - dhyp_num[0],dhy[0],dhyp_num[0]))
        print("db - db_num = {0}, db = {1}, db_num = {2}".format(dhy[1] - dhyp_num[1],dhy[1],dhyp_num[1]))
        print("dpseudo_inputs - dpseudo_inputs_num = {0},\ndpseudo_inputs =\n{1},\ndpseudo_inputs_num =\n{2}".format(dpseudo_inputs - dpseudo_inputs_num,dpseudo_inputs,dpseudo_inputs_num))

def unpack_params(X,dim,M):
    assert(len(X) == get_packed_param_len(dim,M))
    sigma_sq = X[0]
    hyp = [X[1],X[2:2+dim]]
    pseudo_inputs = X[2+dim:].reshape(M,dim)
    return sigma_sq, hyp, pseudo_inputs

def pack_params(sigma_sq, hyp, pseudo_inputs,M,dim):
    X = np.zeros(get_packed_param_len(dim,M))
    X[0] = sigma_sq
    X[1], X[2:2+dim] = hyp
    X[2+dim:] = pseudo_inputs.flatten()
    return X

def get_packed_param_len(dim, M):
    return 1 + 1 + dim + (M * dim)