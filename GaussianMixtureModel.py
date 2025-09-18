
import numpy as np

from scipy.special import psi

from sklearn.cluster import kmeans_plusplus

from scipy.special import softmax

class GaussianMixtureModel:

    def __init__(self, X : np.ndarray, M : int) -> None:
        
        self.X = X

        self.N, self.D = self.X.shape

        self.M = M

        self.alpha_0 = 1/self.M

        self.tau_0 = 1

        self.mu_0 = np.zeros(shape = self.D)

        self.nu_0 = self.D

        self.Sigma_0 = np.identity(n = self.D)

        self.Lambda_0 = np.linalg.inv(self.Sigma_0)

        self.E_log_pi = np.zeros(shape = self.M)

        self.E_log_det_Lambda = np.zeros(shape = self.M)

        self.gamma = np.zeros(shape = (self.N, self.M))

        self.N_barra = np.zeros(shape = self.M)

        self.X_barra = np.zeros(shape = (self.M, self.D))

        self.S_barra = np.zeros(shape = (self.M, self.D, self.D))

        self.alpha = np.zeros(shape = self.M)

        self.tau = np.zeros(shape = self.M)

        self.mu = np.zeros(shape = (self.M, self.D))

        self.nu = np.zeros(shape = self.M)

        self.Phi = np.zeros(shape = (self.M, self.D, self.D))

        self.Psi = np.zeros(shape = (self.M, self.D, self.D))

        self.pi = np.zeros(shape = self.M)

        self.Z = np.zeros(shape = self.N)

        self.Sigma = np.zeros(shape = (self.M, self.D, self.D))

        self.Lambda = np.zeros(shape = (self.M, self.D, self.D))

        self.delta = 0

    def initialize_parameters(self) -> None:

        self.alpha = np.repeat(self.alpha_0, repeats = self.M)

        self.tau = np.repeat(self.tau_0, repeats = self.M)

        self.mu = kmeans_plusplus(X = self.X, n_clusters = self.M)[0]

        self.nu = np.repeat(self.nu_0, repeats = self.M)

        self.Phi = np.tile(self.Sigma_0, reps = (self.M, 1, 1))

        self.Psi = np.linalg.inv(self.Phi)

    def update_E_log_pi(self) -> None:

        self.E_log_pi = psi(self.alpha)

        self.E_log_pi -= psi(self.alpha.sum())

    def update_E_log_det_Lambda(self) -> None:

        self.E_log_det_Lambda = np.add.outer(self.nu, 1 - np.arange(start = 1, stop = self.D))/2

        self.E_log_det_Lambda = psi(self.E_log_det_Lambda).sum(axis = 1)

        self.E_log_det_Lambda += self.D*np.log(2) + np.log(np.linalg.det(self.Psi))

    def update_gamma(self) -> None:

        self.gamma = np.expand_dims(self.X, axis = 1) - np.expand_dims(self.mu, axis = 0)
        
        self.gamma = -self.nu*np.einsum('nmd, mdd, nmd -> nm', self.gamma, self.Psi, self.gamma)/2

        self.gamma += self.E_log_pi + (self.E_log_det_Lambda - self.D/self.tau)/2

        self.gamma = softmax(self.gamma, axis = 1)

    def update_N_barra(self) -> None:

        self.N_barra = self.gamma.sum(axis = 0)

    def update_X_barra(self) -> None:

        self.X_barra = self.gamma.T @ self.X

        self.X_barra /= np.expand_dims(self.N_barra, axis = 1)

    def update_S_barra(self) -> None:

        self.S_barra = np.expand_dims(self.X, axis = 0) - np.expand_dims(self.X_barra, axis = 1)

        self.S_barra = np.einsum('mn, mnD, mnd -> mDd', self.gamma.T, self.S_barra, self.S_barra)

        self.S_barra /= np.expand_dims(self.N_barra, axis = (1, 2))

    def update_alpha(self) -> None:

        self.alpha = self.alpha_0 + self.N_barra

    def update_tau(self) -> None:

        self.tau = self.tau_0 + self.N_barra

    def update_mu(self) -> None:

        self.mu = np.expand_dims(self.N_barra, axis = 1)*self.X_barra

        self.mu += self.tau_0*self.mu_0

        self.mu /= np.expand_dims(self.tau, axis = 1)

    def update_nu(self) -> None:

        self.nu = self.nu_0 + self.N_barra

    def update_Phi(self) -> None:

        self.Phi = self.X_barra - np.expand_dims(self.mu_0, axis = 0)

        self.Phi = np.einsum('mD, md -> mDd', self.Phi, self.Phi)

        self.Phi *= np.expand_dims(self.tau_0*self.N_barra/self.tau, axis = (1, 2))

        self.Phi += np.expand_dims(self.N_barra, axis = (1, 2))*self.S_barra + self.Lambda_0

    def update_Psi(self) -> None:

        self.Psi = np.linalg.inv(self.Phi)

    def update_parameters(self) -> None:

        self.update_E_log_pi()

        self.update_E_log_det_Lambda()

        self.update_gamma()

        self.update_N_barra()

        self.update_X_barra()
        
        self.update_S_barra()

        self.update_alpha()

        self.update_tau()

        self.update_mu()

        self.update_nu()

        self.update_Phi()

        self.update_Psi()

    def estimate_pi(self) -> None:

        self.pi = self.alpha/self.alpha.sum()

    def estimate_Z(self) -> None:

        self.Z = np.argmax(self.gamma, axis = 1)

    def estimate_Sigma(self) -> None:

        self.Sigma = self.Phi/(np.expand_dims(self.nu, axis = (1, 2)) + self.D + 1)

    def estimate_Lambda(self) -> None:

        self.Lambda = np.expand_dims(self.nu, axis = (1, 2))*self.Psi

    def estimate_parameters(self) -> None:

        self.estimate_pi()

        self.estimate_Z()

        self.estimate_Sigma()

        self.estimate_Lambda()

    def iterate_steps(self, MAX : int = 1000, TOL : float = 1e-6) -> None:

        self.initialize_parameters()

        for i in range(MAX):

            self.delta = self.mu.copy()

            self.update_parameters()

            self.delta -= self.mu

            self.delta = np.linalg.norm(self.delta, axis = 1).max()

            if self.delta < TOL:

                break

        self.estimate_parameters()
