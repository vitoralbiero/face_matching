import numpy as np
from scipy import ndimage


class TanTriggsPreprocessing():
    def __init__(self, alpha=0.1, tau=10.0, gamma=0.2, sigma0=2.0, sigma1=3.0):
        self._alpha = float(alpha)
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._sigma0 = float(sigma0)
        self._sigma1 = float(sigma1)

    def compute(self, X):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        X = np.array(X, dtype=np.float32)
        X = np.power(X, self._gamma)
        X = np.asarray(ndimage.gaussian_filter(X, self._sigma1) - ndimage.gaussian_filter(X, self._sigma0))
        X = X / np.power(np.mean(np.power(np.abs(X), self._alpha)), 1.0 / self._alpha)
        X = X / np.power(np.mean(np.power(np.minimum(np.abs(X), self._tau), self._alpha)), 1.0 / self._alpha)
        X = self._tau * np.tanh(X / self._tau)
        return X
