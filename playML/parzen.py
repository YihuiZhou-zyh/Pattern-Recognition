import numpy as np
import matplotlib.pyplot as plt


def parzen_estimation(mu, sigma, mode='gauss'):
    """
    Implementation of a parzen-window estimation
    Keyword arguments:
        x: A "nxd"-dimentional numpy array, which each sample is
                  stored in a separate row (=training example)
        mu: point x for density estimation, "dx1"-dimensional numpy array
        sigma: window width
    Return the density estimate p(x)
    """

    def log_mean_exp(a):
        max_ = a.max(axis=1)
        return max_ + np.log(np.exp(a - np.expand_dims(max_, axis=0)).mean(1))

    def gaussian_window(x, mu, sigma):
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / sigma
        b = np.sum(- 0.5 * (a ** 2), axis=-1)
        E = log_mean_exp(b)
        Z = mu.shape[1] * np.log(sigma * np.sqrt(np.pi * 2))
        return np.exp(E - Z)

    def hypercube_kernel(x, mu, h):
        n, d = mu.shape
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / h
        b = np.all(np.less(np.abs(a), 1 / 2), axis=-1)
        kn = np.sum(b.astype(int), axis=-1)
        return kn / (n * h ** d)

    if mode is 'gauss':
        return lambda x: gaussian_window(x, mu, sigma)
    elif mode is 'hypercube':
        return lambda x: hypercube_kernel(x, mu, h=sigma)
