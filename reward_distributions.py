# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

# Implements distributions to be used to encode and predict posterior reward.
import numpy as np
import random


class NormalGamma:
    """The Normal-Gamma distribution, prior for a Normal distribution."""

    def __init__(self, mu=0, lamb=0, alpha=0.5, beta=0.1):
        self._mu = mu
        self._lamb = lamb
        self._alpha = alpha
        self._beta = beta

    def update(self, y):
        # Update the posteriors directly
        new_lamb = self._lamb + 1
        self._alpha = self._alpha + 0.5
        self._beta = self._beta + self._lamb*(y - self._mu)**2/(2*new_lamb)
        self._mu = (self._lamb * self._mu + y) / new_lamb
        self._lamb = new_lamb

    def sample_posterior(self):
        # Sample gamma with parameters alpha and beta, using k = alpha, theta = 1/beta
        tau = random.gammavariate(self._alpha, 1/self._beta)
        # Sample gaussian with mean mu and std 1/sqrt(lambda*tau)
        mean = random.gauss(self._mu, 1/np.sqrt(self._lamb * tau))
        # Sample gaussian with mean mean and std 1/sqrt(tau)
        return random.gauss(mean, 1/np.sqrt(tau))

    def posterior_mean(self):
        return self._mu


def test_normal_gamma_params():
    y = np.array([1, 1.5, 3])
    dist = NormalGamma()
    for i in range(3):
        dist.update(y[i])

    # mu = 1.375
    print("mu = {}".format(dist._mu))
    # lambda = 4
    print("lambda = {}".format(dist._lamb))
    # alpha = 2.5
    print("alpha = {}".format(dist._alpha))
    # beta = 3.34375
    print("beta = {}".format(dist._beta))


def test_normal_gamma_convergence():
    dist = NormalGamma()

    # Test convergence to N(0.55, 0.3)
    for i in range(500):
        dist.update(np.random.normal(0.55, 0.3))

    pred = np.zeros(500)
    for i in range(500):
        pred[i] = dist.sample_posterior()

    print("Empirical mean = {}".format(np.mean(pred)))
    print("Empirical std = {}".format(np.std(pred)))
