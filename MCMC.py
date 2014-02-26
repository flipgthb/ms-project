#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import scipy.stats as st
import math
from collections import namedtuple


class MCMC(object):
    def __init__(self, system, n, b, m, ds=1., df=1.):
        self.n = n       # number of steps
        self.b = b       # burned steps
        self.m = m       # measure period
        self.ds = ds     # propose scale
        self.df = df     # adjacency increment scale
        # samples will be drawn from scipy.stats.norm
        self._data = None
        # system is an instance of society
        self.system = system
        self.explain = ("beta", "rho", "gamma", "eps", "fi",
                        "sig", "h1", "h2", "v1", "v2")

    @property
    def data(self):
        data_means = np.array(self._data).mean(axis=0)
        s = self.system.state.copy()
        a = self.system.adjacency.copy()
        final_data = {"statistics":data_means,
                      "state": s,
                      "adjacency": a,
                      "explain": self.explain}
        return final_data

    @data.setter
    def data(self, more):
        try:
            self._data = np.vstack([self._data, more])
        except:
            self._data = more

    def propose(self, i):
        w0 = self.system.state[i].copy()
        w = np.random.multivariate_normal(np.zeros_like(w0),
                                          np.identity(*w0.shape))
        w /= np.linalg.norm(w)
        w += w0
        w /= np.linalg.norm(w)
        self.system.state[i] = w.copy()
        return w0.copy()

    def step(self):
        # picking pair of agents: first pick i uniformly over N
        i = np.random.choice(self.system.N)

        # than, get the weight pij gives to each other agent.
        # The weights are given by the system topology
        pij = self.system.topology[i]

        # and finally pick j from N/i with probability pij
        j = np.random.choice(self.system.N, p=pij)

        # updating adjacency. Basically, add df to adjacency[i,j]
        # if the they agree, zero otherwise. Note that the adition
        # adjacency is kept symmetric.
        fi = (1 + np.sign(self.system.agreement(i, j)))/2
        self.system.adjacency[i, j] += fi * self.df
        self.system.adjacency[j, i] += fi * self.df

        # proposing a new state with Metropolis-Hastings
        E0 = self.system.energy(i, j)
        w0 = self.propose(i)
        E = self.system.energy(i, j)
        bdE = self.system.beta*(E - E0)
        acc = min(1, math.exp(-bdE))
        rej = np.random.rand()
        if rej >= acc:
            self.system.state[i] = w0.copy()

    def sample(self):
        for k in xrange(self.n):
            self.step()
            if k >= self.b and k%self.m == 0:
                self.measure()

    def measure(self):
        h1, h2 = self.system.field.mean(axis=0)
        v1, v2 = self.system.field.var(axis=0)
        beta = self.system.beta
        rho = self.system.rho
        gamma = self.system.gamma
        eps = self.system.eps
        fi = self.df
        sig = self.ds
        self.data = np.array([beta, rho, gamma, eps, fi, sig, h1, h2, v1, v2])


if __name__ == "__main__":
    from Society import Society

    w = np.ones((64, 5))
    rho = .8
    eps = .2
    beta = 1.
    gamma = np.pi / 3
    S = Society(w, gamma, rho, eps, beta)
    mc = MCMC(S, 10, 0, 1)
    mc.sample()
    print(mc.data)
