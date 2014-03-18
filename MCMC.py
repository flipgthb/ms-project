#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import scipy.stats as st
import math
from collections import namedtuple

def row_norm(X):
    nX = np.sqrt((X*X).sum(axis=1)[:,np.newaxis])
    return nX

class MCMC(object):
    def __init__(self, system, num_steps, burn, measure_period,
                 d_omega=1., d_eps=.1):
        self.num_steps = num_steps                 # number of steps
        self.burn = burn                           # burned steps
        self.measure_period = measure_period       # measure period
        self.delta_omega = d_omega                 # propose scale
        self.delta_epsilon = d_eps                 # adjacency increment scale

        self._data = None

        # system is an instance of society
        self.system = system
        self.explain = ("beta", "delta", "R_max", "R_mean", "R_var")

    @property
    def data(self):
        data_means = np.array(self._data).mean(axis=0)
        w = self.system.w.copy()
        s = self.system.social_network.copy()
        z = self.system.zeitgeist.copy()
        final_data = {"statistics":data_means,
                      "state": w,
                      "social_network": s,
                      "zeitgeist": z,
                      "explain": self.explain}
        return final_data

    @data.setter
    def data(self, more):
        try:
            self._data = np.vstack([self._data, more])
        except:
            self._data = more

    def propose(self, i):
        w0 = self.system.w[i].copy()
        w = np.random.multivariate_normal(
            np.zeros_like(w0),
            self.delta_omega*np.identity(*w0.shape)
        )
        w /= np.linalg.norm(w)
        w += w0
        w /= np.linalg.norm(w)
        self.system.w[i] = w.copy()
        return w0.copy()

    def step(self):
        # picking pair of agents: first pick i uniformly over N
        i = np.random.choice(self.system.N)

        # than, get the weight pij gives to each other agent.
        # The weights are given by the system social network
        pij = self.system.listening_probability[i]

        # and finally pick j from N/i with probability pij
        j = np.random.choice(self.system.N, p=pij)

        # update the coupling vector
        E0 = self.system.potential(i, j)
        w0 = self.propose(i)
        E = self.system.potential(i, j)
        bdE = self.system.beta*(E - E0)
        acc = min(1, math.exp(-bdE))
        rej = np.random.rand()
        if rej >= acc:
            self.system.w[i] = w0.copy()

        # update the social network
        x = self.system.social_network[i]
        sign = np.sign(self.system.agreement(i,j))
        x[j] += sign*self.delta_epsilon*(1+1/self.system.N)
        x -= sign*self.delta_epsilon*(1/self.system.N)
        # x[x<0] = 0
        x[i] = 0
        x *= self.system.N / x.sum()

        # update the zeitgeist
        z_new = self.system.social_network.dot(self.system.w)
        z_new /= row_norm(z_new)
        self.system.zeitgeist = z_new.copy()

    def sample(self):
        for k in xrange(self.num_steps):
            self.step()
            if k >= self.burn and k%self.measure_period == 0:
                self.measure()

    def measure(self):
        R_max = self.system.reputation.max()
        R_mean = self.system.reputation.mean()
        R_var = self.system.reputation.var()
        beta = self.system.beta
        delta = self.system.delta
        self.data = np.array([beta, delta, R_max, R_mean, R_var])


if __name__ == "__main__":
    from Society import Society

    S = Society(3, 5, .2, 5)
    mc = MCMC(S, 100, 0, 1)
    mc.sample()
    x = mc.data
    for k, v in x.items():
        print(k,':\n',v, end='\n\n')
