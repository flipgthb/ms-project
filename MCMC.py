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

def Heavyside(x):
    return 1 if x>=0 else 0

def weighted_choice(weights):
    rnd = np.random.rand() * weights.sum()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

class MCMC(object):
    def __init__(self, system, num_steps, burn, measure_period,
                 d_omega=1., d_eps=.2, drift=1.0):
        self.num_steps = num_steps                 # number of steps
        self.burn = burn                           # burned steps
        self.measure_period = measure_period       # measure period
        self.delta_omega = d_omega                 # propose scale
        self.delta_epsilon = d_eps                 # adjacency increment scale
        self.drift = drift                         # zeitgeist drift factor

        self._data = None

        # system is an instance of society
        self.system = system
        self.explain = ("beta","delta","gamma","m","v","m2",
                        "stag_m","M","r","n","R_max","R_mean","R_var")

    @property
    def data(self):
        data_means = np.array(self._data).mean(axis=0)
        data_cumulative = np.array(self._data).cumsum(axis=0)
        n = np.arange(1.,data_cumulative.shape[0]+1)[:, np.newaxis]
        data_cumulative /= n
        w = self.system.w.copy()
        s = self.system.social_network.copy()
        s0 = self.system.initial_social_network.copy()
        z = self.system.zeitgeist.copy()
        final_data = {"statistics":data_means,
                      "state": w,
                      "social_network": s,
                      "zeitgeist": z,
                      "cumulative": data_cumulative,
                      "initial_social_network": s0,
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
        # pick an agent i distributed with prop exp(-Ri)
        # pi = self.system.agent_weights()
        # i = np.random.choice(self.system.N, p=pi)
        i = np.random.choice(self.system.N)
        # pick an neighbor j of i based on Rij
        pij = self.system.neighbor_weights(i)
        j = np.random.choice(self.system.N, p=pij)

        # update the coupling vector
        x0 = self.system.agreement(i,j)
        # hj = self.system.field[j]
        ni = self.system.social_network[i]
        if x0 < -self.system.gamma:
            self.system.social_network[i,j] -= 1.0  # works, in some way
            # self.system.social_network[i,j] -= 10.0
            # ni[j] /= 2
            ni *= self.system.N/ni.sum()
            return

        E0 = self.system.potential(i,j)
        w0 = self.propose(i)
        E = self.system.potential(i,j)

        bdE = self.system.beta*(E - E0)
        acc = min(1, math.exp(-bdE))
        rej = np.random.rand()
        if rej >= acc:
            self.system.w[i] = w0.copy()

        # update the affinity
        self.system.social_network[i,j] += x0 # works, with the previous
        # self.system.social_network[i,j] += np.sign(x0)*10
        # ni[j] *= 2**np.sign(x0)
        ni *= self.system.N/ni.sum()


    def sample(self):
        for k in xrange(self.num_steps):
            self.step()
            if k >= self.burn and k%self.measure_period == 0:
                self.measure()

    def measure(self):
        m = self.system.field.mean()
        r = np.abs(self.system.field).mean()
        m2 = (self.system.field*self.system.field).mean()
        v = self.system.field.var()
        h = self.system.field
        n_pos = h[h>0].shape[0]
        n_neg = h[h<0].shape[0]
        m_pos = h[h>0].sum()/(n_pos+1)
        m_neg = h[h<0].sum()/(n_neg+1)
        stag_m = (m_pos - m_neg)/2#self.system.N
        M = (m_pos + m_neg)/self.system.N
        n = (n_pos - n_neg)/self.system.N

        R = self.system.reputation
        R_max = R.max()
        R_mean = R.mean()
        R_var = R.var()
        beta = self.system.beta
        delta = self.system.delta
        gamma = self.system.gamma
        self.data = np.hstack([beta, delta, gamma, m, v, m2, stag_m, M,
                               r,n,R_max, R_mean, R_var])


if __name__ == "__main__":
    from Society import Society

    S = Society(64, 20, 5, .2, 5, .5)
    mc = MCMC(S, 100, 0, 1, 1)
    mc.sample()
    x = mc.data
    for k, v in x.items():
        print(k,':\n',v, end='\n\n')
