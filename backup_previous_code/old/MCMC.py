#/usr/bin/env python
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
        self.explain = (
            "beta",
            "delta",
            "gamma",
            "m",
            "stag_m",
            "M",
            "r",
            "n",
            "n_pos",
            "n_neg",
            "R_max",
            "R_mean",
            "V_max",
            "V_mean"
        )

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
        a = self.system.activity.copy()
        final_data = {"statistics":data_means,
                      "state": w,
                      "social_network": s,
                      "zeitgeist": z,
                      "cumulative": data_cumulative,
                      "initial_social_network": s0,
                      "activity": a,
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
            # np.zeros_like(w0),
            w0,
            self.delta_omega*np.identity(*w0.shape)
        )
        w /= np.linalg.norm(w)
        # w += w0
        # w /= np.linalg.norm(w)
        self.system.w[i] = w.copy()
        return w0.copy()

    def step(self):
        # pick an agent i uniformly in N
        i = np.random.choice(self.system.N)

        # pick an neighbor j of i based on Rij
        pij = self.system.neighbor_weights(i)
        j = np.random.choice(self.system.N, p=pij)

        # update the activity between (i,j)
        self.system.activity[i,j] += 1

        # update the coupling vector
        hi, hj = self.system.field[[i,j]]
        x0 = hi*np.sign(hj)
        ni = self.system.social_network[i]
        if x0 < -self.system.gamma:
            ni[j] -= self.delta_epsilon
            ni *= self.system.N/ni.sum()
            return

        # coupling vector proposition
        E0 = self.system.potential(i,j)
        # E0 = self.system.energy()
        # E0 = self.system.local_energy(i)
        w0 = self.propose(i)
        E = self.system.potential(i,j)
        # E = self.system.energy()
        # E = self.system.local_energy(i)
        bdE = self.system.beta*(E - E0)
        acc = min(0, -bdE)
        rej = math.log(np.random.rand())
        if rej >= acc:
            self.system.w[i] = w0.copy()

        # update the affinity
        ni[j] += x0*self.delta_epsilon
        ni *= self.system.N/ni.sum()

    def sample(self):
        for k in xrange(self.num_steps):
            self.step()
            if k >= self.burn and k%self.measure_period == 0:
                self.measure()

    def measure(self):
        # opinion order parameters
        m = self.system.field.mean()
        r = np.abs(self.system.field).mean()
        h = self.system.field
        n_pos = h[h>=0].shape[0]
        n_neg = h[h<=0].shape[0]
        n_null = h[h==0].shape[0]
        m_pos = h[h>0].sum()/(n_pos+1e-5)
        m_neg = h[h<0].sum()/(n_neg+1e-5)
        stag_m = (m_pos*n_pos - m_neg*n_neg)/self.system.N
        M = (m_pos*n_pos + m_neg*n_neg)/self.system.N
        n = (n_pos - n_neg)/self.system.N

        # structure order parameters
        V = self.system.visibility
        V_max = V.max()
        V_mean = V.mean()
        R = self.system.reputation
        R_max = R.max()
        R_mean = R.mean()
        beta = self.system.beta
        delta = self.system.delta
        gamma = self.system.gamma

        self.data = np.hstack([
            beta,
            delta,
            gamma,
            m,
            stag_m,
            M,
            r,
            n,
            n_pos,
            n_neg,
            R_max,
            R_mean,
            V_max,
            V_mean
        ])


if __name__ == "__main__":
    from Society import Society

    S = Society(64, 20, 5, .2, 5, .5)
    mc = MCMC(S, 100, 0, 1, 1)
    mc.sample()
    x = mc.data
    for k, v in x.items():
        print(k,':\n',v, end='\n\n')
