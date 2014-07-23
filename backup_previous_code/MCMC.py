#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import math
import pandas as pd
from utility import prob_row_norm, row_norm

class MCMC(object):
    def __init__(self, system, max_sweeps=10, therm_time=0,
                 autocor_time=1, epsilon=0.5, dW=1.0):
        self.system = system
        self.max_sweeps = max_sweeps
        self.therm_time = therm_time
        self.autocor_time = autocor_time
        self.epsilon = epsilon
        self.dW = dW

        h = self.system.opinion
        m = h.mean()
        r =  np.abs(h).mean()
        R = self.system.reputation_score.copy()
        x = np.zeros_like(R)
        x[R<1] = 1
        x -= np.diag(np.diag(x))
        n_op = x.mean(axis=1).mean()
        a = R.mean()

        self.trace = {
            "m":np.array([m]),
            "r":np.array([r]),
            "n_op":np.array([n_op]),
            "R":np.array([a])
        }

    @property
    def data(self):
        df = pd.DataFrame(self.trace)
        sn = self.system.social_network
        rep = self.system.reputation_score
        act = self.system.activity_record
        w = self.system.W
        return df, sn, rep, act, w

    def measure(self):
        h = self.system.opinion
        m = h.mean()
        r =  np.abs(h).mean()
        R = self.system.reputation_score.copy()
        x = np.zeros_like(R)
        x[R<1] = 1
        x -= np.diag(np.diag(x))
        n_op = x.mean(axis=1).mean()
        a = R.mean()

        self.trace["m"] = np.hstack([self.trace["m"], m])
        self.trace["r"] = np.hstack([self.trace["r"], r])
        self.trace["n_op"] = np.hstack([self.trace["n_op"], n_op])
        self.trace["R"] = np.hstack([self.trace["R"], a])

    def propose(self, i):
        w0 = self.system.W[i].copy()
        S = self.dW*np.identity(*w0.shape)
        w = np.random.multivariate_normal(w0, S)
        w /= np.linalg.norm(w)
        self.system.W[i] = w.copy()
        return w0.copy()

    def sample(self):
        n = 0
        t = 0
        for _ in xrange(self.max_sweeps*self.system.N):
            self.step()
            n += 1
            if n == self.system.N:
                n = 0
                t += 1
                if t >= self.therm_time and t%self.autocor_time == 0:
                    self.measure()
        return self.data

    def step(self):
        i = np.random.choice(self.system.N)
        pij = self.system.neighbor_weights(i)
        j = np.random.choice(self.system.N, p=pij)
        hi, hj = self.system.opinion[[i,j]]
        x0 = hi*hj
        self.system.activity_record[i,j] += 1
        # self.system.reputation_score[i,j] += np.sign(x0)*self.epsilon
        self.system.reputation_score[i,j] += x0*self.epsilon

        if x0 < -self.system.tau:
            return

        V0 = self.system.potential(i,j)
        w0 = self.propose(i)
        V = self.system.potential(i,j)
        bdV = self.system.beta*(V - V0)
        acc = min(0, -bdV)
        rej = math.log(np.random.rand())
        if rej > acc:
            self.system.W[i] = w0.copy()


if __name__ == "__main__":
    from Society import Society
    top = dict(type="complete", parameters=dict(n=10))
    s = Society(top)
    mc = MCMC(s)
    print(mc.data)
    data = mc.sample()
    print(data)
