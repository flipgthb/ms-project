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
        self.trace = {
            "m":np.array([self.system.opinion.sum()]),
            "r":np.array([np.abs(self.system.opinion).sum()]),
            "R": np.array([np.max(self.system.reputation.sum(axis=0))]),
            "<R>":np.array([np.mean(self.system.reputation.sum(axis=0))]),
            "A":np.array([np.max(self.system.activity_record.sum(axis=0))]),
            "<A>": np.array([np.mean(self.system.activity_record.sum(axis=0))]),
            "n_pos":np.array([h[h>0].shape[0]/self.system.N]),
            "n_neg":np.array([h[h<0].shape[0]/self.system.N]),
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
        H_new = self.system.opinion.mean()
        abs_H_new =  np.abs(self.system.opinion).mean()
        max_rep_new = np.max(self.system.reputation_score.mean(axis=0))
        mean_rep_new = np.mean(self.system.reputation_score.mean(axis=0))
        max_act_new = np.max(self.system.activity.mean(axis=0))
        mean_act_new = np.mean(self.system.activity.mean(axis=0))
        h = self.system.opinion
        n_pos = h[h>0].shape[0]/self.system.N
        n_neg = h[h<0].shape[0]/self.system.N

        self.trace["m"] = np.hstack([self.trace["m"], H_new])
        self.trace["r"] = np.hstack([self.trace["r"], abs_H_new])
        self.trace["R"] = np.hstack([self.trace["R"], max_rep_new])
        self.trace["<R>"] = np.hstack([self.trace["<R>"], mean_rep_new])
        self.trace["A"] = np.hstack([self.trace["A"], max_act_new])
        self.trace["<A>"] = np.hstack([self.trace["<A>"], mean_act_new])
        self.trace["n_pos"] = np.hstack([self.trace["n_pos"], n_pos])
        self.trace["n_neg"] = np.hstack([self.trace["n_neg"], n_neg])

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
        x0 = hi*hj  # hi*np.sign(hj)
        self.system.activity_record[i,j] += 1
        self.system.reputation_score[i,j] += x0*self.epsilon

        if x0 < -self.system.gamma:
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
