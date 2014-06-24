#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import math
import networkx as nx

from utility import row_norm,prob_row_norm,create_network,create_initial_state

class Society(object):
    def __init__(self, topology, D=5, delta=1.0, beta=0.0, gamma=1.0,
                 initial_state="disordered", zeitgeist=None):
        self.topology = topology
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.D = D

        # z0 = np.arange(self.D)
        z0 = np.zeros(D)
        z0[0] = 1.0
        z0[-1] = -1
        self.zeitgeist = zeitgeist or z0
        self.zeitgeist /= np.linalg.norm(self.zeitgeist)

        self.social_network = create_network(self.topology)

        self.N = self.social_network.shape[0]
        self.W = create_initial_state(self.N, self.D, initial_state)

        self.reputation_score = self.social_network.copy()
        self.reputation_score *= self.N/prob_row_norm(self.reputation_score)

        self.activity_record = np.zeros_like(self.social_network)

    @property
    def opinion(self):
        h = self.zeitgeist.dot(self.W.T)
        return h

    @property
    def reputation(self):
        r = self.reputation_score
        R = r - r.min(axis=1)[:,None]
        # r *= self.N/prob_row_norm(r)
        # R = np.exp(r)
        R -= np.diag(np.diag(R))
        # R /= prob_row_norm(R)
        return R

    @property
    def activity(self):
        a = self.activity_record.copy()
        total = a.sum()
        if total == 0:
            return a
        return a/total

    def neighbor_weights(self, i):
        p = self.reputation[i].copy()
        # r = self.reputation_score[i].copy()
        # p = r - r.min()
        # r *= self.N/r.sum()
        # p = np.exp(r)
        p /= p.sum()
        return p

    def potential(self, i, j):
        K1, K2 = (1+self.delta)/2, (1-self.delta)/2
        hi, hj = self.opinion[[i,j]]
        x = hi*np.sign(hj)
        if x < -self.gamma:
            # V_max = 1000*self.N*(self.N-1)/2   # should be higher
            V_max = np.infty
            return V_max
        x = hi*hj
        V = -K1*x + K2*abs(x)
        return V

if __name__ == "__main__":
    top = dict(type="complete", parameters=dict(n=10))
    s = Society(top)
    print(
        s.energy(),
        s.opinion,
        s.reputation,
        s.activity,
        sep='\n\n'
    )
