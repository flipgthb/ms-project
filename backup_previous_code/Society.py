#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import math
import networkx as nx

from utility import row_norm,prob_row_norm,create_network,create_initial_state

class Society(object):
    def __init__(self, topology, zeitgeist=None,
                 D=5, delta=1.0, beta=0.0, tau=1.0,
                 initial_state="disordered"):
        self.topology = topology
        self.delta = delta
        self.beta = beta
        self.tau = tau
        self.D = D

        if not zeitgeist:
            print("zeitgeist is None")
            zeitgeist = np.zeros(D)
            zeitgeist[0] = 1
            zeitgeist[-1] = -1
        self.zeitgeist = zeitgeist
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
        r = self.reputation_score.copy()
        R = r - r.min(axis=1)[:,None]
        R -= np.diag(np.diag(R))
        # R /= R.max(axis=1)[:,None]
        R /= prob_row_norm(R)
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
        # p = self.social_network[i].copy()
        # p /= p.sum()
        return p

    def potential(self, i, j):
        K1, K2 = (1+self.delta)/2, (1-self.delta)/2
        hi, hj = self.opinion[[i,j]]
        # x = hi*np.sign(hj)
        # if x < -self.tau:
        #     V_max = np.infty
        #     return V_max
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
