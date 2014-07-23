#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import math
import networkx as nx
from scipy.special import erf

def row_norm(X):
    nX = np.sqrt((X*X).sum(axis=1)[:, np.newaxis])
    return nX

def prob_row_norm(X):
    nX = X.sum(axis=1)[:, np.newaxis]
    return nX

class Society(object):

    def __init__(self, N, k, D, delta, beta, gamma,
                 rewiring_prob=1/2, net_type="complete"):

        self.N = N  # number of agents
        self.D = D  # agent complexity, cognitive vector dimension
        self.k = k # number of neighbors
        self.rewiring_prob = rewiring_prob # watts-strogatz rewiring probability
        self.delta = delta  # cognitive style, learning style
        self.beta = beta  # peer pressure
        self.gamma = gamma  # bullshit threshold
        self.net_type = net_type

        self.w = np.random.randn(self.N, self.D)
        # self.w = np.ones((self.N, self.D))
        # self.w[self.N/2:,:] *= -1
        self.w /= row_norm(self.w)

        G = self.generate_network()
        self.social_graph = G
        self.social_network = np.array(nx.adjacency_matrix(G))
        # self.social_network = np.zeros((self.N,self.N))
        self.initial_social_network = self.social_network.copy()
        # self.activity = np.zeros_like(self.social_network)
        self.activity = self.social_network.copy()

        # self.zeitgeist = np.random.randn(self.D)
        # self.zeitgeist = np.ones(self.D)
        # self.zeitgeist[0] = 1-self.D
        self.zeitgeist = np.zeros(self.D)
        self.zeitgeist[0] = 1.
        self.zeitgeist /= np.linalg.norm(self.zeitgeist)

    def generate_network(self):
        net_type_dict = dict(
            complete=nx.complete_graph(self.N),
            barabasi_albert=nx.barabasi_albert_graph(self.N,self.k),
            watts_strogatz=nx.watts_strogatz_graph(self.N,self.k,
                                                   self.rewiring_prob),
        )
        return net_type_dict[self.net_type]

    @property
    def field(self):
        h = self.zeitgeist.dot(self.w.T)
        return h

    def agreement(self, i, j):
        h = self.field
        hi, hj = h[[i,j]]
        return hi*hj

    def potential(self, i, j):
        a =  (1+self.delta)/2
        b =  (1-self.delta)/2
        z  = self.agreement(i,j)
        if z < -self.gamma:
            return np.infty
        Vij = -a*z + b*abs(z)
        return Vij

    def energy(self):
        a =  (1+self.delta)/2
        b =  (1-self.delta)/2
        h = self.field
        hT = self.field[:,None]
        A = self.initial_social_network
        z = A*h*hT
        abs_z = np.abs(z)
        abs_zg = np.abs(z+self.gamma)
        x = z.sum()/2
        y = abs_z.sum()/2
        E = -a*x + b*y
        return E

    def local_energy(self, i):
        a =  (1+self.delta)/2
        b =  (1-self.delta)/2
        h = self.field
        ai = self.initial_social_network[i]
        z = ai*h*h[i]
        if (z < -self.gamma).any():
            return np.infty
        abs_z = np.abs(z)
        x = z.sum()/2
        y = abs_z.sum()/2
        E = -a*x + b*y
        return E

    @property
    def reputation(self):
        Rij = np.exp(self.social_network)
        Rij -= np.diag(np.diag(Rij))
        # Rij = self.social_network
        # Ri_min, Ri_max = Rij.min(axis=1)[:,None], Rij.max(axis=1)[:,None]
        # Rij = (Rij - Ri_min)/(Ri_max - Ri_min)
        Rij /= Rij.sum(axis=1)[:,None]
        R = (Rij).sum(axis=0)
        return R

    @property
    def visibility(self):
        Vij = self.activity
        Vij /= Vij.sum(axis=1)[:,None]
        V = (Vij).sum(axis=0)
        return V

    def neighbor_weights(self, i):
        y = self.social_network[i].copy()
        x = np.exp(y)
        x[i] = 0
        P = x/x.sum()
        return P


if __name__ == "__main__":
    S = Society(64, 20, 5,.2,5)
    test = """
    S.w =
    {}

    S.zeitgeist =
    {}

    S.social_network =
    {}

    S.listening_probability(0) =
    {}

    S.field =
    {}

    S.agreement(0,2,0) = {}

    S.potential(0,2,0) = {}
    """.format(S.w, S.zeitgeist, S.social_network, S.listening_probability(0),
               S.field, S.agreement(0,2,0), S.potential(0,2,0))
    print(test)
