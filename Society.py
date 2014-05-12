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

    def __init__(self, N, k, D, delta, beta, gamma, rewiring_prob=1/2):

        self.N = N  # number of agents
        self.D = D  # agent complexity, cognitive vector dimension
        self.k = k # number of neighbors
        self.rewiring_prob = rewiring_prob # watts-strogatz rewiring probability
        self.delta = delta  # cognitive style, learning style
        self.beta = beta  # peer pressure
        self.gamma = gamma  # bullshit threshold

        self.w = np.random.randn(self.N, self.D)
        self.w /= row_norm(self.w)

        # G = nx.watts_strogatz_graph(self.N, self.k, self.rewiring_prob)
        # G = nx.barabasi_albert_graph(self.N, self.k)
        G = nx.complete_graph(self.N)
        self.social_graph = G
        self.social_network = np.array(nx.adjacency_matrix(G))
        # self.social_network = np.zeros((self.N,self.N))
        self.initial_social_network = self.social_network.copy()

        self.zeitgeist = np.random.randn(self.D)
        self.zeitgeist /= np.linalg.norm(self.zeitgeist)

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
        Vij = -a*z + b*abs(z)
        return Vij

    @property
    def reputation(self):
        R = self.social_network.sum(axis=0).copy()
        R *= self.N/R.sum()
        return R

    def neighbor_weights(self, i):
        y = self.social_network[i].copy()
        x = np.exp(y)
        x[i] = 0
        P = x/x.sum()
        return P

    def agent_weights(self):
        y = self.social_network.sum(axis=0).copy()
        x = np.exp(-y/np.abs(y).max())
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
