#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import math
from scipy.special import erf

def row_norm(X):
    nX = np.sqrt((X*X).sum(axis=1)[:, np.newaxis])
    return nX

def prob_row_norm(X):
    nX = X.sum(axis=1)[:, np.newaxis]
    return nX

class Society(object):

    def __init__(self, N, D, delta, beta):

        self.N = N  # number of agents
        self.D = D  # agent complexity, cognitive vector dimension
        self.delta = delta
        self.beta = beta

        self.w = np.random.randn(self.N, self.D)
        self.w /= row_norm(self.w)

        self.social_network = np.random.rand(self.N,self.N)
        self.social_network -= np.diag(np.diag(self.social_network))
        self.social_network *= self.N / prob_row_norm(self.social_network)

        self.zeitgeist = self.compute_initial_zeitgeist()

    def compute_initial_zeitgeist(self):
        z = self.social_network.dot(self.w)
        z /= row_norm(z)
        return z

    @property
    def reputation(self):
        A = self.social_network
        R = A.sum(axis=0)
        return R

    @property
    def listening_probability(self):
        min_ = self.social_network.min(axis=1)[:, np.newaxis]
        max_ = self.social_network.max(axis=1)[:, np.newaxis]
        X = self.social_network.copy()
        P = (X - min_)/(max_ - min_)
        P -= np.diag(np.diag(P))
        return P / prob_row_norm(P)

    @property
    def field(self):
        h = (self.zeitgeist*self.w).sum(axis=1)
        return h

    def agreement(self, i, j):
        hi, hj = self.field[[i,j]]
        return hi*hj

    def potential(self, i, j):
        a =  (1+self.delta)/2
        b =  (1-self.delta)/2
        hi, hj  = self.field[[i,j]]
        Vij = -a*hi*hj + b*abs(hi*hj)
        return Vij



if __name__ == "__main__":
    S = Society(3,5,.2,5)
    test = """
    S.N, S.D = {}, {}

    S.w =
    {}

    S.zeitgeist =
    {}

    S.listening_probability = {}

    S.reputation = {}

    S.social_network =
    {}

    S.potential(0,2) = {}
    """.format(S.N,S.D,S.w,S.zeitgeist,S.listening_probability,
               S.reputation,S.social_network,S.potential(0,2))
    print(test)
