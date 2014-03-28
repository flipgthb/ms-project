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
        self.k = N - 1 # number of neighbors
        self.delta = delta  # cognitive style, learning style
        self.beta = beta  # peer pressure

        self.w = np.random.randn(self.N, self.D)
        self.w /= row_norm(self.w)

        self.social_network = np.ones((self.N,self.N)) - np.identity(self.N)
        # self.social_network /= prob_row_norm(self.social_network)

        self.zeitgeist = self.social_network.dot(self.w)
        self.zeitgeist /= row_norm(self.zeitgeist)

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
        z  = self.agreement(i,j)
        Vij = -a*z + b*abs(z)
        return Vij

    @property
    def authority(self):
        C = self.social_network.T.dot(self.social_network)
        w, v = np.linalg.eig(C)
        i = np.argmax(w)
        A = v[:, i]
        A /= A.sum()
        A *= self.N
        return A

    @property
    def reputation(self):
        R = self.social_network.sum(axis=0).copy()
        R /= R.sum()
        R *= self.N
        return R

    def listening_probability(self, i):
        X = self.social_network[i].copy()
        X[i] = 0
        X[X<0] = 1e-5
        P = X / X.sum()
        return P


if __name__ == "__main__":
    S = Society(3,5,.2,5,.1)
    test = """
    S.N, S.D = {}, {}

    S.w =
    {}

    S.zeitgeist =
    {}

    S.listening_probability = {}

    S.reputation = {}

    S.authority = {}

    S.social_network =
    {}

    S.potential(0,2) = {}
    """.format(S.N,S.D,S.w,S.zeitgeist,S.listening_probability(0),
               S.reputation,S.authority,S.social_network,S.potential(0,2))
    print(test)
