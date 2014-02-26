#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import math
from scipy.special import erf

def row_norm(X):
    nX = np.sqrt((X*X).sum(axis=1)[:, np.newaxis])
    return nX

def build_pair(dim, angle, plane=(0,1)):
    # Poor realization of a rotation in a plane, resulting in a
    # pair of vectors with the given angle between then in this
    # given plane. It is poor in the sense that, actually, the
    # function is performing a rotantion in 2 dimensions and
    # embending the vectors in dim dimensions.
    c = math.cos(angle)
    s = math.sin(angle)
    z1 = np.zeros(dim)
    z2 = np.zeros(dim)
    z1[list(plane)] = 1, 0
    z2[list(plane)] = c, s
    return np.vstack([z1, z2])

class Society(object):

    def __init__(self, w, gamma, rho, eps, beta):

        self.N = w.shape[0]
        self.D = w.shape[1]

        self.gamma = gamma
        self.rho = rho
        self.eps = eps
        self.beta = beta

        self.state = w / row_norm(w)
        self.zeitgeist = build_pair(self.D, self.gamma)
        self.adjacency = np.ones((self.N, self.N)) - np.identity(self.N)

    @property
    def field(self):
        fi = self.state.dot(self.zeitgeist.T)
        return fi

    def agreement(self, i, j):
        h = self.field
        hi, hj = h[i], h[j]
        # hi, hj = self.field[[i, j]]
        z_idx = np.argmax(hj)
        s = np.sign(hj[z_idx])
        x = hi[z_idx] * s
        return x

    @property
    def topology(self):
        A = self.adjacency
        n = A.sum(axis=1)[:,np.newaxis]
        return A/n

    def energy(self, i, j):
        x = self.agreement(i, j)
        C = (1/self.rho) * np.sqrt(1 - self.rho*self.rho)
        Fi = 1/2*(1+erf(x/(C*math.sqrt(2))))
        E = math.log(self.eps + (1 - 2*self.eps)*Fi)
        return E


if __name__ == "__main__":

    w = np.ones((4, 5))
    w[2:] *= -1
    rho = .8
    eps = .2
    beta = 1.
    gamma = 3*np.pi/4
    s = Society(w, gamma, rho, eps, beta)
    report = """
    w =
    {0}

    rho={1},  beta={2},  gamma={3}, eps={4}

    z =
    {5}

    z*z.T =
    {6}

    arccos(z*z.T) =
    {7}

    energy(0,1) = {8}
    agreement(0,1) = {9}
    (w*w).sum(axis=1) = {10}
    field = {11}
    indmax(field) = {12}
    """.format(s.state, s.rho, s.beta, s.gamma, s.eps,
               s.zeitgeist, s.zeitgeist.dot(s.zeitgeist.T),
               np.arccos(s.zeitgeist.dot(s.zeitgeist.T)),
               s.energy(0,1), s.agreement(0,1), (s.state*s.state).sum(axis=1),
               s.field, s.field.argmax(axis=1))
    print(report)
