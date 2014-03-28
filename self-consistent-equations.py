#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import pylab as pl
from scipy.optimize import fixed_point
from scipy.integrate import dblquad
from math import sin, cos, exp

y0 = lambda x: 0
y1 = lambda x: np.pi
Dt = 1.

def g(p, t, gm):
    x = cos(t)*cos(gm) + sin(t)*sin(gm)*cos(p)
    return x

def F(p, t, m0, r0, X, n, delta, gamma0):
    G = np.vstack([cos(t), g(p,t, gamma0)])
    M = X.dot((m0*n))
    R = X.dot((r0*n))
    a = (1+delta)/2
    b = (1-delta)/2
    f = a*M.dot(G) - b*R.dot(np.abs(G))
    return f

def Z_int(p, t, m0, r0, X, n, delta, beta, gamma0):
    x = (sin(t)**3)*(sin(p)**2)*np.exp(beta*F(p, t, m0, r0,
                                     X, n, delta, gamma0))
    return x

def m_int(p, t, gamma, m0, r0, X, n, delta, beta, gamma0):
    x = (sin(t)**3)*(sin(p)**2)*g(p,t,gamma)*np.exp(beta*F(p, t, m0, r0, X,
                                                           n, delta, gamma0))
    return x

def r_int(p, t, gamma, m0, r0, X, n, delta, beta, gamma0):
    x = (sin(t)**3)*(sin(p)**2)*abs(g(p,t,gamma))*np.exp(beta*F(p, t, m0, r0, X,
                                                         n, delta, gamma0))
    return x

def sce(p0, X, n, delta, beta, gamma):
    m0, r0 = np.hsplit(p0, 2)
    Z = np.vstack([dblquad(Z_int, 0.0, np.pi, y0, y1,
                           args=(m0,r0,X[i],n,delta, beta, gamma))[0]
                   for i in [0,1]])
    m = np.array([dblquad(m_int, 0.0, np.pi, y0, y1,
                          args=(g,m0,r0,X[i],n,delta, beta, gamma))[0]
                  for g in [0, gamma] for i in [0,1]]).reshape((2,2)) / Z
    r = np.array([dblquad(r_int, 0.0, np.pi, y0, y1,
                          args=(g,m0,r0,X[i],n,delta, beta, gamma))[0]
                  for g in [0, gamma] for i in [0,1]]).reshape((2,2)) / Z
    return np.hstack([m, r])

def solve(J, n1, delta, beta, gamma):
    X = np.array([[1., -J],
                  [-J, 1.]])
    n = np.vstack([n1, 1-n1])
    m0 = np.array([[5.,-5.],
                   [-5.,5.]])
    r0 = np.array([[.5,.5],
                   [.5,.5]])
    p0 = np.hstack([m0,r0])
    p = fixed_point(sce, p0, args=(X, n, delta, beta, gamma))
    m, r = np.hsplit(p, 2)
    return m, r


if __name__ == "__main__":

    import multiprocessing as mp
    import time
    from datetime import timedelta

    J = 0.
    n = 1.
    gamma = 0.
    x = enumerate([(J,n,d,b,gamma)
                   for b in np.arange(0,30,.3)
                   for d in np.arange(0,1.2,.2)])

    def f(x):
        return x[0], solve(*x[1])

    t0 = time.time()
    pool = mp.Pool()
    res = pool.map(f,x)
    pool.close()
    pool.join()
    t = time.time()
    print("solve function took %s"%timedelta(seconds=(t-t0)))

    res.sort()
    m = np.array([p[1][0] for p in res])
    r = np.array([p[1][1] for p in res])

    np.save('/home/felippe/Desktop/mf-test-m', m)
    np.save('/home/felippe/Desktop/mf-test-r', r)

