#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
from scipy.integrate import quad
import sympy as sy

@np.vectorize
def vec_lambdify(var, expr, *args, **kw):
    return sy.lambdify(var, expr, *args, **kw)

@np.vectorize
def vec_quad(f, a, b, *args, **kw):
    return quad(f, a, b, *args, **kw)[0]

if __name__ == "__main__":

    Y = sy.symbols("y1:11")
    x = sy.symbols("x")
    mul_x = [y.subs(y,x*(i+1)) for (i,y) in enumerate(Y)]
    pow_x = [y.subs(y,x**(i+1)) for (i,y) in enumerate(Y)]

    g_sympy = np.array(mul_x + pow_x).reshape((2,10))
    X = x*np.ones_like(g_sympy)
    G = vec_lambdify(X, g_sympy)
    I = vec_quad(G, 0, 100)
    print(I)
