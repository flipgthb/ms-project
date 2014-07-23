#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import pylab as pl
from scipy.integrate import dblquad
from scipy import integrate
from scipy import optimize
from math import sin, cos, exp


def fixed_point(func, x0, args=(), xtol=1e-4, maxiter=500, dt=1.0):
    p = np.asarray(x0)
    eps = 1
    n = 0
    while (eps>xtol and n<maxiter):
        pnew = func(p,*args)
        eps = np.max(np.fabs(pnew-p))
        p = (1-dt)*p + dt*pnew.copy()
        n = n+1
        if eps<xtol:
            msg = "Converged after %d iterations, value is %s"%(n, p)
            print(msg)
            return p
    msg = "Failed to converge after %d iterations, value is %s, with eps=%.2f"
    print(msg%(maxiter,p,eps))
    return p

#@np.vectorize
def I(func, *args):
    return integrate.dblquad(
        func, 0, np.pi, lambda x: 0, lambda x: np.pi, args=args
    )[0]

def quad(func, *fargs):
    return integrate.quad(func, 0, np.pi, args=fargs)[0]


def Ba(theta,m,r,beta,J,delta):
    a = (1. + delta)/2.
    b = (1. - delta)/2.
    ma, mb = m
    ra, rb = r
    Ma = ma + J*mb
    Ra = ra + J*rb
    x = a*Ma*cos(theta)-b*Ra*abs(cos(theta))
    return(exp(beta*x))

def Bb(theta,m,r,beta,J,delta):
    a = (1. + delta)/2.
    b = (1. - delta)/2.
    ma, mb = m
    ra, rb = r
    Mb = mb + J*ma
    Rb = rb + J*ra
    x = a*Mb*cos(theta)-b*Rb*abs(cos(theta))
    return(exp(beta*x))

def Za(m,r,beta,J,delta):
    res=integrate.quad(lambda x:Ba(x,m,r,beta,J,delta)*sin(x)**3,0,np.pi)
    return(res[0])

def Zb(m,r,beta,J,delta):
    res=integrate.quad(lambda x:Bb(x,m,r,beta,J,delta)*sin(x)**3,0,np.pi)
    return(res[0])

def ma_int(m,r,beta,J,delta,Z):
    res=integrate.quad(lambda x:Ba(x,m,r,beta,J,delta)*cos(x)*sin(x)**3,0,np.pi)
    return(res[0]/Z)

def mb_int(m,r,beta,J,delta,Z):
    res=integrate.quad(lambda x:Bb(x,m,r,beta,J,delta)*cos(x)*sin(x)**3,0,np.pi)
    return(res[0]/Z)

def ra_int(m,r,beta,J,delta,Z):
    res=integrate.quad(lambda x:Ba(x,m,r,beta,J,delta)*abs(cos(x))*sin(x)**3,
                       0,np.pi)
    return(res[0]/Z)

def rb_int(m,r,beta,J,delta,Z):
    res=integrate.quad(lambda x:Bb(x,m,r,beta,J,delta)*abs(cos(x))*sin(x)**3,
                       0,np.pi)
    return(res[0]/Z)

def mf_eqs(p,beta,J,delta):
    m, r = np.vsplit(p, 2)
    m = m.reshape((2,))
    r = r.reshape((2,))
    Ca=Za(m,r,beta,J,delta)
    Cb=Zb(m,r,beta,J,delta)
    ma = ma_int(m,r,beta,J,delta,Ca)
    mb = mb_int(m,r,beta,J,delta,Cb)
    ra = ra_int(m,r,beta,J,delta,Ca)
    rb = rb_int(m,r,beta,J,delta,Cb)
    p_new = np.array([[ma,mb],
                      [ra,rb]])
    return p_new

def mr(beta,J,delta):
    x = .5
    p=np.array([[x,-x],
                [x,x]])
    p = fixed_point(mf_eqs, p, args=(beta,J,delta))
    return p


if __name__ == "__main__":

    import multiprocessing as mp
    import time
    from datetime import timedelta

    # J = -1.
    x = [(b, J, d)
         for b in np.arange(0,100,1)
         for J in np.arange(-1,1,.02)
         for d in np.arange(0,1.2,.2)]
    x = enumerate(x)

    def f(x):
        idx, args = x
        p = mp.current_process().name
        b,J,d = args
        a = np.round(np.array([b,J,d]), decimals=2)
        msg1 = """Starting Process {0}
        args: {1}
        args-index: {2}""".format(p,a,idx)
        print(msg1)
        t0 = time.time()
        r = idx, mr(*args)
        t = time.time()
        s = timedelta(seconds=(t-t0))
        msg2 = """Finishing Process {0}
        args: {1}
        took: {2}""".format(p,a,s)
        print(msg2)
        return r

    # print(mr(20,J,.4))

    t0 = time.time()
    pool = mp.Pool()
    res = pool.map(f,x)
    pool.close()
    pool.join()
    t = time.time()
    print("TOTAL TIME SPENT: %s"%timedelta(seconds=(t-t0)))

    res.sort()
    m = np.array([p[1][0] for p in res])
    r = np.array([p[1][1] for p in res])

    np.save('/home/felippe/Desktop/mean-field-m-simple-1', m)
    np.save('/home/felippe/Desktop/mean-field-r-simple-1', r)
