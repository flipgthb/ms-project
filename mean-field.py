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

# -----------------------------------------------------------------------------
# Original Problem: 1 Group, 1 Zeitgeist - WORKING!!!!!
# -----------------------------------------------------------------------------

def B(theta,kalpha,delta,m,r):
    a = (1. + delta)/2.
    b = (1. - delta)/2.
    x = a*m*cos(theta)-b*r*abs(cos(theta))
    return(exp(kalpha*x))

def Z(kalpha,delta,m,r):
    res=integrate.quad(lambda x:B(x,kalpha,delta,m,r)*sin(x)**3,0,np.pi)
    return(res[0])

def m_int(m,r,kalpha,delta,Z):
    res=integrate.quad(lambda x:B(x,kalpha,delta,m,r)*cos(x)*sin(x)**3,0,np.pi)
    return(res[0]/Z)

def r_int(m,r,kalpha,delta,Z):
    res=integrate.quad(lambda x:B(x,kalpha,delta,m,r)*abs(cos(x))*sin(x)**3,
                       0,np.pi)
    return(res[0]/Z)

def mf_eqs(p,kalpha,delta):
    m,r=p
    cte=Z(kalpha,delta,m,r)
    return np.array((m_int(m,r,kalpha,delta,cte),r_int(m,r,kalpha,delta,cte)))

def mr(kalpha,delta):
    p=np.array((0.5,0.5))
    eps=1.
    n=0
    p = fixed_point(mf_eqs, p, args=(kalpha,delta))
    return p

# ----------------------------------------------------------------------------
# My Problem - Simpler: 2 Groups, 1 Zeitgeist - WORKING!!!!
# ----------------------------------------------------------------------------

def B1(theta,beta,delta,J,n1,m,r):
    m1, m2 = m
    r1, r2 = r
    n2 = 1-n1
    a = (1. + delta)/2.
    b = (1. - delta)/2.
    # x = a*m1*cos(theta)-b*r1*abs(cos(theta))
    # y = a*m2*cos(theta)-b*r2*abs(cos(theta))
    M = n1*m1 - J*n2*m2
    R = n1*r1 - J*n2*r2
    g = cos(theta)
    # return exp(beta*(x*n1 - J*y*n2))
    return exp(beta*(a*M*g - b*R*abs(g)))

def B2(theta,beta,delta,J,n1,m,r):
    m1, m2 = m
    r1, r2 = r
    n2 = 1-n1
    a = (1. + delta)/2.
    b = (1. - delta)/2.
    # x = a*m1*cos(theta)-b*r1*abs(cos(theta))
    # y = a*m2*cos(theta)-b*r2*abs(cos(theta))
    M = -J*n1*m1 + n2*m2
    R = -J*n1*r1 + n2*r2
    g = cos(theta)
    # return exp(beta*(x*n1 - J*y*n2))
    return exp(beta*(a*M*g - b*R*abs(g)))

def Z1(beta,delta,J,n1,m,r):
    res=integrate.quad(lambda x:B1(x,beta,delta,J,n1,m,r)*sin(x)**3,0,np.pi)
    return(res[0])

def Z2(beta,delta,J,n1,m,r):
    res=integrate.quad(lambda x:B2(x,beta,delta,J,n1,m,r)*sin(x)**3,0,np.pi)
    return(res[0])

def m1_int(m,r,beta,delta,J,n1,Z):
    res=integrate.quad(lambda x:B1(x,beta,delta,J,n1,m,r)*cos(x)*sin(x)**3,0,
                       np.pi)
    return(res[0]/Z)

def m2_int(m,r,beta,delta,J,n1,Z):
    res=integrate.quad(lambda x:B2(x,beta,delta,J,n1,m,r)*cos(x)*sin(x)**3,0,
                       np.pi)
    return(res[0]/Z)

def r1_int(m,r,beta,delta,J,n1,Z):
    res=integrate.quad(lambda x:B1(x,beta,delta,J,n1,m,r)*abs(cos(x))*sin(x)**3,
                       0,np.pi)
    return(res[0]/Z)

def r2_int(m,r,beta,delta,J,n1,Z):
    res=integrate.quad(lambda x:B2(x,beta,delta,J,n1,m,r)*abs(cos(x))*sin(x)**3,
                       0,np.pi)
    return(res[0]/Z)

def mf12_eqs(p,beta,delta,J,n1):
    m,r=p
    cte1=Z1(beta,delta,J,n1,m,r)
    cte2=Z2(beta,delta,J,n1,m,r)
    m1 = m1_int(m,r,beta,delta,J,n1,cte1)
    m2 = m2_int(m,r,beta,delta,J,n1,cte2)
    r1 = r1_int(m,r,beta,delta,J,n1,cte1)
    r2 = r2_int(m,r,beta,delta,J,n1,cte2)
    pnew = np.array([[m1,m2],
                     [r1,r2]])
    return pnew

def mr12(beta,delta,J,n1):
    p = np.array([[.5,-.5],
                  [.5,.5]])
    eps=1.
    n=0
    p = fixed_point(mf12_eqs, p, args=(beta,delta,J,n1))
    m, r = p
    return m, r

# -----------------------------------------------------------------------------
# My Problem - Full: 2 Groups, 2 Zeitgeists - FUCKING WORKS!!!!
# -----------------------------------------------------------------------------

def G(theta, phi, gamma):
    # return cos(theta)
    return cos(theta)*cos(gamma) + sin(theta)*sin(gamma)*cos(phi)
    # return 0.

def B1f(theta,phi,beta,delta,J,n1,gamma,m,r):
    m11, m21, m12, m22 = m.flat
    r11, r21, r12, r22 = r.flat
    n2 = 1-n1
    a = (1. + delta)/2.
    b = (1. - delta)/2.
    g1 = cos(theta)
    g2 = G(theta, phi, gamma)
    M11 = n1*m11 + J*n2*m21
    M12 = n1*m12 + J*n2*m22
    R11 = n1*r11 + J*n2*r21
    R12 = n1*r12 + J*n2*r22
    f = exp(beta*(a*M11*g1 - b*R11*abs(g1) + a*M12*g2 - b*R12*abs(g2)))
    return f

def B2f(theta,phi,beta,delta,J,n1,gamma,m,r):
    m11, m21, m12, m22 = m.flat
    r11, r21, r12, r22 = r.flat
    n2 = 1-n1
    a = (1. + delta)/2.
    b = (1. - delta)/2.
    g1 = cos(theta)
    g2 = G(theta, phi, gamma)
    M11 = J*n1*m11 + n2*m21
    M12 = J*n1*m12 + n2*m22
    R11 = J*n1*r11 + n2*r21
    R12 = J*n1*r12 + n2*r22
    f = exp(beta*(a*M11*g1 - b*R11*abs(g1) + a*M12*g2 - b*R12*abs(g2)))
    return f

def Z1f(beta,delta,J,n1,g,m,r):
    res=I(lambda phi,theta:B1f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2)
    return(res)

def Z2f(beta,delta,J,n1,g,m,r):
    res=I(lambda phi,theta:B2f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2)
    return(res)

def m11_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B1f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*cos(theta))
    return(res/Z)

def m21_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B2f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*cos(theta))
    return(res/Z)

def m12_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B1f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*G(theta,phi,g))
    return(res/Z)

def m22_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B2f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*G(theta,phi,g))
    return(res/Z)

def r11_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B1f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*abs(cos(theta)))
    return(res/Z)

def r21_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B2f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*abs(cos(theta)))
    return(res/Z)

def r12_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B1f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*abs(G(theta,phi,g)))
    return(res/Z)

def r22_int(m,r,beta,delta,J,n1,g,Z):
    res=I(lambda phi,theta:B2f(theta,phi,beta,delta,J,n1,g,m,r)*\
          sin(theta)**3*sin(phi)**2*abs(G(theta,phi,g)))
    return(res/Z)

def mf12f_eqs(p,beta,delta,J,n1,g):
    m,r=np.vsplit(p, 2)
    cte1=Z1f(beta,delta,J,n1,g,m,r)
    cte2=Z2f(beta,delta,J,n1,g,m,r)
    m11 = m11_int(m,r,beta,delta,J,n1,g,cte1)
    m21 = m21_int(m,r,beta,delta,J,n1,g,cte2)
    m12 = m12_int(m,r,beta,delta,J,n1,g,cte1)
    m22 = m22_int(m,r,beta,delta,J,n1,g,cte2)
    r11 = r11_int(m,r,beta,delta,J,n1,g,cte1)
    r21 = r21_int(m,r,beta,delta,J,n1,g,cte2)
    r12 = r12_int(m,r,beta,delta,J,n1,g,cte1)
    r22 = r22_int(m,r,beta,delta,J,n1,g,cte2)
    pnew = np.array([[m11,m21],
                     [m12,m22],
                     [r11,r21],
                     [r12,r22]])
    return pnew

def mr12f(beta,delta,J,n1,g):
    p = np.array([[.5,.01],
                  [.01,.5],
                  [.5,.01],
                  [.01,.5]])
    eps=1.
    p = fixed_point(mf12f_eqs, p, args=(beta,delta,J,n1,g),
                    xtol=1e-3, maxiter=100)
    m, r = np.vsplit(p, 2)
    return m, r

# # -----------------------------------------------------------------------------
# # My Problem - Full - Other Approach: 2 Groups, 2 Zeitgeists
# # -----------------------------------------------------------------------------

# vec = lambda x: np.vstack(x)

# def Gvec(theta, phi, gamma):
#     g1 = cos(theta)
#     g2 = cos(theta)*cos(gamma) + sin(theta)*sin(gamma)*cos(phi)
#     g = vec([g1,g2])
#     return g

# def Bvec(theta, phi, m, r, delta, beta, J, n, gamma):
#     X = np.array([[1., -J],
#                   [-J, 1.]])
#     N = vec([n,1-n])
#     G = Gvec(theta, phi, gamma)
#     Mg = X.dot((m*N)).dot(G)
#     Rg = X.dot((r*N)).dot(np.abs(G))
#     a = (1+delta)/2
#     b = (1-delta)/2
#     return np.exp(beta*(a*Mg-b*Rg))*(sin(theta)**3)*(sin(phi)**2)

# def Zvec(m, r, delta, beta, J, n, gamma):
#     z = lambda theta, phi: Bvec(theta, phi, m, r, delta, beta, J, n, gamma)
#     return I(z)

# def mvec(m, r, delta, beta, J, n, gamma, z):
#     m = lambda theta, phi: Bvec(theta, phi, m, r, delta, beta,
#                                 J, n, gamma)*(Gvec(theta, phi, gamma).T) / z
#     return I(m)

# def rvec(m, r, delta, beta, J, n, gamma, z):
#     r = lambda theta, phi: Bvec(theta, phi, m, r, delta, beta, J,
#                                 n, gamma)*np.abs(Gvec(theta, phi, gamma).T) / z
#     return I(r)

# def sce(p0, delta, beta, J, n, gamma):
#     m0, r0 = np.vsplit(p0, 2)
#     Z = Zvec(m0, r0, delta, beta, J, n, gamma)
#     m = mvec(m0, r0, delta, beta, J, n, gamma, Z)
#     r = rvec(m0, r0, delta, beta, J, n, gamma, Z)
#     return vec([m,r])

# def solve(delta, beta, J, n, gamma):
#     m0 = np.array([[.5,.5],
#                    [.5,.5]])
#     r0 = np.array([[.5,.5],
#                    [.5,.5]])
#     p0 = vec([m0,r0])
#     p = fixed_point(sce, p0, args=(delta, beta, J, n, gamma), xtol=1e-4)
#     m, r = np.vsplit(p, 2)
#     return m, r

if __name__ == "__main__":

    import multiprocessing as mp
    import time
    from datetime import timedelta

    # J = .1
    n = .5
    gamma = np.pi/3.
    # beta_max = 30.
    # x = [(d, b, J, n, gamma, beta_max)
    #      for b in np.arange(0,30,1.5)
    #      for d in [.4]]#np.arange(0,1.2,.2)])
    # x = enumerate(x)
    # x = [(d, b)
    #      for b in np.arange(0,30,.3)
    #      for d in [.4]]#np.arange(0,1.2,.2)])
    # x = enumerate(x)
    # x = [(d, b, J, n)
    #      for b in np.arange(0,100,1)
    #      for d in [.4]]#np.arange(0,1.2,.2)])
    # x = enumerate(x)
    x = [(d, b, J, n, gamma)
         for b in np.arange(0,120,20)
         for J in np.arange(-1,1,.02)
         for d in np.arange(0,1.2,.2)]
    x = enumerate(x)

    def f(x):
        idx, args = x
        p = mp.current_process().name
        d, b = args[:2]
        a = np.round(np.array([d,b]), decimals=2)
        msg1 = """Starting Process {0}
        args: {1}
        args-index: {2}""".format(p,a,idx)
        print(msg1)
        t0 = time.time()
        # r = idx, solve(*args)
        # r = idx, mr(*args)
        # r = idx, mr12(*args)
        r = idx, mr12f(*args)
        # r = idx, solve(*args)
        t = time.time()
        s = timedelta(seconds=(t-t0))
        msg2 = """Finishing Process {0}
        args: {1}
        took: {2}""".format(p,a,s)
        print(msg2)
        return r

    # print(solve(20,.4,J,n,gamma))

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

    np.save('/home/felippe/Desktop/mean-field-m-coop', m)
    np.save('/home/felippe/Desktop/mean-field-r-coop', r)

