#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import pandas as pd
import networkx as nx
import math

def zip_dicts(a, b):
    if set(a.keys()) != set(b.keys()):
        raise Exception("can't do it")
    for k in a.keys():
        a[k] = np.hstack([a[k],b[k]])


def program(tau, delta, beta, soc_dict, mc_dict):
    # mcmc variables
    autocor_time = mc_dict["autocor_time"]
    dw = mc_dict["dW"]
    epsilon = mc_dict["epsilon"]
    max_sweeps = mc_dict["max_sweeps"]
    therm_time = mc_dict["therm_time"]

    # society variables
    D = soc_dict["D"]
    N = soc_dict["topology"]["parameters"]['n']
    zeitgeist = soc_dict["zeitgeist"]

    w = np.ones((N, D)) / math.sqrt(D)
    z = zeitgeist / np.linalg.norm(zeitgeist)
    social_network = nx.complete_graph(N)
    lattice = np.asarray(nx.adjacency_matrix(social_network))
    reputation_score = lattice.copy()
    activity_record = np.zeros_like(lattice)

    # initial measurements
    def measure():
        h = z.dot(w.T)
        m = h.mean()
        r = np.abs(h).mean()
        v = h.var()
        x = np.zeros_like(reputation_score)
        x[reputation_score<1] = 1
        n_op = x.mean()
        R = reputation_score.mean()
        y = np.zeros_like(x)
        y[reputation_score<R] = 1
        G = y.mean()
        got = dict(
            m=np.array([m]),
            r=np.array([r]),
            v=np.array([v]),
            n_op=np.array([n_op]),
            R=np.array([R]),
            G=np.array([G])
        )
        return got

    trace = measure()

    # energy function
    def energy(hi, hj, d):
        a, b = (1 + d)/2, (1 - d)/2
        x = hi*hj
        return -a*x + b*abs(x)

    # metropolis
    n = 0
    t = 0
    for _ in xrange(max_sweeps*N):
        i = np.random.choice(N)
        x = reputation_score[i].copy()
        pij = x - x.min()
        pij[i] = 0
        pij /= pij.sum()
        j = np.random.choice(N, p=pij)
        h = z.dot(w.T)
        hi, hj = h[[i,j]]
        activity_record[i,j] += 1
        reputation_score[i,j] += hi*hj*epsilon
        # reputation_score[i,j] += np.sign(hi*hj)*epsilon

        if hi*hj >= -tau:
            V0 = energy(hi, hj, delta)
            new_w = np.random.multivariate_normal(w[i], dw*np.identity(D))
            new_w /= np.linalg.norm(new_w)
            new_hi = new_w.dot(z)
            V = energy(new_hi, hj, delta)
            bdV = beta*(V - V0)
            acc = min(0, -bdV)
            rej = math.log(np.random.rand())
            if rej < acc:
                w[i] = new_w.copy()

        n += 1
        if n == N:
            n = 0
            t += 1
            if t >= therm_time and t%autocor_time == 0:
                more_data = measure()
                zip_dicts(trace, more_data)

    # organizing data and finishing
    df = pd.DataFrame(trace)
    sn = lattice
    rep = reputation_score
    act = activity_record
    data = df, sn, rep, act, w

    return data
