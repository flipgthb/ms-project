#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import pandas as pd
import json
import os
import argparse
import multiprocessing as mp
from Society import Society
from MCMC import MCMC
import Parameter as pmt

SAVE_DIR = "/media/backup/simulation-data/ms-project/data/"
# SAVE_DIR = "/home/felippe/Desktop/"


def run(x):
    idx, beta, rho, gamma = x
    N = 64
    D = 5
    n = 25000
    b = 5000
    m = 100
    eps = .2
    df = 1.
    ds = 1.
    w = np.ones((N,D))
    S = Society(w=w, gamma=gamma, beta=beta, rho=rho, eps=eps)
    mc = MCMC(system=S, n=n, b=b, m=m, df=df, ds=ds)
    mc.sample()
    r = mc.data
    return idx, r

def concatenate_dicts(*dicts):
    keys = set().union(*dicts)
    c = {}
    for k in keys:
        x = []
        for d in dicts:
            x.append(d.get(k, ''))
        c[k] = np.array(x)
    return c

def save_data(data_list):

    x = np.vstack(data_list)[:,1:].flat
    explain = x[0]["explain"]
    data = concatenate_dicts(*x)
    for k, v in data.items():
        data[k] = np.vstack(v)

    with open(SAVE_DIR+'statistics.csv', 'w') as file_:
        df = pd.DataFrame(data["statistics"], columns=explain)
        df.to_csv(file_, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'adjacency.npy', 'w') as file_:
        np.save(file_, data["adjacency"])

    with open(SAVE_DIR+'state.npy', 'w') as file_:
        np.save(file_, data["state"])


def simulation():

    beta_grid = slice(0, 50, 100j)
    rho_grid = slice(1e-4, 0.999, 10j)
    gamma_grid = slice(0, np.pi, 10j)

    grid = np.mgrid[beta_grid, rho_grid, gamma_grid]
    points = np.vstack([x.ravel() for x in grid]).T
    order = np.arange(points.shape[0])[:,np.newaxis]
    args = np.hstack([order, points])

    pool = mp.Pool()
    result = pool.map(run, args)
    pool.close()
    pool.join()

    result.sort()
    save_data(result)


if __name__ == "__main__":
    simulation()
