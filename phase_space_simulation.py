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
import time
from datetime import timedelta

SAVE_DIR = "/media/backup/simulation-data/ms-project/data/"
# SAVE_DIR = "/home/felippe/Desktop/"


def run(x):
    idx, delta, beta = x
    N = 64
    D = 5
    n = 25000
    b = 5000
    m = 100
    eps = .2
    df = 1.
    ds = 1.
    S = Society(N=N, D=D, delta=delta, beta=beta)
    mc = MCMC(system=S, num_steps=n, burn=b, measure_period=m,
              d_omega=df, d_eps=ds)
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

    with open(SAVE_DIR+'statistics_3.csv', 'w') as file_:
        df = pd.DataFrame(data["statistics"], columns=explain)
        df.to_csv(file_, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'social-network_3.npy', 'w') as file_:
        np.save(file_, data["social_network"])

    with open(SAVE_DIR+'state_3.npy', 'w') as file_:
        np.save(file_, data["state"])

    with open(SAVE_DIR+'zeitgeist_3.npy', 'w') as file_:
        np.save(file_, data["zeitgeist"])

def simulation():

    beta_grid = slice(0, 50, .5)
    delta_grid = slice(0, 1.2, .2)

    grid = np.mgrid[delta_grid, beta_grid]
    points = np.vstack([x.ravel() for x in grid]).T
    order = np.arange(points.shape[0])[:,np.newaxis]
    args = np.hstack([order, points])


    pool = mp.Pool()
    t0 = time.time()
    result = pool.map(run, args)
    pool.close()
    pool.join()
    t = time.time()
    print("simulation took %s"%(timedelta(seconds=(t-t0))))

    result.sort()
    save_data(result)


if __name__ == "__main__":
    simulation()
