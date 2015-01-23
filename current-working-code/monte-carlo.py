#!/usr/bin/env python
# encoding: utf-8

# -----------------------------------------------------------------------------
# ---------------------------------- Modules ----------------------------------
# ---------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
import networkx as nx
import pandas as pd
from scipy.special import erfc, erf
import math
from argparse import ArgumentParser
import os
import json
import multiprocessing as mp
import time
from datetime import timedelta

# -----------------------------------------------------------------------------
# ---------------------------- Utility Functions ------------------------------
# -----------------------------------------------------------------------------
sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2*np.pi)


def save_data(idx_data_list, config, label=["epsilon", "rho", "beta"]):
    d0 = idx_data_list.pop(0)[-1]
    stat_dict = d0["stat"]
    rep_dict = d0["rep"]
    agt_dict = d0["agt"]
    for entry in idx_data_list:
        d = entry[-1]
        stat_dict.update(d["stat"])
        rep_dict.update(d["rep"])
        agt_dict.update(d["agt"])
    stat_panel = pd.Panel(stat_dict)
    rep_panel = pd.Panel(rep_dict)
    agt_panel = pd.Panel(agt_dict)
    stat_panel.items.set_names(label, inplace=True)
    rep_panel.items.set_names(label, inplace=True)
    agt_panel.items.set_names(label, inplace=True)

    SAVE_DIR = config["save_directory"] + config["label"]
    if os.path.exists(config["save_directory"]):
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
    else:
        raise Exception("%s is not a valid path"%config["save_directory"])

    with open(SAVE_DIR+'/config.json', 'w') as file_:
        json.dump(config, file_, indent=4)

    with open(SAVE_DIR+'/statistics.csv', 'w') as file_:
        df = stat_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/reputation.csv', 'w') as file_:
        df = rep_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/state.csv', 'w') as file_:
        df = agt_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)


# -----------------------------------------------------------------------------
# ---------------------------- Monte Carlo ------------------------------------
# -----------------------------------------------------------------------------
def mc(rho, beta, eps, config):
    N = config['N']
    D = config['D']
    dw = config['dw']
    dr = config['dr']
    sweeps = config['sweeps']

    sqrtD = math.sqrt(D)
    sqrtN = math.sqrt(N)
    norm = sqrtD
    Q = norm/sqrtD
    Gamma = math.sqrt(1 - rho*rho)/rho

    w = np.random.randn(N,D)
    w *= norm/np.sqrt((w*w).sum(axis=1))[:,None]

    z = np.ones(D)
    z *= norm/np.linalg.norm(z)

    A0 = np.random.rand(N,N)
    A0 -= np.diag(np.diag(A0))
    A = A0.copy()

    def measure(acc):
        h = z.dot(w.T)/norm/norm
        s = np.sign(h)
        m = h.mean()
        r = A.mean()
        Op = np.where(A<A0, 1, 0)
        n_op = Op.mean()
        q = (A*h*h[:,None]).mean()
        n_a = (1 - (A*s*s[:,None]).mean())/2
        trace = dict(
            m=m,
            r=r,
            q=q,
            n_op=n_op,
            n_a=n_a,
            accepted=acc
        )
        return trace

    trace = measure(0)

    def H(x):
        return erfc(x/sqrt2)/2

    def energy(hi, hj, rij):
        X = hi*np.sign(hj)/Gamma/Q
        Ep = -sqrt2pi*Gamma*Q*np.log(eps + (1-2*eps)*erfc(-X/sqrt2)/2)
        return Ep

    accepted = 0
    for t in xrange(sweeps*N):
        i = np.random.choice(N)
        x = A[i]
        pij = x/x.sum()
        j = np.random.choice(N, p=pij)

        h = z.dot(w.T)/sqrtD
        hi, hj = h[i], h[j]

        rij = A[i,j]
        A[i,j] = rij + rij*(1-rij)*np.sign(hi*hj)*dr

        E0 = energy(hi, hj, rij)
        nw = dw*np.random.randn(D) + w[i]
        nw *= norm/np.linalg.norm(nw)
        nhi = z.dot(nw)/sqrtD
        E = energy(nhi, hj, A[i,j])
        if E < E0 or np.random.rand() < math.exp(beta*(E0-E)):
            accepted += 1
            w[i] = nw.copy()

        if t%N == 0:
            more_data = measure(accepted/(t+1))
            zip_dict_items(trace, more_data)

    dfs = pd.DataFrame(trace)
    dfA = pd.DataFrame(A)
    dfw = pd.DataFrame(w)
    p = (eps, rho, beta)
    stat_dict = {p:dfs}
    A_dict = {p:dfA}
    w_dict = {p:dfw}
    return dict(stat=stat_dict, rep=A_dict, agt=w_dict)


# -----------------------------------------------------------------------------
# ---------------------------- Main Program -----------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    args = read_args()

    input_ = args.filename
    label = args.label
    config = json.load(input_)
    config["label"] = label
    config["zeitgeist"] = [1.0]*config["D"]
    input_.close()

    beta_grid =  slice(*config["beta"])
    rho_grid = slice(*config["rho"])
    eps_grid = slice(*config["epsilon"])
    grid = np.mgrid[eps_grid, rho_grid, beta_grid]
    points = np.vstack([x.ravel() for x in grid]).T
    order = np.arange(points.shape[0])[:,np.newaxis]
    args = np.hstack([order, points])

    def run(x):
        idx, eps, rho, beta = x
        r = mc(rho, beta, eps, config)
        return idx, r

    pool = mp.Pool()
    t0 = time.time()
    print(40*"=")
    print("Passed:", json.dumps(config, indent=4), sep="\n")
    print("Starting at: ", time.asctime())
    result = pool.map(run, args)
    pool.close()
    pool.join()
    result.sort()
    t = time.time()
    print("pool.map(run, args) took %s"%(timedelta(seconds=(t-t0))))

    t0_save = time.time()
    save_data(result, config)
    t_save = time.time()
    print("save_data took %s"%(timedelta(seconds=(t_save-t0_save))))
    t_final = time.time()
    print("Total time spent: %s"%(timedelta(seconds=(t_save-t0))))
    print("Finished at: ", time.asctime())
    print(40*"=")
