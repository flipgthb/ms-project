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

def H(x):
    return erfc(x/sqrt2)/2

def zip_dict_items(a, b):
    for k in a.keys():
        try:
            a[k] = np.hstack([a[k], b[k]])
        except KeyError:
            raise Exception("dicts must have the same keys")

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
       return open(arg,'r')  #return an open file handle

def read_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-i", "--input", dest="filename", required=True,
        help="input file with model parameters.",
        metavar="FILE",
        type=lambda x: is_valid_file(parser,x)
    )

    parser.add_argument(
        'label',
        help='string: label given to simulation files'
    )

    args = parser.parse_args()
    return args

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
    D = config['D']
    dw = config['dw']
    sweeps = config['sweeps']

    n1, n2, n3, n4 = config['party_sizes']
    N = sum(config['party_sizes'])
    beta1, beta2, beta3 = config['political_pressure']
    beta4 = beta
    rho1, rho2, rho3 = config['liberal_index']
    rho4 = rho

    sqrtD = math.sqrt(D)
    norm = sqrtD
    Q = norm/sqrtD

    np.random.seed(78956347)

    block = (D-1)//2
    excess = D-1 - 2*block
    # agendas = np.array([
    #     [-1]+(D-1)*[1.0],
    #     [1]+block*[-1]+block*[1]+excess*[1],
    #     [1]+block*[1]+block*[-1]+excess*[1],
    # ])
    agendas = np.array([
        [3/2,-1/2,1,-1,-1],
        [1,1,1,1,1],
        [-1,-1,-1,-1,-1]
    ], dtype=np.float)

    w1 = np.vstack(n1*[agendas[0]]) #+ 0.1*np.random.randn(n1, D)
    w2 = np.vstack(n2*[agendas[1]]) #+ 0.1*np.random.randn(n2, D)
    w3 = np.vstack(n3*[agendas[2]]) #+ 0.1*np.random.randn(n3, D)
    w4 = np.random.randn(n4, D)

    w = np.vstack([w1,w2,w3,w4])
    w *= norm/np.sqrt((w*w).sum(axis=1))[:,None]
    agendas *= norm/np.sqrt((agendas*agendas).sum(axis=1))[:,None]
    # z = np.ones(D)
    # z *= norm/np.linalg.norm(z)

    O = lambda n, m: np.zeros((n,m))
    I = lambda n, m: np.ones((n,m))

    A1 = np.hstack([I(n1,n1), O(n1, n2), O(n1, n3), O(n1, n4)])
    A2 = np.hstack([O(n2,n1), I(n2, n2), O(n2, n3), O(n2, n4)])
    A3 = np.hstack([O(n3,n1), O(n3, n2), I(n3, n3), O(n3, n4)])
    A4 = np.hstack([I(n4,n1), I(n4, n2), I(n4, n3), I(n4, n4)])
    A = np.vstack([A1, A2, A3, A4])

    party1 = slice(0, n1)
    party2 = slice(n1, n1+n2)
    party3 = slice(n1+n2, n1+n2+n3)
    party4 = slice(n1+n2+n3, n1+n2+n3+n4)
    Beta = np.array(n1*[beta1]+n2*[beta2]+n3*[beta3]+n4*[beta4])
    Rho = np.array(n1*[rho1]+n2*[rho2]+n3*[rho3]+n4*[rho4])
    Gamma = np.sqrt(1 - Rho*Rho)/Rho
    L = np.array(n1*[0]+n2*[1]+n3*[2]+n4*[-1])

    def measure():
        h = agendas.dot(w.T)/norm/norm
        s = np.sign(h)
        m = h.mean(axis=1)
        # V1 = A*h[0]*h[0][:,None]
        # V2 = A*h[1]*h[1][:,None]
        # V3 = A*h[2]*h[2][:,None]
        V1 = A*s[0]*s[0][:,None]
        V2 = A*s[1]*s[1][:,None]
        V3 = A*s[2]*s[2][:,None]
        q1 = V1.mean()
        q2 = V2.mean()
        q3 = V3.mean()
        a21 = h[0, party2].mean()
        a31 = h[0, party3].mean()
        a41 = h[0, party4].mean()
        a42 = h[1, party4].mean()
        a43 = h[2, party4].mean()
        v1 = (V1[party1]+V2[party1]+V3[party1]).mean()
        v2 = (V1[party2]+V2[party2]+V3[party2]).mean()
        v3 = (V1[party3]+V2[party3]+V3[party3]).mean()
        v4 = (V1[party4]+V2[party4]+V3[party4]).mean()
        cov11 = V1[party1].mean()
        cov12 = V2[party1].mean()
        cov13 = V3[party1].mean()
        cov21 = V1[party2].mean()
        cov22 = V2[party2].mean()
        cov23 = V3[party2].mean()
        cov31 = V1[party3].mean()
        cov32 = V2[party3].mean()
        cov33 = V3[party3].mean()
        cov41 = V1[party4].mean()
        cov42 = V2[party4].mean()
        cov43 = V3[party4].mean()

        trace = dict(
            m_1=m[0],
            m_2=m[1],
            m_3=m[2],
            q_1=q1,
            q_2=q2,
            q_3=q3,
            a_21=a21,
            a_31=a31,
            a_41=a41,
            a_42=a42,
            a_43=a43,
            v_1=v1,
            v_2=v2,
            v_3=v3,
            v_4=v4,
            cov_11=cov11,
            cov_12=cov12,
            cov_13=cov13,
            cov_21=cov21,
            cov_22=cov22,
            cov_23=cov23,
            cov_31=cov31,
            cov_32=cov32,
            cov_33=cov33,
            cov_41=cov41,
            cov_42=cov42,
            cov_43=cov43,
        )
        return trace

    trace = measure()

    def energy(hi, hj, g):
        X = hi*np.sign(hj)/g/Q
        Ep = -g*g*np.log(eps + (1-2*eps)*erfc(-X/sqrt2)/2)
        return Ep

    for t in xrange(sweeps*N*2):
        i = np.random.choice(N)
        x = A[i]
        pij = x/x.sum()
        j = np.random.choice(N, p=pij)

        k = L[j]
        if k == -1:
            k = np.random.randint(3)
        z = agendas[k]
        h = z.dot(w.T)/sqrtD
        hi, hj = h[i], h[j]

        E0 = energy(hi, hj, Gamma[i])
        nw = dw*np.random.randn(D) + w[i]
        nw *= norm/np.linalg.norm(nw)
        nhi = z.dot(nw)/sqrtD
        E = energy(nhi, hj, Gamma[i])
        if E < E0 or np.random.rand() < math.exp(Beta[j]*(E0-E)):
            w[i] = nw.copy()

        if t%N == 0:
            more_data = measure()
            zip_dict_items(trace, more_data)

        if t == N*sweeps-1:
            Beta = np.array(n1*[beta2]+n2*[beta1]+n3*[beta3]+n4*[beta4])

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
    # config["zeitgeist"] = [1.0]*config["D"]
    D = config["D"]
    block = (D-1)//2
    excess = D-1 - 2*block
    config["agendas"] = np.array([
                            [3/2,-1/2,1,-1,-1],
                            [1,1,1,1,1],
                            [-1,-1,-1,-1,-1]
                        ]).tolist()
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

    # result = map(run, args[:3])
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
