#!/usr/bin/env python
# encoding: utf-8

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
import datetime
from datetime import timedelta

sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2*np.pi)

def H(x):
    return erfc(x/sqrt2)/2

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
       return open(arg,'r')

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

def save_data(path, data_tuple, time_stamp=True):

    ts = ""
    if time_stamp:
        lt = time.localtime()
        ymd = lt.tm_year,lt.tm_mon,lt.tm_mday
        date = datetime.date(*ymd)
        ts = "_"+str(date)


    data = dict.fromkeys(data_tuple[0].keys())
    for entry in data_tuple:
        for k,v in entry.items():
            try:
                data[k].append(v)
            except:
                data[k] = [v]

    for file_name, data_list in data.items():
        full_name = path + file_name + ts + ".csv"
        with open(full_name, "w") as file_:
            panel = pd.concat(data_list)
            panel.to_frame().to_csv(file_)


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
    h = z.dot(w.T)/sqrtD

    G = nx.complete_graph(N)
    # G = nx.connected_watts_strogatz_graph(N, 10, .1)
    A0 = np.asarray(nx.attr_matrix(G)[0])/2
    # A0 *= (1+ h*h[:,None]/D)/2
    # A0 += 0.4*np.random.randn(*A0.shape)
    # A0 -= np.diag(np.diag(A0))
    A = A0.copy()

    def measure():
        # h = z.dot(w.T)/D
        m = h.mean()/sqrtD
        v = h.var()
        q = np.abs(A*h*h[:,None]).mean()
        Enemies = np.where(A < A0, 1, 0)
        n = Enemies.mean()
        data = pd.Series([m,v,q,n],
                         index="m v q n".split(" "))
        return data

    trace = pd.DataFrame()
    measured = measure()
    measured["acceptance_ratio"] = 0
    trace = trace.append(measured, ignore_index=True)

    def energy(hi, hj, rij):
        x = hi*np.sign(hj)/Gamma/sqrt2
        # Jij = np.sign(rij - 1/2)
        Jij = 1.0
        a = Jij
        return -a*Gamma*Gamma*np.log(eps + (1-2*eps)*erfc(-x)/2)

    accepted = 0
    for t in xrange(sweeps*N):
        i = np.random.choice(N)
        ni = np.where(A[i] != 0, 1, 0)
        # ni = A[i]
        pij = ni/ni.sum()
        j = np.random.choice(N, p=pij)

        # h = z.dot(w.T)/sqrtD
        hi, hj = h[i], h[j]

        # rij0 = A[i,j]
        # rij = rij0 + rij0*(1-rij0)*np.sign(hi*hj)*dr
        # rij = rij + pI*(1-pI)*np.sign(hi*hj)*dr
        # A[i,j] = rij

        pI = (1+hi*hj/D)/2
        rij0 = A[i,j]
        rij = pI
        # pI = 1.0
        # pI = rij
        if np.random.rand() < pI:
            E0 = energy(hi, hj, rij0)
            u = np.random.randn(D)
            nw = dw*u/np.linalg.norm(u) + w[i]
            nw *= norm/np.linalg.norm(nw)
            nhi = z.dot(nw)/sqrtD
            E = energy(nhi, hj, rij)
            if E < E0 or np.random.rand() < math.exp(beta*(E0-E)):
                accepted += 1
                w[i] = nw.copy()
                h[i] = nhi
                A[i,j] = pI

        if t%N == 0:
            measured = measure()
            measured["acceptance_ratio"] = accepted/(t+1)
            trace = trace.append(measured, ignore_index=True)

    p = (eps, rho, beta)
    names = ("epsilon","rho","beta")
    stats = pd.Panel({p:trace})
    result = {"statistics":stats}
    result["agents"]= pd.Panel({p:pd.DataFrame(w)})
    result["network"] = pd.Panel({p:pd.DataFrame(A)})
    for k,v in result.items():
        v.items.set_names(names, inplace=True)
    return result

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
    # order = np.arange(points.shape[0])[:,np.newaxis]
    args = enumerate(points)# np.hstack([order, points])

    def run(x):
        idx, (eps, rho, beta) = x
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
    result = [res[-1] for res in result]
    t = time.time()
    print("pool.map(run,args) took %s"%(timedelta(seconds=(t-t0))))

    path = config["save_dir"] + label + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path+"config.json","w") as file_:
        json.dump(config, file_, indent=4)

    t0_save = time.time()
    save_data(path, result)
    t_save = time.time()
    print("save_data took%s"%(timedelta(
                              seconds=(t_save-t0_save))))
    t_final = time.time()
    print("Total time spent: %s"%(timedelta(seconds=(t_save-t0))))
    print("Finished at: ", time.asctime())
    print(40*"=")
