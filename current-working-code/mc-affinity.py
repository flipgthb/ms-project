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
    A0 = np.asarray(nx.attr_matrix(G)[0])/2
    A = A0.copy()

    def measure():
        m = h.mean()/sqrtD
        r = np.abs(h).mean()/sqrtD
        v = h.var()
        q = np.abs(A*h*h[:,None]).mean()
        enemies = np.where(A>1/2,0,1)
        n = (enemies-np.diag(np.diag(enemies))).mean()
        data = pd.Series([m,v,q,r,n],
                         index="m v q r n".split(" "))
        return data

    trace = pd.DataFrame()
    measured = measure()
    measured["acceptance_ratio"] = 0
    trace = trace.append(measured, ignore_index=True)

    def energy(hi, hj):
        x = hi*np.sign(hj)/Gamma/sqrt2
        Jij = 1/D
        a = Jij
        return -a*Gamma*Gamma*np.log(eps + (1-2*eps)*erfc(-x)/2)

    accepted = 0
    for t in xrange(sweeps*N):
        i = np.random.choice(N)
        ni = A[i]
        pij = ni/ni.sum()
        j = np.random.choice(N, p=pij)

        hi, hj = h[i], h[j]
        rij = A[i,j]
        rij = rij + dr*rij*(1-rij)*np.sign(hi*hj)
        A[i,j] = rij

        E0 = energy(hi, hj)
        nw = np.random.multivariate_normal(w[i], dw*np.eye(D))
        # u = np.random.randn(D)
        # nw = dw*u/np.linalg.norm(u) + w[i]
        nw *= norm/np.linalg.norm(nw)
        nhi = z.dot(nw)/sqrtD
        E = energy(nhi, hj)
        if E < E0 or np.random.rand() < math.exp(beta*(E0-E)):
            accepted += 1
            w[i] = nw.copy()
            h[i] = nhi

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
    args = enumerate(points)

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

    ts = ""
    lt = time.localtime()
    ymd = lt.tm_year,lt.tm_mon,lt.tm_mday
    date = datetime.date(*ymd)
    ts = "_"+str(date)
    with open(path+"config"+ts+".json","w") as file_:
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
