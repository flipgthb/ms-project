#! /usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import pandas as pd
from scipy.integrate import quad
import math
from argparse import ArgumentParser
import os
import json
import multiprocessing as mp
import time
import datetime
from datetime import timedelta


def is_valid_file(parser, arg):
    """Check if the given arg is an existent file"""
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
       return open(arg,'r')  #return an open file handle

def read_args():
    """Parsing the command line arguments. Defines 'input' and 'label'"""
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
    """Organize and save a tuple composed of dict with format 'data_source':'pandas object'"""
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
            panel.to_csv(file_)

# --- self consistent equations ---
def Psi1(h, m1, r1, m2, r2, n1, n2, rho, eps, J, beta):
    A = (1 - rho)/2
    B = rho/2
    Vmf1 = -A*h*m1 + B*np.abs(h)*r1 - np.abs(1-2*eps+h*m1)/2
    Vmf2 = -A*h*m2 + B*np.abs(h)*r2 - np.abs(1-2*eps+h*m2)/2
    psi = (1 - h*h)*np.exp(-beta*(n1*Vmf1 - n2*J*Vmf2))
    return psi

def Psi2(h, m1, r1, m2, r2, n1, n2, rho, eps, J, beta):
    A = (1 - rho)/2
    B = rho/2
    Vmf1 = -A*h*m1 + B*np.abs(h)*r1 - np.abs(1-2*eps+h*m1)/2
    Vmf2 = -A*h*m2 + B*np.abs(h)*r2 - np.abs(1-2*eps+h*m2)/2
    psi = (1 - h*h)*np.exp(-beta*(n2*Vmf2 - n1*J*Vmf1))
    return psi

def Z1(*args):
    z = quad(Psi1, -1, 1, args=args)[0]
    return z + 1e-8

def Z2(*args):
    z = quad(Psi2, -1, 1, args=args)[0]
    return z + 1e-8

def m1(z, *args):
    f = lambda x: x*Psi1(x, *args)
    m = quad(f, -1, 1)[0]/z
    return m

def m2(z, *args):
    f = lambda x: x*Psi2(x, *args)
    m = quad(f, -1, 1)[0]/z
    return m

def r1(z, *args):
    f = lambda x: np.abs(x)*Psi1(x,*args)
    r = quad(f, -1, 1)[0]/z
    return r

def r2(z, *args):
    f = lambda x: np.abs(x)*Psi2(x,*args)
    r = quad(f, -1, 1)[0]/z
    return r

def sce(m10, rho, eps, J, beta):
    m20 = -m10
    r10 = np.abs(m10)
    r20 = r10
    n1 = .5
    n2 = 1 - n1
    M1 = m10
    R1 = r10
    M2 = m20
    R2 = r20
    delta = .1
    # g = np.sqrt(1-rho**2)/rho
    # eps = (-g + g*np.log(2*eps)/2)/30
    precision = 1e-6
    converged = False
    n = 0
    while (not converged) and n < 10000:
        M10 = M1
        R10 = R1
        M20 = M2
        R20 = R2
        args = M10, R10, M20, R20, n1, n2, rho, eps, J, beta
        z1 = Z1(*args)
        z2 = Z2(*args)
        M1 = (1 - delta)*M10 + delta*m1(z1, *args)
        R1 = (1 - delta)*R10 + delta*r1(z2, *args)
        M2 = (1 - delta)*M20 + delta*m2(z1, *args)
        R2 = (1 - delta)*R20 + delta*r2(z2, *args)
        fixed1 = (np.abs(M1 - M10)<precision or np.abs(R1 - R10)<precision)
        fixed2 = (np.abs(M2 - M20)<precision or np.abs(R2 - R20)<precision)
        if fixed1 and fixed2:
            converged = True
        n += 1

    point = (m10, J, eps, rho, beta)
    point_names = "m10 J eps rho beta".split(" ")
    idx = "m1 r1 m2 r2 m_s m converged n_iter".split(" ")
    result = [M1, R1, M2, R2, (M1-M2)/2, (M1+M2)/2, converged, n]
    s = pd.Series(result,index=idx)
    df = pd.DataFrame({point:s}).T
    df.index.set_names(point_names, inplace=True)
    data = {"self-consistent-solution":df}
    return data
#---------

if __name__ == "__main__":

    args = read_args()

    input_ = args.filename
    label = args.label
    config = json.load(input_)
    config["label"] = label
    input_.close()

    m10_grid =   np.array([.5])
    config["m10_values"] = m10_grid.tolist()

    rho_grid =   np.arange(*config["rho_slice"])
    J_grid = np.arange(*config["J_slice"])
    eps_grid =   np.arange(*config["eps_slice"])
    beta_grid = np.hstack([np.linspace(0,2,100),
                           np.linspace(2,17,200),
                           np.linspace(17,30,100)])
    config["beta_values"] = beta_grid.tolist()
    grid = np.meshgrid(
        m10_grid,
        rho_grid,
        eps_grid,
        J_grid,
        beta_grid
    )
    points = np.vstack([x.ravel() for x in grid]).T
    parameters = enumerate(points)

    def run(x):
        idx, prmt = x
        res = sce(*prmt)
        return idx, res

    pool = mp.Pool()
    t0 = time.time()
    print(40*"=")
    print("Passed:", json.dumps(config, indent=4), sep="\n")
    print("Starting at: ", time.asctime())
    results = pool.map(run, parameters)
    pool.close()
    pool.join()
    results.sort()
    results =  tuple([res[-1] for res in results])
    t = time.time()
    print("pool.map(run, args) took %s"%(timedelta(seconds=(t-t0))))

    path = config["save_dir"] + label + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path+"config.json","w") as file_:
        json.dump(config, file_, indent=4)

    t0_save = time.time()
    save_data(path, results)
    t_save = time.time()
    print("save_data took %s" % (timedelta(seconds=(t_save-t0_save))))

    t_final = time.time()
    print("Total time spent: %s"%(timedelta(seconds=(t_save-t0))))
    print("Finished at: ", time.asctime())
    print(40*"=")

