#!/usr/bin/env python
# encoding: utf-8

# -----------------------------------------------------------------------------
# ---------------------------------- Modules ----------------------------------
# ---------------------------------------------------------------------------
# importing from __future__ module provides compatibility with
# python 3, where the division operator '/' is float division,
# and print is a function instead of a keyword
from __future__ import division, print_function

import numpy as np  # provides arrays and appropriate math for it

import networkx as nx  # provides graphs and methods on them

import pandas as pd  # provides Panel and DataFrame, useful to manage data

from scipy.special import erfc, erf  # some special functions from scipy

import math  # standard python math module

from argparse import ArgumentParser  # useful to parse command line arguments

import os  # provides access to file path and othe OS utilities

import json  # provides an API to use json files

import multiprocessing as mp  # API to parallel computing

import time  # access to clock and other time related functions

from datetime import timedelta  # compute and format difference in time objects

# -----------------------------------------------------------------------------
# ---------------------------- Utility Functions ------------------------------
# -----------------------------------------------------------------------------
sqrt2 = math.sqrt(2)

def H(x):
    """
    The function H(x) defined as:

                1     (oo                 1      (   x   )
     H(x) = --------- |dt exp(-t*t/2) = -----erfc|-------|
            sqrt(2pi) )x                  2      ( sqrt2 )

    using the scipy.special.erfc

    """
    return erfc(x/sqrt2)/2

def zip_dict_items(a, b):
    """
    Stack the values of two dicts with the same keys.
    -------
    example
    -------
        a = {'foo':1}
        b = {'foo':2}
        zip_dict_items(a,b)
        >>> {'foo':array([1,2])}
    """
    for k in a.keys():
        try:
            a[k] = np.hstack([a[k], b[k]])
        except KeyError:
            raise Exception("dicts must have the same keys")

def is_valid_file(parser, arg):
    """
    Check for the existence of file arg and returns the file object if it does.
    """
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
       return open(arg,'r')  #return an open file handle

def read_args():
    """
    Parse command line options.

    There are two possible options:
        -i or --input to specify the input file containing the simulation
        parameters

        label as a positional argument used to name the simulation results
        saved in the disk

    To use the program with open the shell and run:
        python monte_carlo.py -i 'parameters.json' 'simulation_name'

    and make sure to have a paramters file.
    """
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

def save_data(idx_data_list, config,
              label=["epsilon", "rho", "beta"]):
    """
    Format the simulation data and save it to the disk.

    idx_data_list is a list of tuple with the form:
        (simulation_index, result_dict)

    which is supposed to be sorted. save_data runs through
    this list, ignoring the simulation_index and creating dicts of
    DataFrames. Than it convert them to Panels, set the items names
    with `label` and save in the appropriate directory with suitable
    file names.
    """
    d0 = idx_data_list.pop(0)[-1]  # pop(i) remove and returns item i in a list
    stat_dict = d0["stat"]
    rep_dict = d0["rep"]
    agt_dict = d0["agt"]
    for entry in idx_data_list:
        d = entry[-1]  # get the last item in entry
        stat_dict.update(d["stat"])  # update adds a pair key, value to dict or
        rep_dict.update(d["rep"])    # replaces the value for a existent key
        agt_dict.update(d["agt"])
    stat_panel = pd.Panel(stat_dict)  # a Panel is a special 3 dimendional table
    rep_panel = pd.Panel(rep_dict)
    agt_panel = pd.Panel(agt_dict)
    stat_panel.items.set_names(label, inplace=True)
    rep_panel.items.set_names(label, inplace=True)
    agt_panel.items.set_names(label, inplace=True)

    # check the saving path and create the file directory
    SAVE_DIR = config["save_directory"] + config["label"]
    if os.path.exists(config["save_directory"]):
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
    else:
        raise Exception("%s is not a valid path"%config["save_directory"])

    # save a the configuration used in simulation
    with open(SAVE_DIR+'/config.json', 'w') as file_:
        json.dump(config, file_, indent=4)

    # table with stattics: m, n_op and r
    with open(SAVE_DIR+'/statistics.csv', 'w') as file_:
        df = stat_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    # table with the final reputantion matrix
    with open(SAVE_DIR+'/reputation.csv', 'w') as file_:
        df = rep_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    # table with the final agent coupling vectors
    with open(SAVE_DIR+'/state.csv', 'w') as file_:
        df = agt_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)


# -----------------------------------------------------------------------------
# ---------------------------- Monte Carlo ------------------------------------
# -----------------------------------------------------------------------------
def mc(rho, beta, eps, config):
    """
    Runs a Markov Chain Monte Carlo for a Bayesian Agent Society
    with liberal index `rho`, at social pressure `beta` and
    expected noise `eps`. The `config` parameter must be a dict with
    format:
        {
            'N': [int,int,int],
            'D': int,
            'dw': float,
            'sweeps': int,
            'beta_parties': [float,float,float],
            'rho_parties': [float,float,float]
        }

    The energy function is:
        E(x) = - G*G * ln(eps - (1-2*eps)*H(-x/G/Q))
    where G = sqrt(1-rho*rho)/rho

    """
    n1, n2, n3 = config['n']
    N = n1 + n2 + n3
    D = config['D']
    dw = config['dw']
    sweeps = config['sweeps']

    # political pressures for each party
    beta1, beta2, beta3 = config['beta']
    # liberal index for each party
    rho1, rho2, rho3 = config['rho']

    sqrtD = math.sqrt(D)
    norm = sqrtD
    Q = norm/sqrtD
    gamma_func = np.vectorize(lambda u: math.sqrt(1 - u*u)/u)

    np.random.seed(78956347)
    # sampling a N x D matrix with entries distributed by a Normal
    # with zero mean and unity variance.
    w1 = .1*np.random.randn(n1, D) + .75
    w2 = .1*np.random.randn(n2, D) - .75
    w3 = np.random.randn(n3, D)
    w = np.vstack([w1,w2,w3])
    # normalize each line i so that w[i].dot(w[i]) = norm*norm.
    # the lines are the agents cognitive vectors
    w *= norm/np.sqrt((w*w).sum(axis=1))[:,None]

    # the zeitgeist vector
    z = np.ones(D)
    # with the same normalization as the agents vectors
    z *= norm/np.linalg.norm(z)

    # generating the social network
    A = np.ones((N,N)) - np.identity(N)
    # G = nx.barabasi_albert_graph(N, 20)
    # A = np.asarray(nx.adjacency_matrix(G))

    Beta = np.array(n1*[beta1] + n2*[beta2] + n3*[beta3])
    Rho = np.array(n1*[rho1] + n2*[rho2] + n3*[rho3])
    Gamma = gamma_func(Rho)

    # this function is used to compute the order paramters
    # through the simulation
    def measure():
        h = z.dot(w.T)/norm/norm
        m = h.mean()
        hp = h[h>0]
        hn = h[h<0]
        np, nn = hp.shape[0], hn.shape[0]
        ms = (np*hp.mean() - nn*hn.mean())/(np+nn)
        q = (A*h*h[:,None]).mean()
        h1 = h[:n1]
        h2 = h[n1:n1+n2]
        h3 = h[n1+n2:]
        m1 = h1.mean()
        m2 = h2.mean()
        m3 = h3.mean()
        q1 = (h1*h1[:,None]).mean()
        q2 = (h2*h2[:,None]).mean()
        q3 = (h3*h3[:,None]).mean()
        q13 = (h1*h3[:,None]).mean()
        q23 = (h2*h3[:,None]).mean()
        qs = (q13 - q23)/2

        trace = dict(
            m=m,
            q=q,
            m_1=m1,
            m_2=m2,
            m_3=m3,
            m_s=ms,
            q_1=q1,
            q_2=q2,
            q_3=q3,
            q_s=qs
        )
        return trace

    # initial measurement
    trace = measure()

    # agent interaction potential
    def energy(hi, hj, g):
        X = hi*np.sign(hj)/g/Q
        Ep = -g*g*np.log(eps + (1-2*eps)*erfc(-X/sqrt2)/2)
        return Ep

    # main MC loop 1st gov
    for t in xrange(2*sweeps*N):
        # pick an agent uniformly in {1,..,N}
        i = np.random.choice(N)
        # compute the probabilities of agent i to interact with each of
        # his neighbors throug the reputation matrix
        x = A[i]
        pij = x/x.sum()
        # and pick a neighbor with the computed probability
        try:
            j = np.random.choice(N, p=pij)
        except:
            continue

        # computing the local field for agents i and j
        h = z.dot(w.T)/sqrtD
        hi, hj = h[i], h[j]

        # MCMC step
        E0 = energy(hi, hj, Gamma[i])  # energy before proposal
        nw = dw*np.random.randn(D) + w[i]  # proposal for agent i
        nw *= norm/np.linalg.norm(nw)
        nhi = z.dot(nw)/sqrtD  # proposed local field for i
        E = energy(nhi, hj, Gamma[i])  #  proposed energy
        # if energy lowers with proposal, accept it. if not, accept
        # with probability < exp(-beta*(energy difference))
        if E < E0 or np.random.rand() < math.exp(Beta[j]*(E0-E)):
            w[i] = nw.copy()

        # at each sweep, get data.
        if t%N == 0:
            more_data = measure()
            zip_dict_items(trace, more_data)

        # election happens. changing who's in power
        if t == N*sweeps:
            Beta = np.array(n1*[beta1] + n2*[beta3] + n3*[beta2])

    # formating the acquired data:
    # result is dict of dicts
    # each dict inside result has the form
    #   {(eps, rho, beta): Dataframe}
    # and the Dataframes are special tables that make accessing,
    # reading and saving the data easier.
    dfs = pd.DataFrame(trace)  #
    dfA = pd.DataFrame(A)
    dfw = pd.DataFrame(w)
    p = (eps, -1, -1)
    stat_dict = {p:dfs}
    A_dict = {p:dfA}
    w_dict = {p:dfw}
    return dict(stat=stat_dict, rep=A_dict, agt=w_dict)


# -----------------------------------------------------------------------------
# ---------------------------- Main Program -----------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # using the read_args function to parse command line options into args
    args = read_args()

    input_ = args.filename
    label = args.label
    # this object will be passed through various points of the program and
    # containd almost all information about the simulation, except the
    # name and code of the program that uses it.
    config = json.load(input_)
    config["label"] = label
    config["zeitgeist"] = [1.0]*config["D"]
    input_.close()

    # slices are object to index lists in fancy way.
    # basically, a slice is something like [start:stop:step],
    # but this syntax is only valid inside lists brackets, so
    # to specify a slice outside lists brackes, use the slice
    # function like slice(start,stop,step). below, its use is
    # somewhat obscured by the notation slice(*[a,b,c]) = slice(a,b,c).
    beta_grid =  slice(0,1,1)
    rho_grid = slice(0,1,1)
    eps_grid = slice(*config["epsilon"])
    # the grid contain the control paramter space points,
    # but in saparate matrices
    grid = np.mgrid[eps_grid, rho_grid, beta_grid]
    # its necessary to flatten the matrics and
    # stack them so they for a list of unique triples of points
    points = np.vstack([x.ravel() for x in grid]).T
    # also, its useful to stack this initial order of triple,
    # since the parallel asyncronous computation used below
    # does not keeps the order
    order = np.arange(points.shape[0])[:,np.newaxis]
    args = np.hstack([order, points])

    # this finction runs a mc function above
    # for a given set of paramters and return its results
    # it is used to map the control paramters space with the
    # mc function through the multiprocessing module to run it
    # in parallel
    def run(x):
        idx, eps, rho, beta = x
        r = mc(rho, beta, eps, config)
        return idx, r

    # creating the core pool
    # result = map(run, args[:3])
    pool = mp.Pool()
    t0 = time.time() # intial time (in computer units)
    print(40*"=")
    print("Passed:", json.dumps(config, indent=4), sep="\n")
    print("Starting at: ", time.asctime())
    # here, the simulation begins
    result = pool.map(run, args)
    pool.close()
    pool.join() # here it ends
    # reordering the results so the control paramters have the initial order
    result.sort()
    t = time.time()  # final time (in computer units)
    # timedelta formats the diference t-t0 so its displayed nicely
    print("pool.map(run, args) took %s"%(timedelta(seconds=(t-t0))))

    t0_save = time.time()  # just to compare the saving time to simulation time
    save_data(result, config)
    t_save = time.time()
    print("save_data took %s"%(timedelta(seconds=(t_save-t0_save))))
    t_final = time.time()
    print("Total time spent: %s"%(timedelta(seconds=(t_save-t0))))
    print("Finished at: ", time.asctime())
    print(40*"=")
