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

_D_ = 5
_N_ = 64
_k_ = 20
_P_ = 1
_p_= 1/2
_BETA_ = (0,50,.5)
_DELTA_ = (0,1.2,0.2)
_GAMMA_ = (0.1, 1.1, 0.8)
_NUM_STEPS_ = 6000
_BURN_ = 5000
_MEASURE_ = 1
_D_OMEGA_ = 1.0
_D_EPS_ = 0.1
_DRIFT_ = 1.0

def read_args():
    parser = argparse.ArgumentParser()
    # usage purpose ==========================================================
    parser.add_argument('-v','--verbosity',
                        help='verbosity flag',
                        action='store_true')

    parser.add_argument('alias',
                        help='string: simulation alias, the name'+\
                        ' used to save its results to the disk.')

    parser.add_argument('--explanation',
                        help='breif explanation about simulation')

    # society relative =======================================================
    parser.add_argument('-D', '--agent_complexity',
                        default=_D_,
                        help='int: agent moral vector dimension. defaults'+\
                        ' to %d'%_D_,
                        type=int)

    parser.add_argument('-N', '--num_agents',
                        default=_N_,
                        help='int: number of agents. default is %d'%_N_,
                        type=int)

    parser.add_argument('-k', '--num_neighbors',
                        default=_k_,
                        help='int: number of neighbors of each agents. '+\
                        'default is %d'%_k_,
                        type=int)

    parser.add_argument('-p', '--rewire_probability',
                        default=_p_,
                        help='float: rewire probability to watts-strogatz '+\
                        'model. default is %.2f'%_p_,
                        type=float)

    parser.add_argument('--beta', nargs=3,
                        default=_BETA_,
                        help='tuple of floats: inverse temperature.'+\
                        'default is {}'.format(_BETA_),
                        type=float)

    parser.add_argument('--delta', nargs=3,
                        default=_DELTA_,
                        help='tuple of floats: cognitivel style. default is'+\
                        ' {}'.format(_DELTA_),
                        type=float)

    parser.add_argument('--gamma', nargs=3,
                        default=_GAMMA_,
                        help='tuple of floats: bullshit threshold. default is'+\
                        ' {}'.format(_GAMMA_),
                        type=float)

    # mcmc relative ==========================================================
    parser.add_argument('--d_omega',
                        default=_D_OMEGA_,
                        help='float: proposal step size.'+\
                        'default is %.2f'%_D_OMEGA_,
                        type=float)

    parser.add_argument('--d_eps',
                        default=_D_EPS_,
                        help='float: reputation step size. default '+\
                        'is %.2f'%_D_EPS_,
                        type=float)

    parser.add_argument('--drift',
                        default=_DRIFT_,
                        help='float: zeitgeist drift factor. default '+\
                        'is %.2f'%_DRIFT_,
                        type=float)

    parser.add_argument('-n','--num_steps',
                        default=_NUM_STEPS_,
                        help='int: number of iterations.'+\
                        'defaults to %d'%_NUM_STEPS_,
                        type=int)

    parser.add_argument('-b', '--burn',
                        default=_BURN_,
                        help='int: burned iterations. defaults to %d'%_BURN_,
                        type=int)

    parser.add_argument('-m', '--measure_period',
                        default=_MEASURE_,
                        help='int: measurement period.'+\
                        'defaults to %d'%_MEASURE_,
                        type=int)

    args = parser.parse_args()
    return args


def concatenate_dicts(*dicts):
    keys = set().union(*dicts)
    c = {}
    for k in keys:
        x = []
        for d in dicts:
            x.append(d.get(k, ''))
        c[k] = np.array(x)
    return c


def save_data(data_list, config):

    x = np.vstack(data_list)[:,1:].flat
    explain = x[0]["explain"]
    data = concatenate_dicts(*x)
    for k, v in data.items():
        data[k] = np.vstack(v)

    SAVE_DIR = "/media/backup/simulation-data/ms-project/data/%s"%config["name"]
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    with open(SAVE_DIR+'/config.json', 'w') as file_:
        json.dump(config, file_, indent=4)

    with open(SAVE_DIR+'/statistics.csv', 'w') as file_:
        df = pd.DataFrame(data["statistics"], columns=explain)
        df.to_csv(file_, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/cumulative.csv', 'w') as file_:
        df = pd.DataFrame(data["cumulative"], columns=explain)
        df.to_csv(file_, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/social-network.npy', 'w') as file_:
        np.save(file_, data["social_network"])

    with open(SAVE_DIR+'/activity.npy', 'w') as file_:
        np.save(file_, data["activity"])

    with open(SAVE_DIR+'/initial_social_network.npy', 'w') as file_:
        np.save(file_, data["initial_social_network"])

    with open(SAVE_DIR+'/state.npy', 'w') as file_:
        np.save(file_, data["state"])

    with open(SAVE_DIR+'/zeitgeist.npy', 'w') as file_:
        np.save(file_, data["zeitgeist"])

    if 'readme' in config:
        with open(SAVE_DIR+'/README.txt', 'w') as file_:
            file_.write(config["readme"])


if __name__ == "__main__":

    args = read_args()

    s_conf = dict(N=args.num_agents,
                  D=args.agent_complexity,
                  k=args.num_neighbors)

    mc_conf = dict(num_steps=args.num_steps,
                   burn=args.burn,
                   measure_period=args.measure_period,
                   d_omega=args.d_omega,
                   d_eps=args.d_eps)

    bounds = dict(beta=args.beta,
                  delta=args.delta,
                  gamma=args.gamma)

    config = dict(society=s_conf,
                  mcmc=mc_conf,
                  name=args.alias,
                  bounds=bounds)

    if 'explanation' in args:
        config['readme'] = args.explanation

    beta_grid = slice(*args.beta)
    delta_grid = slice(*args.delta)
    gamma_grid = slice(*args.gamma)

    grid = np.mgrid[gamma_grid, delta_grid, beta_grid]
    # grid = np.mgrid[delta_grid, beta_grid]
    points = np.vstack([x.ravel() for x in grid]).T
    order = np.arange(points.shape[0])[:,np.newaxis]
    args = np.hstack([order, points])

    def run(x):
        idx, gamma, delta, beta = x
        # idx, delta, beta = x
        S = Society(delta=delta, beta=beta, gamma=gamma, **config["society"])
        # S = Society(delta=delta, beta=beta, gamma=0.1, **config["society"])
        mc = MCMC(system=S, **config["mcmc"])
        mc.sample()
        r = mc.data
        return idx, r

    # map(run,args)
    pool = mp.Pool()
    t0 = time.time()
    print(40*"=")
    print("Passed:", json.dumps(config, indent=4), sep="\n")
    print("Starting at: ", time.asctime())
    result = pool.map(run, args)
    pool.close()
    pool.join()
    t = time.time()
    print("Finished at: ", time.asctime())
    print("simulation took %s"%(timedelta(seconds=(t-t0))))

    result.sort()
    save_data(result, config)
    print(40*"=")
