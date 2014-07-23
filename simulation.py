#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
from program import program
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os
import json
import multiprocessing as mp
import time
from datetime import timedelta
from utility import read_args, save_data


if __name__ == "__main__":

    args = read_args()

    input_ = args.filename
    label = args.label
    config = json.load(input_)
    config["label"] = label
    input_.close()

    beta_grid =  slice(*config["bounds"]["beta"])
    delta_grid = slice(*config["bounds"]["delta"])
    tau_grid = slice(*config["bounds"]["tau"])

    grid = np.mgrid[tau_grid, delta_grid, beta_grid]
    points = np.vstack([x.ravel() for x in grid]).T
    order = np.arange(points.shape[0])[:,np.newaxis]
    args = np.hstack([order, points])

    def run(x):
        idx, tau, delta, beta = x
        r = program(tau, delta, beta, config["society"], config["mcmc"])
        return idx, (tau, delta, beta), r

    # result = map(run,args[:3])
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
