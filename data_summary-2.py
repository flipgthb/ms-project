#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    dir_ = "/media/backup/simulation-data/ms-project/data/type-1/"
    sim = "simulation-%d/"
    path = dir_ + sim
    n = 70

    x = path%n
    statistics = pd.read_csv(x+"statistics.csv")
    cumulative = pd.read_csv(x+"cumulative.csv")
    social_network = np.load(x+"social-network.npy")
    state = np.load(x+"state.npy")
    zeitgeist = np.load(x+"zeitgeist.npy")

    for n in xrange(1,n):
        x = path%n
        statistics += pd.read_csv(x+"statistics.csv")
        cumulative += pd.read_csv(x+"cumulative.csv")
        social_network += np.load(x+"social-network.npy")
        state += np.load(x+"state.npy")
        zeitgeist += np.load(x+"zeitgeist.npy")

    statistics /= n
    cumulative /= n
    social_network /= n
    state /= n
    zeitgeist /= n
    SAVE_DIR = dir_+"simulation-summary/"
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    README = """ Summary Statistics from 20 simulations -
    Changing w, reputation and zeitgeist:
        rep(t+1) = rep(t) + sign*d_rep
        sign = sign(h_i * h_j)
        h_i = w_a.dot(z_i)
        z = rep.dot(w)
        w_i(t+1) = w_i(t) + d_V_ij * z_j * sign(h_j)
    """
    explain = statistics.keys()

    with open(SAVE_DIR+'statistics.csv', 'w') as file_:
        df = pd.DataFrame(statistics, columns=explain)
        df.to_csv(file_, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'cumulative.csv', 'w') as file_:
        df = pd.DataFrame(cumulative, columns=explain)
        df.to_csv(file_, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'social-network.npy', 'w') as file_:
        np.save(file_, social_network)

    with open(SAVE_DIR+'state.npy', 'w') as file_:
        np.save(file_, state)

    with open(SAVE_DIR+'zeitgeist.npy', 'w') as file_:
        np.save(file_, zeitgeist)

    with open(SAVE_DIR+'README.txt', 'w') as file_:
        file_.write(README)
