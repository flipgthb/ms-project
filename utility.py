#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import networkx as nx
import pandas as pd
from argparse import ArgumentParser
import os
import json


def row_norm(X):
    nX = np.sqrt((X*X).sum(axis=1)[:, np.newaxis])
    return nX

def prob_row_norm(X):
    nX = X.sum(axis=1)[:, np.newaxis]
    return nX

def create_network(top_dict):
        params = top_dict["parameters"]
        type_ = top_dict["type"]
        net_dict = dict(
            complete=nx.complete_graph,
            barabasi_albert=nx.barabasi_albert_graph,
            watts_strogatz=nx.watts_strogatz_graph
        )
        G = net_dict[type_](**params)
        M = np.asarray(nx.adjacency_matrix(G))
        return M

def create_initial_state(N, D, type_="disordered"):
    W = np.ones((N,D))
    if type_ is "ordered":
        W = np.ones((N, D))
    elif type_ is "disordered":
        W = np.random.randn(N, D)
    W /= row_norm(W)
    return W

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
       parser.error("The file %s does not exist!"%arg)
    else:
       return open(arg,'r')  #return an open file handle

def read_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-i", "--input_file", dest="filename", required=True,
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

def save_data(index_data_list, config):

    f = lambda x: x[1:]
    x = map(f, index_data_list)
    stat_list = [(y[0],y[1][0]) for y in x]
    sn_list = [(y[0],y[1][1]) for y in x]
    rep_list = [(y[0],y[1][2]) for y in x]
    act_list = [(y[0],y[1][3]) for y in x]
    w_list = [(y[0],y[1][4]) for y in x]
    statistics = dict(stat_list)
    social_network = dict(sn_list)
    reputation = dict(rep_list)
    activity = dict(act_list)
    w = dict(w_list)
    label = ["gamma", "delta", "beta"]
    stat_panel = pd.Panel(statistics)
    stat_panel.items.set_names(label, inplace=True)
    sn_panel = pd.Panel(social_network)
    sn_panel.items.set_names(label, inplace=True)
    rep_panel = pd.Panel(reputation)
    rep_panel.items.set_names(label, inplace=True)
    act_panel = pd.Panel(activity)
    act_panel.items.set_names(label, inplace=True)
    w_panel = pd.Panel(w)
    w_panel.items.set_names(label, inplace=True)

    SAVE_DIR = config["save_directory"] + config["label"]
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    with open(SAVE_DIR+'/config.json', 'w') as file_:
        json.dump(config, file_, indent=4)

    with open(SAVE_DIR+'/statistics.csv', 'w') as file_:
        df = stat_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/social_network.csv', 'w') as file_:
        df = sn_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/reputation.csv', 'w') as file_:
        df = rep_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/activity.csv', 'w') as file_:
        df = act_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)

    with open(SAVE_DIR+'/state.csv', 'w') as file_:
        df = w_panel.to_frame()
        df.to_csv(file_)#, mode='a', index=False, index_label=False)
