#!/usr/bin/env python
# encoding: utf-8
from __future__ import division, print_function
import pandas as pd
import math
import random
import time
import datetime
from Model import ModelMeta

__all__ = ["mcmc", "save_data"]

def mcmc(model, sweeps, save_state=False):
    data = pd.DataFrame()
    accepted = 0.0
    
    N = model.size
    total_steps = sweeps*N
    for t in xrange(total_steps):
        proposition = model.propose()
        if proposition is None:
            continue
        logp = model.logp(proposition)
        acceptance = min(1, math.exp(logp))
        rejection = random.random()
        if rejection < acceptance:
            model.state = proposition
            accepted += 1
            
        if t % N == 0:
            measured = model.measure()
            measured["acceptance_ratio"] = accepted / (t+1)
            data = data.append(measured, ignore_index=True)
    
    p,names = zip(model.parameters.items())
    stats = pd.Panel({p:data})
    result = {"statistics":stats}
    if save_state:
        s = model.state
        final_state = {k:pd.Panel({p:v}) for k,v in s.items()}
        result.update(final_state)
    for k,v in result.items():
        v.items.set_names(names, inplace=True)
    return result

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