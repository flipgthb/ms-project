
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import matplotlib as mpl
import brewer2mpl


def graph_style():

    # Getting brewer Set1 colors
    #brewer2mpl.print_all_maps()
    set1 = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors
    # rdbu = brewer2mpl.get_map("RdBu", "Diverging", 8).mpl_colors
    # greys = brewer2mpl.get_map("Greys", "Sequential", 8).mpl_colors

    # Setting some general parameter for plots
    mpl.rcParams["figure.figsize"] = (10, 7)
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['axes.color_cycle'] = set1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['patch.edgecolor'] = 'white'
    mpl.rcParams['patch.facecolor'] = set1[0]
    mpl.rcParams['font.family'] = 'StixGeneral'

# Function to remove spines and ticks
def remove_border(axes=None, top=False, right=False,
                  left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis
    ticks

    The top/right/left/bottom keywords toggle whether the corresponding
    plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

def get_panel(path, header=[0,1,2], index_col=[0,1]):
    with open(path) as file_:
        dataframe = pd.read_csv(file_, header=header, index_col=index_col)
        panel = dataframe.to_panel()
        lvl = list(panel.items.levels)
        new_lvl = map(lambda x: x.astype(float), lvl)
        panel.items.set_levels(new_lvl, inplace=True)
    return panel
