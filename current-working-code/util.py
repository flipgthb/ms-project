#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
from argparse import ArgumentParser
import os
import json
import multiprocessing as mp
import time
from datetime import timedelta
import math

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
