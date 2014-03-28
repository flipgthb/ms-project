#!/usr/bin/env python
# encoding: utf-8

from subprocess import call

if __name__ == "__main__":

    prog = "./phase_space_simulation.py"
    explanation = """changing: w, reputation, zeitgeist:
    rep(t+1) = rep(t) + sign*d_rep
    sign = sign(h_i * h_j)
    h_i = w_a.dot(z_i)
    z = rep.dot(w)
    w_i(t+1) = w_i(t) + d_V_ij * z_j * sign(h_j)

    This simulation is part of a series of equivalent simulations to be used
    to get statistical properties lost in long Monte Carlo runs. This is
    strategy of redundancy may be seen as a mean over fixed time ensambles.
    """
    args = [
        [prog, "simulation-%d"%i, "--explanation", explanation]
        for i in xrange(21,71)
    ]
    for a in args:
        call(a)
    call("./data_summary-1.py")
