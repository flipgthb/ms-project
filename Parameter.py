#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
import json


class Parameter(object):
    def __init__(self, type_, value, name=None):
        self.name = name
        self._type_ = type_
        self._specs = value

    @property
    def slice_(self):
        if self._type_ is "point":
            p0 = self._specs
            s = slice(p0, p0, 1j)
            return s
        elif self._type_ is "grid":
            p0, p, n = self._specs
            n *= 1j
            s = slice(p0, p, n)
            return s

    @property
    def value(self):
        return np.mgrid[self.slice_]

    def __repr__(self):
        y = self.value if self._type_ is "point" else self._specs
        x = dict(zip(("p0", "p", "n"), y))
        this = {'type_':self._type_,
                'value': x}
        if self.name:
            this['name'] = self.name
        rp = json.dumps(this, indent=4, sort_keys=True)
        return rp

    def __str__(self):
        return self.__repr__()


def make_psg(*parameters):
    sl = [x.slice_ for x in parameters]
    psg = np.mgrid[sl]
    flatit = [x.ravel() for x in psg]
    names = [x.name for x in sl]
    psp = np.vstack(flatit).T
    return psp, names

if __name__ == "__main__":
    x = Parameter('point', np.pi)
    y = Parameter('grid', (0,1,10), name='rho')
    z = Parameter('grid', (0,20,5), name='gamma')
    print(x)
    print(y)
    print(z)

    zr = z.__repr__()
    wr = json.loads(zr)
    w = Parameter(**wr)
    print(w)

    ps = make_psg(x,y,z)
    print(ps)


