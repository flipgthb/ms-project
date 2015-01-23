
import numpy as np


def make_weight_matrix(sigma, n):
    i, j = np.indices((n,n))
    w = np.exp(-(i - j) ** 2. / sigma / n)
    w /= w.sum(axis=1)[:,None]
    w /= w.sum(axis=0)
    w = 0.5 * (w + w.T)
    return w

def neighborhood_sort(sim_matrix, sigma, weight_matrix=None):
    n = sim_matrix.shape[0]
    weight_matrix = make_weight_matrix(sigma, n)

    mismatch = np.dot(sim_matrix, weight_matrix)

    idx_m = mismatch.argmax(axis=1)
    val_m = mismatch.max(axis=1)

    mx = val_m.max()
    sort_scr = 1 + idx_m - np.sign(idx_m - n/2)*val_m/mx
    sorted_ind = np.argsort(sort_scr)
    erg = np.trace(
        np.dot(
            sim_matrix[np.meshgrid(sorted_ind, sorted_ind)],
            weight_matrix
        )
    )
    return sorted_ind, erg

def spin(sim_matrix, shuffle=False, n=60, m=20, s=1.01):
    n = sim_matrix.shape[0]
    si = range(n)
    if shuffle:
        np.random.shuffle(si)
    sigma = 2 ** 5.
    for j in range(n):
        w = make_weight_matrix(sigma, n)
        out = []
        for i in range(m):
            sim_matrix = sim_matrix[np.meshgrid(si, si)]
            si_last = np.array(si).copy()
            si, e = neighborhood_sort(sim_matrix, sigma, w)
            if e in out:
                #print("break")
                break
            else:
                out.append(e)
        sigma = sigma / s
    return sim_matrix, si_last
