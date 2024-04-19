import numpy as np


def from_dirac_to_basis(vec, d):  # |00> -> [1,0,...,0] -> len() == other_size**2
    if isinstance(d, int):
        d = [d] * len(vec)

    basis_vecs = []
    for i, basis in enumerate(vec):
        temp = [0] * d[i]
        temp[basis] = 1
        basis_vecs.append(temp)

    ret = basis_vecs[0]
    for e_i in range(1, len(basis_vecs)):
        ret = np.kron(np.array(ret), np.array(basis_vecs[e_i]))

    return ret


def calculate_q0_q1(lev, dim):
    q1 = lev % dim
    q0 = (lev - q1) // dim

    return q0, q1


def insert_at(big_arr, pos, to_insert_arr):
    """Quite a forceful way of embedding a parameters into big_arr"""
    x1 = pos[0]
    y1 = pos[1]
    x2 = x1 + to_insert_arr.shape[0]
    y2 = y1 + to_insert_arr.shape[1]

    assert x2 <= big_arr.shape[0], "the position will make the small parameters exceed the boundaries at x"
    assert y2 <= big_arr.shape[1], "the position will make the small parameters exceed the boundaries at y"

    big_arr[x1:x2, y1:y2] = to_insert_arr

    return big_arr
