"""
Various function to manipulate GL strings.
"""

import numpy as np
from numba import jit


@jit(cache=True)
def compute_winding(u, v):
    """
    Compute the winding number for a field in each frame of a GL string.
    """

    F, N, _ = u.shape
    w = np.zeros(F)

    for n in range(F):

        for i in range(N - 1):
            w[n] += (
                +u[n, i, 0] * (v[n, i + 1, 0] - v[n, i, 0])
                - v[n, i, 0] * (u[n, i + 1, 0] - u[n, i, 0])
            ) / (u[n, i, 0] ** 2 + v[n, i, 0] ** 2)

            w[n] += (
                +u[n, -1, i] * (v[n, -1, i + 1] - v[n, -1, i])
                - v[n, -1, i] * (u[n, -1, i + 1] - u[n, -1, i])
            ) / (u[n, -1, i] ** 2 + v[n, -1, i] ** 2)

            w[n] += (
                +u[n, -1 - i, -1] * (v[n, -2 - i, -1] - v[n, -1 - i, -1])
                - v[n, -1 - i, -1] * (u[n, -2 - i, -1] - u[n, -1 - i, -1])
            ) / (u[n, -1 - i, -1] ** 2 + v[n, -1 - i, -1] ** 2)

            w[n] += (
                +u[n, 0, -1 - i] * (v[n, 0, -2 - i] - v[n, 0, -1 - i])
                - v[n, 0, -1 - i] * (u[n, 0, -2 - i] - u[n, 0, -1 - i])
            ) / (u[n, 0, -1 - i] ** 2 + v[n, 0, -1 - i] ** 2)

    return w


@jit(cache=True)
def int2bit(x, n):
    """
    Integer to bit function.
    """
    return (x >> n) & 1


@jit(cache=True)
def build_nn_map(sc_domain):
    """
    Given a domain definition build the nearest neighbour map.
    """

    N, M = sc_domain.shape

    nn_map = np.zeros((N, M))

    for i, j in np.ndindex(N, M):

        nn = 0

        if sc_domain[i, j] != 0:
            if i != N - 1:
                if sc_domain[i + 1, j] != 0:
                    nn += 1 << 0

            if (i != N - 1) and (j != N - 1):
                if sc_domain[i + 1, j + 1] != 0:
                    nn += 1 << 1

            if j != N - 1:
                if sc_domain[i, j + 1] != 0:
                    nn += 1 << 2

            if (j != N - 1) and (i != 0):
                if sc_domain[i - 1, j + 1] != 0:
                    nn += 1 << 3

            if i != 0:
                if sc_domain[i - 1, j] != 0:
                    nn += 1 << 4

            if (i != 0) and (j != 0):
                if sc_domain[i - 1, j - 1] != 0:
                    nn += 1 << 5

            if j != 0:
                if sc_domain[i, j - 1] != 0:
                    nn += 1 << 6

            if (j != 0) and (i != N - 1):
                if sc_domain[i + 1, j - 1] != 0:
                    nn += 1 << 7

        nn_map[i, j] = nn

    return nn_map.astype(np.int32)


@jit(cache=True)
def compute_normal(nn_map):
    """
    Given an nn map returns the normal vector to the domain boundary.
    """

    N, M = nn_map.shape

    normal_x = np.zeros((N, M))
    normal_y = np.zeros((N, M))

    for i, j in np.ndindex(N, M):

        if (nn_map[i, j] != 0) and (nn_map[i, j] != 255):
            if not int2bit(nn_map[i, j], 0):
                normal_x[i, j] = 1

            if not int2bit(nn_map[i, j], 4):
                normal_x[i, j] = -1

            if not int2bit(nn_map[i, j], 2):
                normal_y[i, j] = 1

            if not int2bit(nn_map[i, j], 6):
                normal_y[i, j] = -1

    norm = np.sqrt(normal_x ** 2 + normal_y ** 2) + 1e-30
    normal_x /= norm
    normal_y /= norm

    return normal_x, normal_y


@jit(cache=True)
def get_indices(x0, y0, L, dx):
    """
    Given a coordinate returns the indices.
    """
    return int((x0 + L / 2.0) / dx), int((y0 + L / 2.0) / dx)
