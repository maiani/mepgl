from numba import jit
import numpy as np


def build_domain(Li, Lf, N):
    """
    Build the meshgrids in cartesian and polar coordinates.
    """
    
    x_axis = np.linspace(Li, Lf, N)
    y_axis = np.linspace(Li, Lf, N)

    x, y = np.meshgrid(x_axis, y_axis, indexing='ij')

    r = np.sqrt(x ** 2 + y ** 2) + 0.000001
    theta = np.arctan2(y, x)
    
    return x, y, r, theta


def build_vortex_lattice(xx, yy, vortices, theta_0 = 0):
    M = vortices.shape[0]

    psi_abs = 1 + 0 * xx
    psi_theta = 0*xx + theta_0
    
    for i in range(M):
        if vortices[i, 0] != 0: 
            r_i = np.sqrt((xx - vortices[i, 1]) ** 2 + (yy - vortices[i, 2]) ** 2)
            psi_abs *= np.tanh(-(r_i ** 2))
            psi_theta += vortices[i, 0] * np.arctan2(yy - vortices[i, 2], xx - vortices[i, 1])
            
    u = psi_abs * np.cos(psi_theta)
    v = psi_abs * np.sin(psi_theta)

    return u, v
