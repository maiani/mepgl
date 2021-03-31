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


@jit(cache=True)
def get_indices(x0, y0, L, dx):   
    return int((x0 + L/2.0)/dx), int((y0+L/2.0)/dx)


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


########################### POST ##############################################

@jit(cache=True)
def int2bit(x, n):
    return (x >> n) & 1


@jit(cache=True)
def build_nn_map(sc_domain):
    N, M = sc_domain.shape
    
    nn_map = np.zeros((N, M))
    
    for i,j in np.ndindex(N,M):
    
        nn = 0
    
        if(sc_domain[i, j]!=0):
            if(i != N-1):
                if(sc_domain[i+1, j]!=0):
                    nn += 1 << 0

            if((i != N-1) and (j != N-1)):
                if(sc_domain[i+1, j+1]!=0):
                    nn += 1 << 1
    
            if(j != N-1):
                if(sc_domain[i, j+1]!=0):
                    nn += 1 << 2
            
            if((j != N-1) and (i != 0)):
                if(sc_domain[i-1, j+1]!=0):
                    nn += 1 << 3
    
            if((i != 0)):
                if(sc_domain[i-1, j]!=0):
                    nn += 1 << 4
            
            if((i != 0)and(j != 0)):
                    if(sc_domain[i-1,j-1]!=0):
                        nn += 1 << 5
            
            if(j != 0):
                if(sc_domain[i, j-1 ]!=0):
                    nn += 1 << 6
    
            if((j != 0)and(i != N-1)):
                if(sc_domain[i+1, j-1]!=0):
                    nn += 1 << 7
                        
        nn_map[i, j] = nn
    
    return nn_map.astype(np.int32)



@jit(cache=True)
def compute_normal(nn_map):
    N, M = nn_map.shape
    
    normal_x = np.zeros((N,M))
    normal_y = np.zeros((N,M))
    
    for i,j in np.ndindex(N,M):
        
        if (nn_map[i, j]!=0) and (nn_map[i, j]!=255):
            if not int2bit(nn_map[i,j], 0):
                    normal_x[i,j] = 1

            if not int2bit(nn_map[i,j], 4):
                    normal_x[i,j] = -1

            if not int2bit(nn_map[i,j], 2):
                    normal_y[i,j] = 1

            if not int2bit(nn_map[i,j], 6):
                    normal_y[i,j] = -1                    

    norm = np.sqrt(normal_x**2 + normal_y**2)+1e-30
    normal_x /= norm
    normal_y /= norm

    return normal_x, normal_y

