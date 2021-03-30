from numba import jit
import numpy as np
from scipy.special import k0, k1


def build_domain(Li, Lf, N):
    """
    Build the meshgrids in cartesian and polar coordinates
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


@jit(cache=True)
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


############################# BEAN LIVINGSTON BARRIER #########################

def g_BL(x, H, k):
    return 4*np.pi/k*(H*(np.exp(-x)-1)-1/k*k0(2*x)+np.log(k)/k)


def f_BL(x, H, k):
    return 4*np.pi/k*(H*np.exp(-x)-2/k*k1(2*x)) 

######################### OLD PYTHON ROUTINES ##################################

#@jit(cache=True)
# def free_energy_density(nn_map, dx, kappa, h, ax, ay, u, v):
    
#     F, N, M = u.shape
    
#     nl_term = (0.5*(1 - u**2 - v**2)**2)*(int2bit(nn_map, 1) + int2bit(nn_map, 3) + int2bit(nn_map, 5) + int2bit(nn_map, 7))/4.0

#     kinetic_term_x = np.zeros_like(nl_term)
#     kinetic_term_y = np.zeros_like(nl_term)
    
#     cov_dev_u_x_b =  (-2.0/kappa*(u[:, +1:, :] - u[:, :-1, :])/dx + ax[:, :-1, :]*v[:, :-1, :]) 
#     cov_dev_u_x_f =  (-2.0/kappa*(u[:, +1:, :] - u[:, :-1, :])/dx + ax[:, +1:, :]*v[:, +1:, :]) 
#     cov_dev_v_x_b =  (+2.0/kappa*(v[:, +1:, :] - v[:, :-1, :])/dx + ax[:, :-1, :]*u[:, :-1, :]) 
#     cov_dev_v_x_f =  (+2.0/kappa*(v[:, +1:, :] - v[:, :-1, :])/dx + ax[:, +1:, :]*u[:, +1:, :]) 
    
    
#     kinetic_term_x = 0.25*( \
#                             cov_dev_u_x_b**2 + \
#                             cov_dev_u_x_f**2 + \
#                             cov_dev_v_x_b**2 + \
#                             cov_dev_v_x_f**2   \
#             )

#     cov_dev_u_y_b = (-2.0/kappa*(u[:, :, +1:] - u[:, :, :-1])/dx + ay[:, :, :-1 ]*v[:, :, :-1])
#     cov_dev_u_y_f = (-2.0/kappa*(u[:, :, +1:] - u[:, :, :-1])/dx + ay[:, :, +1: ]*v[:, :, +1:])
#     cov_dev_v_y_b = (+2.0/kappa*(v[:, :, +1:] - v[:, :, :-1])/dx + ay[:, :, :-1 ]*u[:, :, :-1])
#     cov_dev_v_y_f = (+2.0/kappa*(v[:, :, +1:] - v[:, :, :-1])/dx + ay[:, :, +1: ]*u[:, :, +1:])


#     kinetic_term_y = 0.25*( \
#                 cov_dev_u_y_b**2 + \
#                 cov_dev_u_y_f**2 + \
#                 cov_dev_v_y_b**2 + \
#                 cov_dev_v_y_f**2   \
#         )
   
#     em_term_02 = np.zeros((F, N, M))
#     em_term_24 = np.zeros((F, N, M))
#     em_term_46 = np.zeros((F, N, M))
#     em_term_60 = np.zeros((F, N, M))

#     for n in range(F):
#         kinetic_term_x[n] *= (int2bit(nn_map[:-1, :], 1) + int2bit(nn_map[:-1, :], 7))/2.0
#         kinetic_term_y[n] *= (int2bit(nn_map[:, :-1], 1) + int2bit(nn_map[:, :-1], 3))/2.0

#         em_term_02[n, :-1, :-1]  = (0.25*0.5*((ay[n, +1:, :-1] - ay[n, :-1, :-1])/dx - (ax[n, :-1, +1:] - ax[n, :-1, :-1])/dx - h[:-1, :-1])**2)*(int2bit(nn_map[:-1, :-1], 0) & int2bit(nn_map[:-1, :-1], 2))
#         em_term_24[n, +1:, :-1]  = (0.25*0.5*((ay[n, +1:, :-1] - ay[n, :-1, :-1])/dx - (ax[n, +1:, +1:] - ax[n, +1:, :-1])/dx - h[+1:, :-1])**2)*(int2bit(nn_map[+1:, :-1], 2) & int2bit(nn_map[+1:, :-1], 4))
#         em_term_46[n, +1:, +1:]  = (0.25*0.5*((ay[n, +1:, +1:] - ay[n, :-1, +1:])/dx - (ax[n, +1:, +1:] - ax[n, +1:, :-1])/dx - h[+1:, +1:])**2)*(int2bit(nn_map[+1:, +1:], 4) & int2bit(nn_map[+1:, +1:], 6))
#         em_term_60[n, :-1, +1:]  = (0.25*0.5*((ay[n, +1:, +1:] - ay[n, :-1, +1:])/dx - (ax[n, :-1, +1:] - ax[n, :-1, :-1])/dx - h[:-1, +1:])**2)*(int2bit(nn_map[:-1, +1:], 6) & int2bit(nn_map[:-1, +1:], 0))


#     returned = np.zeros((F))

#     for n in range(F):
#         returned[n] =   np.sum(nl_term[n]) + \
#                         np.sum(kinetic_term_x[n]) + \
#                         np.sum(kinetic_term_y[n]) + \
#                         np.sum(em_term_02[n] + em_term_24[n] + em_term_46[n] + em_term_60[n])
    
#     return returned*dx*dx


#@jit(cache=True)
# def magnetic_field(nn_map, dx, h, ax, ay):
    
#     F, N, M = ax.shape
    
      
#     em_term_02 = np.zeros((F, N, M))
#     em_term_24 = np.zeros((F, N, M))
#     em_term_46 = np.zeros((F, N, M))
#     em_term_60 = np.zeros((F, N, M))

#     for n in range(F):

#         em_term_02[n, :-1, :-1]  = (((ay[n, +1:, :-1] - ay[n, :-1, :-1])/dx - (ax[n, :-1, +1:] - ax[n, :-1, :-1])/dx))*(int2bit(nn_map[:-1, :-1], 0) & int2bit(nn_map[:-1, :-1], 1) & int2bit(nn_map[:-1, :-1], 2))
#         em_term_24[n, +1:, :-1]  = (((ay[n, +1:, :-1] - ay[n, :-1, :-1])/dx - (ax[n, +1:, +1:] - ax[n, +1:, :-1])/dx))*(int2bit(nn_map[+1:, :-1], 2) & int2bit(nn_map[+1:, :-1], 3) & int2bit(nn_map[+1:, :-1], 4))
#         em_term_46[n, +1:, +1:]  = (((ay[n, +1:, +1:] - ay[n, :-1, +1:])/dx - (ax[n, +1:, +1:] - ax[n, +1:, :-1])/dx))*(int2bit(nn_map[+1:, +1:], 4) & int2bit(nn_map[+1:, +1:], 5) & int2bit(nn_map[+1:, +1:], 6))
#         em_term_60[n, :-1, +1:]  = (((ay[n, +1:, +1:] - ay[n, :-1, +1:])/dx - (ax[n, :-1, +1:] - ax[n, :-1, :-1])/dx))*(int2bit(nn_map[:-1, +1:], 6) & int2bit(nn_map[:-1, +1:], 7) & int2bit(nn_map[:-1, +1:], 0))


#     return (em_term_02 + em_term_24 + em_term_46 + em_term_60)/((int2bit(nn_map, 0) & int2bit(nn_map, 1) & int2bit(nn_map, 2)) + \
#                                                                 (int2bit(nn_map, 2) & int2bit(nn_map, 3) & int2bit(nn_map, 4)) + \
#                                                                 (int2bit(nn_map, 4) & int2bit(nn_map, 5) & int2bit(nn_map, 6)) + \
#                                                                 (int2bit(nn_map, 6) & int2bit(nn_map, 7) & int2bit(nn_map, 0)))
    
         
