#!/usr/bin/env python3
import json
import numpy as np
from mepgl_lib.builder import build_domain, get_indices, build_vortex_lattice

# Device number
dev_number = 0

############################# Batched parameters ##########################################

with open('./batched_params.json') as json_file:
    batched_params = json.load(json_file)

h_field = 1.2 # batched_params['h_field']
gamma   = batched_params['gamma']

simulation_name = f"m2f-gamma{gamma:+4.2f}"


#############################################################################################

# Name of the simulation
# simulation_name = "domain-wall-vortex"

# Number of points per side 
N = 401

# Number of computational frames per stage
F = np.array([61])

# Number of iterations per stage
iterations = np.array([1500])

# Modes of each stage:
# M -> Maxwell solver 
# S -> Normal solver
# L -> SSM with linear spline
# C -> SSM with cubic spline
modes = np.array(['L'])

default_relaxation_step_number = 10

########################## Computational domain definition
L   = 8.0

x, y, r, theta = build_domain(-L/2, +L/2, N)
dx = x[1,0]-x[0,0]

ones_mask = np.ones((N, N))

comp_domain = np.ones((N,N), dtype=np.int16)

#rc = 0.5 
#comp_domain[x**2 + (y-L/2)**2 <= rc] = 0
#comp_domain[x**2 + (y+L/2)**2 <= rc] = 0

# comp_domain[(x>0)&(y>0)]=0

#comp_domain[x**2 + y**2 > r_2] = 0

########################## Superconductive domain definition

sc_domain = comp_domain.copy()

#sc_domain[(N-1)//2, (N-1)//2] = 0
#sc_domain[x**2 + y**2 < r_1] = 0

########################## Superconductive parameters definition

multicomponent = True

q_1    = -1.0
a_1    = -1.0*ones_mask
b_1    = 1.0*ones_mask
m_xx_1 = 1.0*ones_mask
m_yy_1 = 1.0*ones_mask

q_2    = -1.0
a_2    = -1.0*ones_mask
b_2    = 1.0*ones_mask
m_xx_2 = 2.5*ones_mask
m_yy_2 = 2.5*ones_mask

eta    = 0.0
# gamma  = 0.0
delta  = 0.0

# h_field = 0.00

h = comp_domain * h_field

#############################################################################################

psi0_1 = np.mean(np.sqrt(-a_1/b_1))
psi0_2 = np.mean(np.sqrt(-a_2/b_2))

if eta!=0:
    if delta!= 0:
        theta_12 = np.arccos(-eta/delta * psi0_1 * psi0_2 / 2)
    else:
        theta_12 = -np.pi
else:
    if delta!= 0:
        theta_12 = np.pi/2
    else:
        theta_12 = 0


################################ Initial guess

# Position and winding number of the vortex 
# Structure of the vortices matrix:
# [vortex_id, frame, 0 ] = w, winding number, 0 to have no vortex
# [vortex_id, frame, 1 ] = x0, coordinate of the core
# [vortex_id, frame, 2 ] = y0, coordinate of the core

vortices_number = 1
vortices_1 = np.zeros((vortices_number, F[0], 3))
vortices_2 = np.zeros((vortices_number, F[0], 3))

# First vortex
x0_1 = np.linspace(0, 0,  F[0])
y0_1 = np.linspace(0, 0,  F[0])
w_1  = np.linspace(0, 0,  F[0])

vortices_1[0, :, 0] = w_1
vortices_1[0, :, 1] = x0_1
vortices_1[0, :, 2] = y0_1

x0_2 = np.linspace(L/2 + 4, 0,  F[0])
y0_2 = np.linspace(0, 0,  F[0])
w_2  = np.linspace(-1, -1,  F[0])

vortices_2[0, :, 0] = w_2
vortices_2[0, :, 1] = x0_2
vortices_2[0, :, 2] = y0_2

# Second vortex
# x0_1 = np.linspace(0, 0,  F[0])
# y0_1 = np.linspace(0, 0,  F[0])
# w_1  = np.linspace(-N_1, -N_1,  F[0])

# vortices_1[1, :, 0] = w_1
# vortices_1[1, :, 1] = x0_1
# vortices_1[1, :, 2] = y0_1

# x0_2 = np.linspace(0, 0,  F[0])
# y0_2 = np.linspace(0, 0,  F[0])
# w_2  = np.linspace(-N_2, -N_2,  F[0])

# vortices_2[1, :, 0] = w_2
# vortices_2[1, :, 1] = x0_2
# vortices_2[1, :, 2] = y0_2

# Build string
u_1 = np.zeros(shape=(F[0], N, N))
v_1 = np.zeros(shape=(F[0], N, N))
u_2 = np.zeros(shape=(F[0], N, N))
v_2 = np.zeros(shape=(F[0], N, N))
ax  = np.zeros(shape=(F[0], N, N))
ay  = np.zeros(shape=(F[0], N, N))


for n in range(F[0]):
    u_n_1, v_n_1 = build_vortex_lattice(x, y, vortices_1[:, n, :])
    u_1[n] = u_n_1 * sc_domain + 0.01 * np.random.randn(N, N) * sc_domain
    v_1[n] = v_n_1 * sc_domain + 0.01 * np.random.randn(N, N) * sc_domain

    u_n_2, v_n_2 = build_vortex_lattice(x, y, vortices_2[:, n, :],  theta_0 = - theta_12)
    u_2[n] = u_n_2 * sc_domain + 0.01 * np.random.randn(N, N) * sc_domain
    v_2[n] = v_n_2 * sc_domain + 0.01 * np.random.randn(N, N) * sc_domain
