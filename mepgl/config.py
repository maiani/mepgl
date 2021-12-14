#!/usr/bin/env python3
import numpy as np
from numpy.core.numeric import zeros_like

from mepgl_lib.builder import StringBuilder

############################# Batched parameters ##########################################
# import json
# with open('./batched_params.json') as json_file:
#     batched_params = json.load(json_file)

# h = batched_params['h']
# gamma = batched_params['gamma']
# simulation_name = f"syse-composite-{h:3.2}-gamma{gamma:+4.2f}"

#############################################################################################

# Name of the simulation
simulation_name = "single_component_test"

# Number of computational frames per stage
F = np.array([2])

# Number of iterations per stage
iterations = np.array([50])

# Modes of each stage:
# M -> Maxwell solver
# S -> Normal solver
# L -> SSM with linear spline
# C -> SSM with cubic spline
modes = np.array(["S"])

default_relaxation_step_number = 30

multicomponent = False

########################################################################

# Number of points per side
N  = 501
L = 10

x_lim = [-L / 2, L / 2]
y_lim = [-L / 2, L / 2]

builder = StringBuilder(F[0], Nx=N, Ny=N, x_lim=x_lim, y_lim=y_lim, multicomponent=multicomponent)

x, y = builder.x, builder.y
dx = builder.x_ax[1] - builder.x_ax[0]

comp_domain, sc_domain = builder.get_domain_matrices()

########################## Superconductive parameters definition

q_1 = -1.0
a_1 = -1.0
b_1 = 1.0
m_xx_1 = 1.0
m_yy_1 = 1.0

q_2 = -1.0
a_2 = -1.0
b_2 = 1.0
m_xx_2 = 1.0
m_yy_2 = 1.0

eta = +0.00
gamma = +0.00
delta = +0.00

h_z = 0.8

####################
ones_mask = np.ones((N, N), dtype=float)

a_1 *= ones_mask
b_1 *= ones_mask
m_xx_1 *= ones_mask
m_yy_1 *= ones_mask

a_2 *= ones_mask
b_2 *= ones_mask
m_xx_2 *= ones_mask
m_yy_2 *= ones_mask

h = h_z * comp_domain

#############################################################################################

psi0_1 = np.mean(np.sqrt(-a_1 / b_1))
psi0_2 = np.mean(np.sqrt(-a_2 / b_2))

if eta != 0:
    if delta != 0:
        theta_12 = np.arccos(-eta / delta * psi0_1 * psi0_2 / 2)
    else:
        theta_12 = -np.pi
else:
    if delta != 0:
        theta_12 = np.pi / 2
    else:
        theta_12 = 0

builder.add_phase_diff(lambda x, y: theta_12)

xv = np.linspace(L / 2 + 2, 0, F[0])
yv = np.linspace(0.0, 0.0, F[0])
wv = np.linspace(-1, -1, F[0])

builder.add_vortex(xv, yv, wv, 1)
# builder.add_vortex(xv, yv, wv, 2)

u_1, v_1, u_2, v_2 = builder.generate_matter_fields(
    psi_abs_1=psi0_1, psi_abs_2=psi0_2, noise=0.0
)

#ax = np.zeros_like(u_1)
#ay = np.zeros_like(u_1)
builder.B_z = h_z
ax, ay = builder.generate_vector_potential(gauge="symmetric")
