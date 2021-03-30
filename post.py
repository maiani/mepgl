#!/usr/bin/env python3

"""
Post processing for string methods applied to Ginzburg-Landau systems
"""

import argparse
import importlib.util
import os

import numpy as np

from matplotlib import pyplot as plt

from meplib.gl import build_nn_map, compute_normal, g_BL, f_BL

from meplib.plot_utilities import *


params = {
   'font.family' : 'STIXGeneral',
   'mathtext.fontset': 'stix',
   'axes.labelsize': 10,
   'legend.fontsize': 9,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'figure.figsize': [3.4, 2.5]
   }


plt.rcParams.update(params)

plt.close('all')

# Parse input
parser = argparse.ArgumentParser(description='Post processing.')
parser.add_argument('-s', '--simulation', help="simulation to process")
parser.add_argument('-a', '--animations', help="create animations", action='store_true')
args = parser.parse_args()


# If no input argumets are given use config file present in the root directory
if args.simulation is None:
    import config
    simulation_name = config.simulation_name

# If input argument is given use the selected simulation config file
else:
    spec = importlib.util.spec_from_file_location('config', f'./simulations/{args.simulation}/config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    simulation_name = args.simulation
    
# Manually import all the used variables
multicomponent = config.multicomponent

###############################################################################

print("Loading data...", end="")

simulation_dir = "./simulations/"+simulation_name+"/"
output_dir = "./simulations/"+simulation_name+"/output_data/"

# Loading meshgrid and external field
x = np.load(output_dir+"x.npy").astype(np.float64)
y = np.load(output_dir+"y.npy").astype(np.float64)

comp_domain = np.load(output_dir+"comp_domain.npy").astype(np.int32)
sc_domain = np.load(output_dir+"sc_domain.npy").astype(np.int32)

a_1 = np.load(output_dir+"a_1.npy").astype(np.float64)
b_1 = np.load(output_dir+"b_1.npy").astype(np.float64)
m_xx_1 = np.load(output_dir+"m_xx_1.npy").astype(np.float64)
m_yy_1 = np.load(output_dir+"m_yy_1.npy").astype(np.float64)

a_2 = np.load(output_dir+"a_2.npy").astype(np.float64)
b_2 = np.load(output_dir+"b_2.npy").astype(np.float64)
m_xx_2 = np.load(output_dir+"m_xx_2.npy").astype(np.float64)
m_yy_2 = np.load(output_dir+"m_yy_2.npy").astype(np.float64)

q = np.load(output_dir+"q.npy").astype(np.float64)
h = np.load(output_dir+"h.npy").astype(np.float64)

fenergy = np.load(output_dir+"fenergy.npy").astype(np.float64)
s_gd = np.load(output_dir+"s.npy").astype(np.float64)
s_ph = np.load(output_dir+"s_jb.npy").astype(np.float64)

dx = x[1][0]-x[0][0]
N = x.shape[0]
F = s_gd.shape[0]

# Loading fields and states
u_1  = np.zeros(shape=(F,N,N))
v_1  = np.zeros(shape=(F,N,N))
a_x = np.zeros(shape=(F,N,N))
a_y = np.zeros(shape=(F,N,N))
b  = np.zeros(shape=(F,N,N))
j_x_1 = np.zeros(shape=(F,N,N))
j_y_1 = np.zeros(shape=(F,N,N))

u_2 = np.zeros(shape=(F,N,N))
v_2 = np.zeros(shape=(F,N,N))
j_x_2 = np.zeros(shape=(F,N,N))  
j_y_2 = np.zeros(shape=(F,N,N))


# Build mask
comp_mask = np.ones((N, N))
sc_mask = np.ones((N, N))
comp_mask[comp_domain==0] = np.nan
sc_mask[sc_domain==0] = np.nan
    
for n in range(F):
    u_1[n] = np.load(output_dir + f"{n}/u_1.npy").astype(np.float64)
    v_1[n] = np.load(output_dir + f"{n}/v_1.npy").astype(np.float64)
    a_x[n] = np.load(output_dir + f"{n}/ax.npy").astype(np.float64)
    a_y[n] = np.load(output_dir + f"{n}/ay.npy").astype(np.float64)
    b[n] = np.load(output_dir + f"{n}/b.npy").astype(np.float64)
    j_x_1[n] = np.load(output_dir + f"{n}/jx_1.npy").astype(np.float64)
    j_y_1[n] = np.load(output_dir + f"{n}/jy_1.npy").astype(np.float64)
    
    if multicomponent:
        u_2[n] = np.load(output_dir + f"{n}/u_2.npy").astype(np.float64)
        v_2[n] = np.load(output_dir + f"{n}/v_2.npy").astype(np.float64)
        j_x_2[n] = np.load(output_dir + f"{n}/jx_2.npy").astype(np.float64)
        j_y_2[n] = np.load(output_dir + f"{n}/jy_2.npy").astype(np.float64)        

print(" done.")
    

#################### COMPUTING QUANTITIES #####################################
print("Elaborating data...", end="")

# Compute superconducting field
psi_abs_1 = np.sqrt(u_1**2+v_1**2)
psi_theta_1 = np.arctan2(v_1, u_1)
if multicomponent:
    psi_abs_2 = np.sqrt(u_2**2+v_2**2)
    psi_theta_2 = np.arctan2(v_2, u_2)    

phi = np.zeros(F)
for n in range(F):
    phi[n] = np.sum(b[n])*dx*dx

# Winding number
    
w_1 = np.zeros(F)
w_2 = np.zeros(F)

if not multicomponent:
    w_2 *= np.nan

for n in range(F):
    
    for i in range(N-1):
        w_1[n] += (+u_1[n, i, 0]*(v_1[n, i+1, 0]-v_1[n, i, 0])
                 -v_1[n, i, 0]*(u_1[n, i+1, 0]-u_1[n, i, 0])
        )/(u_1[n, i, 0]**2 + v_1[n, i, 0]**2)    

        w_1[n] += (+u_1[n, -1, i]*(v_1[n, -1,i+1]-v_1[n, -1, i])
                 -v_1[n, -1, i]*(u_1[n, -1,i+1]-u_1[n, -1, i])
        )/(u_1[n, -1, i]**2 + v_1[n, -1, i]**2)    

        w_1[n] += (+u_1[n, -1-i, -1]*(v_1[n, -2-i, -1]-v_1[n, -1-i, -1])
                 -v_1[n, -1-i, -1]*(u_1[n, -2-i, -1]-u_1[n, -1-i, -1])
        )/(u_1[n, -1-i, -1]**2 + v_1[n, -1-i, -1]**2)    

        w_1[n] += (+u_1[n, 0, -1-i]*(v_1[n, 0, -2-i]-v_1[n, 0, -1-i])
                 -v_1[n, 0, -1-i]*(u_1[n, 0, -2-i]-u_1[n, 0, -1-i])
        )/(u_1[n, 0, -1-i]**2 + v_1[n, 0, -1-i]**2)    

        if multicomponent:    
            w_2[n] += (+u_2[n, i, 0]*(v_2[n, i+1, 0]-v_2[n, i, 0])
                    -v_2[n, i, 0]*(u_2[n, i+1, 0]-u_2[n, i, 0])
            )/(u_2[n, i, 0]**2 + v_2[n, i, 0]**2)    

            w_2[n] += (+u_2[n, -1, i]*(v_2[n, -1,i+1]-v_2[n, -1, i])
                    -v_2[n, -1, i]*(u_2[n, -1,i+1]-u_2[n, -1, i])
            )/(u_2[n, -1, i]**2 + v_2[n, -1, i]**2)    

            w_2[n] += (+u_2[n, -1-i, -1]*(v_2[n, -2-i, -1]-v_2[n, -1-i, -1])
                    -v_2[n, -1-i, -1]*(u_2[n, -2-i, -1]-u_2[n, -1-i, -1])
            )/(u_2[n, -1-i, -1]**2 + v_2[n, -1-i, -1]**2)    

            w_2[n] += (+u_2[n, 0, -1-i]*(v_2[n, 0, -2-i]-v_2[n, 0, -1-i])
                    -v_2[n, 0, -1-i]*(u_2[n, 0, -2-i]-u_2[n, 0, -1-i])
            )/(u_2[n, 0, -1-i]**2 + v_2[n, 0, -1-i]**2)  
        

w_1 /= 2*np.pi
w_2 /= 2*np.pi
print(" done.")

# Saving data
np.savez(simulation_dir + f"{simulation_name}.npz", fenergy=fenergy,
                                                    s_gd=s_gd,
                                                    s_ph=s_ph,
                                                    w_1 = w_1,
                                                    w_2 = w_2, 
                                                    phi = phi)

############################ PLOTS  ############################################
#    
# Make a directory for the plots
plot_dir = "./simulations/"+simulation_name+"/plots/"

# Remove old output
#if os.path.exists(plot_dir):
#    shutil.rmtree(plot_dir)

os.makedirs(plot_dir, exist_ok = True)

## Plotting stats
## iter_axis = np.arange(0, f_opt.shape[1])
#
## fig, ax1 = plt.subplots()
## for n in range(F_final):
##     ax1.plot(iter_axis, f_opt[n], label=f"{n}")
## plt.title('Free Energy')
## plt.savefig(plot_dir+"fenergy_iter.pdf")
#
## fig, ax2 = plt.subplots()
## ax2.set_yscale('log')
## for n in range(F_final):
##     ax2.plot(iter_axis, g_norm2[n], ':', label=f"{n}")
## plt.title(r"$||g||^2$")
## plt.savefig(plot_dir+"g_norm2.pdf")

# Final free energy 
fig, ax = plt.subplots()
ax.set_xlabel('Arc Length')
ax.set_ylabel('Free energy')
ax.scatter(s_gd, fenergy, color='C0', alpha=0.3)
ax.plot(s_gd, fenergy, color='C0', label = r"$s^{gd}$")
ax.scatter(s_ph, fenergy, color='C3', alpha=0.3)
ax.plot(s_ph, fenergy, color='C3', label = r"$s^{ph}$")
deltaFn = np.max(fenergy-fenergy[0])
deltaFe = np.max(fenergy-fenergy[-1])
ax.set_title(rf"$\Delta F_n =${deltaFn:.6}     $\Delta F_e =${deltaFe:.6}")
ax.legend()
ax.grid()
fig.savefig(plot_dir+"fenergy.pdf", bbox_inches="tight")
plt.close()


s = s_ph

# F and N
fig, ax = plt.subplots()
ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$F$')
ax.plot(s, fenergy, '-C3', linewidth=1, label=r"$F$")

ax2 = ax.twinx()
ax2.set_ylabel(r'$N$')
ax2.plot(s, w_1, '-', linewidth=0.6, label=r"$N_1$")
ax2.plot(s, w_2, '-C2', linewidth=0.6, label=r"$N_2$")

fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
fig.tight_layout()
fig.savefig(plot_dir+"fenergy-N.pdf")
plt.close()

# Phi and N
fig, ax = plt.subplots()
ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$N$')
ax.plot(s, w_1, '.-', label=r"$N_1$")
ax.plot(s, w_2, '.-', label=r"$N_2$")
ax2 = ax.twinx()
ax2.set_ylabel(r'$\Phi$')
ax2.plot(s, phi, '.-C3')
fig.tight_layout()
fig.savefig(plot_dir+"Phi-N.pdf")
plt.close()


# Plotting fields
# os.makedirs(plot_dir + "psi_1/", exist_ok = True)
# os.makedirs(plot_dir + "psi_2/", exist_ok = True)
#os.makedirs(plot_dir + "psi_3D/", exist_ok = True)
# os.makedirs(plot_dir + "magnetic_field/", exist_ok = True)
# os.makedirs(plot_dir + "magnetic_field_3D/", exist_ok = True)
# os.makedirs(plot_dir + "current_field_1/", exist_ok = True)
# os.makedirs(plot_dir + "current_field_2/", exist_ok = True)
os.makedirs(plot_dir + "all/", exist_ok = True)

#fig, ax = plot_single(x, y, fenergy, psi_abs*sc_mask, b*comp_mask, j_x_1, j_y_1)
#fig.savefig(plot_dir + "single.png", dpi=500)


# b_max = np.ceil(100*np.nanmax(b))/100.0
# b_min = np.floor(100*np.nanmin(b))/100.0

for n in range(F):
    print(f"Frame {n+1}/{F}")

    str_n = '{:03d}'.format(n)
     
    # complex_plot(x, y, psi_abs_1[n]*sc_mask, psi_theta_1[n]*sc_mask)
    # plt.savefig(plot_dir + f"psi_1/{str_n }-psi_1.png", bbox_inches="tight")
    # plt.close()    
    
#    complex_plot_3D(x, y, psi_abs[n]*sc_mask, psi_theta[n]*sc_mask)
#    plt.savefig(plot_dir + f"psi_3D/{str_n }-psi.png")
#    plt.close()

    # magnetic_field_plot(x, y, b[n]*comp_mask, b_max, b_min)
    # plt.savefig(plot_dir + f"magnetic_field/{str_n }-magnetic_field.png", bbox_inches="tight")
    # plt.close()
        
#    magnetic_field_plot_3D(x, y, b[n]*comp_mask, b_max, b_min)
#    plt.savefig(plot_dir + f"magnetic_field_3D/{str_n }-magnetic_field_3D.png")
#    plt.close()
        
    # current_plot(x, y, j_x_1[n]*sc_mask, j_y_1[n]*sc_mask, 5)
    # plt.savefig(plot_dir + f"current_field_1/{str_n }-current_field_1.png", bbox_inches="tight")
    # plt.close()

    if multicomponent:
        fig = mgl_plot(n, x, y, fenergy, s_ph, psi_abs_1, psi_theta_1, j_x_1, j_x_2, psi_abs_2, psi_theta_2, j_x_2, j_y_2, b)
        fig.savefig(plot_dir + f"all/{str_n }.png", dpi=400)
        plt.close()

        # complex_plot(x, y, psi_abs_2[n]*sc_mask, psi_theta_2[n]*sc_mask)
        # plt.savefig(plot_dir + f"psi_2/{str_n }-psi_2.png", bbox_inches="tight")
        # plt.close()  

        # current_plot(x, y, j_x_2[n]*sc_mask, j_y_2[n]*sc_mask, 5)
        # plt.savefig(plot_dir + f"current_field_2/{str_n }-current_field_2.png", bbox_inches="tight")
        # plt.close()
