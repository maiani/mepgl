#!/usr/bin/env python3

import os
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter

params = {
   'font.family' : 'STIXGeneral',
   'mathtext.fontset': 'stix',
   'axes.labelsize': 10,
   'legend.fontsize': 9,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'figure.figsize': [3.4, 3.4/1.618]
   }

plt.rcParams.update(params)
plt.close('all')


def mgl_plot(n, x, y, fenergy, s, psi_abs_1, psi_theta_1, j_x_1, j_y_1, \
                               psi_abs_2, psi_theta_2, j_x_2, j_y_2, b):
        
    nx, ny = x.shape    
    delta = int(nx/16)
    X  = x[::delta, ::delta]
    Y  = y[::delta, ::delta]    
    
    psi_delta = psi_theta_1-psi_theta_2
    psi_delta = ( psi_delta + np.pi) % (2 * np.pi ) - np.pi

    psi_abs_1_levels = np.linspace(0, np.nanmax(psi_abs_1), 64)
    psi_abs_2_levels = np.linspace(0, np.nanmax(psi_abs_2), 64)
    b_levels = np.linspace(np.nanmin(b), np.nanmax(b), 64)
    delta_levels = np.linspace(-np.pi, np.pi, 64)


    fig = plt.figure(figsize=(9, 5), constrained_layout=True)
    widths = [1, 1, 1.5]
    heights = [1, 1, 1, 1]
    gs = fig.add_gridspec(ncols=3, nrows=4, width_ratios=widths, height_ratios=heights)

    # UPPER LEFT
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_aspect('equal')
    im = ax1.contourf(x, y, psi_abs_1[n], levels=psi_abs_1_levels, cmap = 'viridis_r')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)        
    cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(0,np.nanmax(psi_abs_1),3))    
       
    # U  = uniform_filter(j_x_1[n], delta)[::delta,::delta]
    # V  = uniform_filter(j_y_1[n], delta)[::delta,::delta]
        
    # U[np.abs(U) < 1e-5] = np.nan
    # V[np.abs(V) < 1e-5] = np.nan
    # ax1.quiver(X, Y, U, V, scale=50*np.max(np.sqrt(U**2+V**2)), color=[1,1,1,0.5])
    
    ax1.set_title(r"Order parameter $\psi_1$")

    #UPPER RIGHT
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax2.set_aspect('equal')
    im = ax2.contourf(x, y, psi_abs_2[n], levels=psi_abs_2_levels, cmap = 'viridis_r')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)        
    cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(0,np.nanmax(psi_abs_2),3))    
 
    # U  = uniform_filter(j_x_2[n], delta)[::delta,::delta]
    # V  = uniform_filter(j_y_2[n], delta)[::delta,::delta]
        
    # U[np.abs(U) < 1e-5] = np.nan
    # V[np.abs(V) < 1e-5] = np.nan
    # ax2.quiver(X, Y, U, V, scale=50*np.max(np.sqrt(U**2+V**2)), color=[1,1,1,0.5])
    
    ax2.set_title(r"Order parameter $\psi_2$")

    # LOWER LEFT
    ax3 = fig.add_subplot(gs[2:4, 0])
    ax3.set_aspect('equal')
    im = ax3.contourf(x, y, psi_delta[n], levels=delta_levels, cmap = 'twilight')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(-np.pi, np.pi, 3))  
    ax3.set_title(r"$\Delta \theta$")

    # LOWER RIGHT
    ax4 = fig.add_subplot(gs[2:4, 1])
    ax4.set_aspect('equal')
    im = ax4.contourf(x, y, b[n], levels=b_levels, cmap = 'jet')     
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(np.nanmin(b), np.nanmax(b), 3))  
    ax4.set_title(r"Magnetic field $B_z$")  
    
    # RIGHT
    ax5 = fig.add_subplot(gs[1:3, 2])
    ax5.plot(s, fenergy, '.-')
    ax5.scatter(s[n], fenergy[n], color='C3')
    ax5.set_xlabel(r"s")
    ax5.set_ylabel(r"F(s)")
    ax5.set_title(r"Minimum Energy Path")  
    
    gs.tight_layout(figure=fig)

    return fig
    
multicomponent = True

###############################################################################

print("Loading data...", end="")

output_dir = "./output_data/"

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


############################ PLOTS  ############################################
#    
# Make a directory for the plots
plot_dir = "./plots/"

# Remove old output
#if os.path.exists(plot_dir):
#    shutil.rmtree(plot_dir)

os.makedirs(plot_dir, exist_ok = True)

# Final free energy 
#fig, ax = plt.subplots()
#ax.set_xlabel('Arc Length')
#ax.set_ylabel('Free energy')
#ax.scatter(s_gd, fenergy, color='C0', alpha=0.3)
#ax.plot(s_gd, fenergy, color='C0', label = r"$s^{gd}$")
#ax.scatter(s_ph, fenergy, color='C3', alpha=0.3)
#ax.plot(s_ph, fenergy, color='C3', label = r"$s^{ph}$")
#deltaFn = np.max(fenergy-fenergy[0])
#deltaFe = np.max(fenergy-fenergy[-1])
#ax.set_title(rf"$\Delta F_n =${deltaFn:.6}     $\Delta F_e =${deltaFe:.6}")
#ax.legend()
#ax.grid()
#fig.savefig(plot_dir+"fenergy.pdf", bbox_inches="tight")
#plt.close()

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

fig.legend(loc="lower right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
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

os.makedirs(plot_dir + "all/", exist_ok = True)


b_max = np.ceil(100*np.nanmax(b))/100.0
b_min = np.floor(100*np.nanmin(b))/100.0

for n in range(F):
    print(f"Frame {n+1}/{F}")

    str_n = '{:02d}'.format(n)

    fig = mgl_plot(n, x, y, fenergy, s_ph, psi_abs_1, psi_theta_1, j_x_1, j_x_2, psi_abs_2, psi_theta_2, j_x_2, j_y_2, b)
    fig.savefig(plot_dir + f"all/{str_n }-psi_2.png", dpi=300)
    plt.close()

