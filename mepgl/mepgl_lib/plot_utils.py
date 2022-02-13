import numpy as np
from matplotlib import animation, cm, colorbar, colors, gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter

plt.style.use("mepgl.mplstyle")

############################### STATIC PLOTS SINGLE COMPONENT ###############################


def mfep_with_bl(x, y, s_ph, fenergy, psi_abs, b, j_x, j_y, bl, x_ax, x0):

    ns = fenergy.argmax()
    j = np.sqrt(j_x ** 2 + j_y ** 2)
    levels_j = np.linspace(0, np.nanmax(j), 64)

    nx, ny = x.shape
    dn = int(nx * 4 / 100)
    delta = int((nx - 2 * dn) / 5)

    X = x[dn:-dn:delta, dn:-dn:delta]
    Y = y[dn:-dn:delta, dn:-dn:delta]

    a_color = [232 / 255, 211 / 255, 63 / 255, 1]
    a_width = 0.015

    fig = plt.figure(figsize=(4, 3))
    gs = gridspec.GridSpec(
        4,
        4,
        wspace=0.08,
        width_ratios=[1, 1, 1, 0.08],
        height_ratios=[0.005, 2, 2, 0.5],
    )

    ax = plt.subplot(gs[0:2, 0])
    ax.contourf(x, y, j[0], levels=levels_j, cmap="Blues_r")
    ax.set_yticks([-4.5, 0, 4.5])
    ax.set_xticks([-4.5, 0, 4.5])
    ax.set_yticklabels(["-5", "0", "5"])
    ax.set_xticklabels(["-5", "0", "5"])
    ax.set_aspect("equal")

    U = uniform_filter(j_x[0], delta)[dn:-dn:delta, dn:-dn:delta]
    V = uniform_filter(j_y[0], delta)[dn:-dn:delta, dn:-dn:delta]

    U[np.abs(U) < 1e-8] = 0
    V[np.abs(V) < 1e-8] = 0

    ax.quiver(
        X,
        Y,
        U,
        V,
        color=a_color,
        scale=9 * np.max(np.sqrt(U ** 2 + V ** 2)),
        width=a_width,
    )

    ax = plt.subplot(gs[0:2, 1])
    ax.contourf(x, y, j[ns], levels=levels_j, cmap="Blues_r")
    ax.set_yticks([-4.5, 0, 4.5])
    ax.set_xticks([-4.5, 0, 4.5])
    ax.set_yticklabels([])
    ax.set_xticklabels(["-5", "0", "5"])
    ax.set_aspect("equal")

    U = uniform_filter(j_x[ns], delta)[dn:-dn:delta, dn:-dn:delta]
    V = uniform_filter(j_y[ns], delta)[dn:-dn:delta, dn:-dn:delta]

    U[np.abs(U) < 1e-8] = 0
    V[np.abs(V) < 1e-8] = 0

    ax.quiver(
        X,
        Y,
        U,
        V,
        color=a_color,
        scale=9 * np.max(np.sqrt(U ** 2 + V ** 2)),
        width=a_width,
    )

    ax = plt.subplot(gs[0:2, 2])
    im = ax.contourf(x, y, j[-1], levels=levels_j, cmap="Blues_r")
    ax.set_yticks([-4.5, 0, 4.5])
    ax.set_xticks([-4.5, 0, 4.5])
    ax.set_yticklabels([])
    ax.set_xticklabels(["-5", "0", "5"])
    ax.set_aspect("equal")

    U = uniform_filter(j_x[-1], delta)[dn:-dn:delta, dn:-dn:delta]
    V = uniform_filter(j_y[-1], delta)[dn:-dn:delta, dn:-dn:delta]

    U[np.abs(U) < 1e-8] = 0
    V[np.abs(V) < 1e-8] = 0

    ax.quiver(
        X,
        Y,
        U,
        V,
        color=a_color,
        scale=9 * np.max(np.sqrt(U ** 2 + V ** 2)),
        width=a_width,
    )

    cax = plt.subplot(gs[1, 3])

    fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(j.min(), j.max(), 3))
    # cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(psi_abs.min(), psi_abs.max(), 3))
    # cbar.set_label(r"$|\psi|$") #, rotation=0)
    cax.set_title(r"$\left| \mathbf{j} \right|$", size=13)

    ax = plt.subplot(gs[2:, :])
    # ax.plot(s_ph, fenergy, color="C7", linewidth=0.8)
    ax.plot(x_ax, bl, "--", color="C7", linewidth=0.8, label="BL model")
    ax.plot(
        np.abs(x0 - x[-1, -1]),
        fenergy,
        ".",
        color="r",
        markersize=2,
        alpha=0.9,
        label="MFEP",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F(x)$")
    ax.legend()

    gs.tight_layout(fig, pad=0.2)
    return fig


def mfep_psi_B(x, y, fenergy, psi_abs, b, j_x, j_y):

    ns = fenergy.argmax()

    levels = np.linspace(0, np.nanmax(psi_abs), 20)

    fig, ax = plt.subplots(2, 3, dpi=300, figsize=(7, 3), sharex=True, sharey=True)

    ax[0, 0].contourf(x, y, psi_abs[0], levels=levels, cmap="viridis_r")
    ax[0, 1].contourf(x, y, psi_abs[ns], levels=levels, cmap="viridis_r")
    im = ax[0, 2].contourf(x, y, psi_abs[-1], levels=levels, cmap="viridis_r")

    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(0, 1, 3))
    cbar.set_label(r"$|\psi|$")

    nx, ny = x.shape
    delta = int(nx / 25)

    X = x[::delta, ::delta]
    Y = y[::delta, ::delta]

    U = uniform_filter(j_x[0], delta)[::delta, ::delta]
    V = uniform_filter(j_y[0], delta)[::delta, ::delta]

    U[np.abs(U) < 1e-8] = 0
    V[np.abs(V) < 1e-8] = 0

    ax[0, 0].quiver(
        X,
        Y,
        U,
        V,
        color=[0.9, 0.9, 0.9, 0.4],
        scale=18 * np.max(np.sqrt(U ** 2 + V ** 2)),
    )

    U = uniform_filter(j_x[ns], delta)[::delta, ::delta]
    V = uniform_filter(j_y[ns], delta)[::delta, ::delta]

    U[np.abs(U) < 1e-8] = 0
    V[np.abs(V) < 1e-8] = 0

    ax[0, 1].quiver(
        X,
        Y,
        U,
        V,
        color=[0.9, 0.9, 0.9, 0.4],
        scale=18 * np.max(np.sqrt(U ** 2 + V ** 2)),
    )

    U = uniform_filter(j_x[-1], delta)[::delta, ::delta]
    V = uniform_filter(j_y[-1], delta)[::delta, ::delta]

    U[np.abs(U) < 1e-8] = 0
    V[np.abs(V) < 1e-8] = 0

    ax[0, 2].quiver(
        X,
        Y,
        U,
        V,
        color=[0.9, 0.9, 0.9, 0.4],
        scale=18 * np.max(np.sqrt(U ** 2 + V ** 2)),
    )

    levels = np.linspace(np.nanmin(b), np.nanmax(b), 40)

    ax[1, 0].contourf(x, y, b[0], levels=levels, cmap="jet")
    ax[1, 1].contourf(x, y, b[ns], levels=levels, cmap="jet")
    im = ax[1, 2].contourf(x, y, b[-1], levels=levels, cmap="jet")

    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(
        im, cax=cax, format="%1.1f", ticks=np.linspace(np.nanmin(b), np.nanmax(b), 3)
    )
    cbar.set_label(r"$|\vec B|$")

    fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    return fig, ax


def plot_single(x, y, fenergy, psi_abs, b, j_x, j_y):

    # levels = np.linspace(0, 1, 20)
    levels = np.linspace(0, np.nanmax(psi_abs), 20)

    fig, ax = plt.subplots(1, 3, dpi=300, figsize=(16, 4), sharex=True, sharey=True)

    im = ax[0].contourf(x, y, psi_abs[0], levels=levels, cmap="viridis_r")

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(0, 1, 3))

    # levels = np.linspace(0, b.max(), 40)
    levels = np.linspace(np.nanmin(b), np.nanmax(b), 40)

    im = ax[1].contourf(x, y, b[0], levels=levels, cmap="jet")

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(0,b.max(),3))
    cbar = fig.colorbar(
        im, cax=cax, format="%1.1f", ticks=np.linspace(np.nanmin(b), np.nanmax(b), 3)
    )

    nx, ny = x.shape
    delta = int(nx / 32)

    X = x[::delta, ::delta]
    Y = y[::delta, ::delta]

    U = uniform_filter(j_x[0], delta)[::delta, ::delta]
    V = uniform_filter(j_y[0], delta)[::delta, ::delta]

    U[np.abs(U) < 1e-8] = 0
    V[np.abs(V) < 1e-8] = 0

    ax[2].quiver(X, Y, U, V, scale=110 * np.max(np.sqrt(U ** 2 + V ** 2)))

    ax[0].set_title(r"Order parameter $\psi$")
    ax[1].set_title(r"Magnetic field magnitude $\abs{\vec{B}}$")
    ax[2].set_title(r"Superconductive current $\vec j_s$")

    fig.tight_layout()

    return fig, ax


def current_with_abs(x, y, j_x, j_y, psi_abs, borders=None):
    fig, ax = plt.subplots(dpi=300)

    levels = np.linspace(0, 1, 20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    im = ax.contourf(x, y, psi_abs, levels=levels)

    cbar = fig.colorbar(im, cax=cax, format="%1.1f", ticks=np.linspace(0, 1, 11))
    cbar.set_label(r"$|\psi|$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    nx, ny = x.shape

    delta = int(nx / 32)

    X = x[::delta, ::delta]
    Y = y[::delta, ::delta]

    U = uniform_filter(j_x, delta)[::delta, ::delta]
    V = uniform_filter(j_y, delta)[::delta, ::delta]

    ax.quiver(X, Y, U, V, color=[0.2, 0.2, 0.2, 0.6], scale=6)

    if borders is not None:
        idx_x, idx_y = np.argwhere(borders == 1).T
        xx = x[idx_x, idx_y]
        yy = y[idx_x, idx_y]
        ax.scatter(xx, yy, 0.05, "r")

    fig.tight_layout()

    return fig, ax


def mgl_plot(
    n,
    x,
    y,
    fenergy,
    s,
    psi_abs_1,
    psi_theta_1,
    j_x_1,
    j_y_1,
    psi_abs_2,
    psi_theta_2,
    j_x_2,
    j_y_2,
    b,
):

    nx, ny = x.shape
    delta = int(nx / 16)
    X = x[::delta, ::delta]
    Y = y[::delta, ::delta]

    psi_delta = psi_theta_1 - psi_theta_2
    psi_delta = (psi_delta + np.pi) % (2 * np.pi) - np.pi

    psi_abs_1_levels = np.linspace(0, np.nanmax(psi_abs_1), 64)
    psi_abs_2_levels = np.linspace(0, np.nanmax(psi_abs_2), 64)
    b_levels = np.linspace(np.nanmin(b), np.nanmax(b), 64)
    delta_levels = np.linspace(-np.pi, np.pi, 64)

    fig = plt.figure(figsize=(2.2 * 3.375, 1.1 * 3.375))#, constrained_layout=True)
    widths = [1, 0.05, 1, 0.05, 1.5]
    heights = [0.05, 1, 1, 0.05, 1, 1]
    gs = fig.add_gridspec(ncols=5, nrows=6, width_ratios=widths, height_ratios=heights,
     left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

    # UPPER LEFT
    ax1 = fig.add_subplot(gs[1:3, 0])
    ax1.set_aspect("equal")
    im = ax1.contourf(x, y, psi_abs_1[n], levels=psi_abs_1_levels, cmap="viridis_r", zorder=-10)
    ax1.set_rasterization_zorder(0)
    
    cax1 = fig.add_subplot(gs[1:3, 1])
    cbar = fig.colorbar(
        im, cax=cax1, format="%1.1f", ticks=np.linspace(0, np.nanmax(psi_abs_1), 3)
    )

    # U  = uniform_filter(j_x_1[n], delta)[::delta,::delta]
    # V  = uniform_filter(j_y_1[n], delta)[::delta,::delta]

    # U[np.abs(U) < 1e-5] = np.nan
    # V[np.abs(V) < 1e-5] = np.nan
    # ax1.quiver(X, Y, U, V, scale=50*np.max(np.sqrt(U**2+V**2)), color=[1,1,1,0.5])

    #ax1.set_title(r"$|\psi_1|$")

    # UPPER RIGHT
    ax2 = fig.add_subplot(gs[1:3, 2])
    ax2.set_aspect("equal")
    im = ax2.contourf(x, y, psi_abs_2[n], levels=psi_abs_2_levels, cmap="viridis_r", zorder=-10)
    ax2.set_rasterization_zorder(0)
    divider = make_axes_locatable(ax2)
    cax2 = fig.add_subplot(gs[1:3, 3])
    cbar = fig.colorbar(
        im, cax=cax2, format="%1.1f", ticks=np.linspace(0, np.nanmax(psi_abs_2), 3)
    )

    # U  = uniform_filter(j_x_2[n], delta)[::delta,::delta]
    # V  = uniform_filter(j_y_2[n], delta)[::delta,::delta]

    # U[np.abs(U) < 1e-5] = np.nan
    # V[np.abs(V) < 1e-5] = np.nan
    # ax2.quiver(X, Y, U, V, scale=50*np.max(np.sqrt(U**2+V**2)), color=[1,1,1,0.5])

    #ax2.set_title(r"$|\psi_2|$")

    # LOWER LEFT
    ax3 = fig.add_subplot(gs[4:6, 0])
    ax3.set_aspect("equal")
    im = ax3.contourf(x, y, psi_delta[n], levels=delta_levels, cmap="twilight", zorder=-10)
    ax3.set_rasterization_zorder(0)
    divider = make_axes_locatable(ax3)
    cax3 = fig.add_subplot(gs[4:6, 1])
    cbar = fig.colorbar(
        im, cax=cax3, format="%1.1f", ticks=np.linspace(-np.pi, np.pi, 3)
    )
    #ax3.set_title(r"$\Delta \theta_{12}$")

    # LOWER RIGHT
    ax4 = fig.add_subplot(gs[4:6, 2])
    ax4.set_aspect("equal")
    im = ax4.contourf(x, y, b[n], levels=b_levels, cmap="jet", zorder=-10)
    ax4.set_rasterization_zorder(0)
    divider = make_axes_locatable(ax4)
    cax4 = fig.add_subplot(gs[4:6, 3])
    cbar = fig.colorbar(
        im, cax=cax4, format="%1.1f", ticks=np.linspace(np.nanmin(b), np.nanmax(b), 3)
    )
    #ax4.set_title(r"$B_z$")

    # RIGHT
    ax5 = fig.add_subplot(gs[2:5, 4])
    ax5.plot(s, fenergy, ".-")
    ax5.scatter(s[n], fenergy[n], color="C3")
    deltaf = np.max(fenergy) - np.min(fenergy)
    ax5.set_xlim(-0.1, 1.1)
    ax5.set_ylim(np.min(fenergy) - 0.05 * deltaf, np.max(fenergy) + 0.05 * deltaf)
    ax5.set_xlabel(r"s")
    ax5.set_ylabel(r"F(s)")
    ax5.set_title(r"Minimum Energy Path")

    #gs.tight_layout(fig, rect=[0, 0, 1, 1], h_pad=0.005, w_pad=0.005)

    return fig


def complex_plot(x, y, psi_abs, psi_theta):

    N = psi_abs.shape[0]
    M = psi_abs.shape[1]
    hsb = np.zeros((N, M, 3))

    hsb[:, :, 0] = (psi_theta.T / 3.14 + 1) / 2.0
    hsb[:, :, 1] = (1.03 ** 6 - psi_abs.T ** 6) ** (1.0 / 6.0)
    hsb[:, :, 2] = psi_abs.T

    rgb = colors.hsv_to_rgb(hsb.clip(0, 1))

    rgba = np.ones((N, M, 4))
    rgba[:, :, 0:3] = rgb
    nan_mask = np.nan_to_num(psi_abs.T * 0 + 1)
    rgba[:, :, 3] = nan_mask

    fig, ax = plt.subplots(dpi=300)
    ax.imshow(
        rgba,
        origin="lower",
        interpolation="nearest",
        extent=[x.min(), x.max(), y.min(), y.max()],
    )

    fig.tight_layout()

    return fig, ax


def gl_plot(
    n,
    x,
    y,
    fenergy,
    s,
    psi_abs_1,
    psi_theta_1,
    j_x_1,
    j_y_1,
    b,
):

    nx, ny = x.shape
    delta = int(nx / 16)
    X = x[::delta, ::delta]
    Y = y[::delta, ::delta]

    psi_abs_1_levels = np.linspace(0, np.nanmax(psi_abs_1), 64)
    b_levels = np.linspace(-1, 1, 64) * np.max(np.abs([np.nanmax(b), np.nanmin(b)]))
    delta_levels = np.linspace(-np.pi, np.pi, 64)

    fig = plt.figure(figsize=(2.2 * 3.375, 1.1 * 3.375))
    widths = [1, 0.05, 1, 0.05, 1.5]
    heights = [0.05, 1, 1, 0.05, 1, 1]
    gs = fig.add_gridspec(ncols=5, nrows=6, width_ratios=widths, height_ratios=heights,
     left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

    # UPPER LEFT
    ax1 = fig.add_subplot(gs[1:3, 0])
    ax1.set_aspect("equal")
    im = ax1.contourf(x, y, psi_abs_1[n], levels=psi_abs_1_levels, cmap="viridis_r", zorder=-10)
    ax1.set_rasterization_zorder(0)
    
    cax1 = fig.add_subplot(gs[1:3, 1])
    cbar = fig.colorbar(
        im, cax=cax1, format="%1.1f", ticks=np.linspace(0, np.nanmax(psi_abs_1), 3)
    )

    # U  = uniform_filter(j_x_1[n], delta)[::delta,::delta]
    # V  = uniform_filter(j_y_1[n], delta)[::delta,::delta]

    # U[np.abs(U) < 1e-5] = np.nan
    # V[np.abs(V) < 1e-5] = np.nan
    # ax1.quiver(X, Y, U, V, scale=50*np.max(np.sqrt(U**2+V**2)), color=[1,1,1,0.5])

    #ax1.set_title(r"$|\psi_1|$")

    # UPPER RIGHT
    # ax2 = fig.add_subplot(gs[1:3, 2])
    # ax2.set_aspect("equal")
    # im = ax2.contourf(x, y, psi_abs_2[n], levels=psi_abs_2_levels, cmap="viridis_r", zorder=-10)
    # ax2.set_rasterization_zorder(0)
    # divider = make_axes_locatable(ax2)
    # cax2 = fig.add_subplot(gs[1:3, 3])
    # cbar = fig.colorbar(
    #     im, cax=cax2, format="%1.1f", ticks=np.linspace(0, np.nanmax(psi_abs_2), 3)
    # )

    # U  = uniform_filter(j_x_2[n], delta)[::delta,::delta]
    # V  = uniform_filter(j_y_2[n], delta)[::delta,::delta]

    # U[np.abs(U) < 1e-5] = np.nan
    # V[np.abs(V) < 1e-5] = np.nan
    # ax2.quiver(X, Y, U, V, scale=50*np.max(np.sqrt(U**2+V**2)), color=[1,1,1,0.5])

    #ax2.set_title(r"$|\psi_2|$")

    # LOWER LEFT
    # ax3 = fig.add_subplot(gs[4:6, 0])
    # ax3.set_aspect("equal")
    # im = ax3.contourf(x, y, psi_delta[n], levels=delta_levels, cmap="twilight", zorder=-10)
    # ax3.set_rasterization_zorder(0)
    # divider = make_axes_locatable(ax3)
    # cax3 = fig.add_subplot(gs[4:6, 1])
    # cbar = fig.colorbar(
    #     im, cax=cax3, format="%1.1f", ticks=np.linspace(-np.pi, np.pi, 3)
    # )
    #ax3.set_title(r"$\Delta \theta_{12}$")

    # LOWER RIGHT
    ax4 = fig.add_subplot(gs[4:6, 2])
    ax4.set_aspect("equal")
    im = ax4.contourf(x, y, b[n], levels=b_levels, cmap="jet", zorder=-10)
    ax4.set_rasterization_zorder(0)
    divider = make_axes_locatable(ax4)
    cax4 = fig.add_subplot(gs[4:6, 3])
    cbar = fig.colorbar(
        im, cax=cax4, format="%1.1f", ticks=np.linspace(np.nanmin(b), np.nanmax(b), 3)
    )
    #ax4.set_title(r"$B_z$")

    # RIGHT
    ax5 = fig.add_subplot(gs[2:5, 4])
    ax5.plot(s, fenergy, ".-")
    ax5.scatter(s[n], fenergy[n], color="C3")
    deltaf = np.max(fenergy) - np.min(fenergy)
    ax5.set_xlim(-0.1, 1.1)
    ax5.set_ylim(np.min(fenergy) - 0.05 * deltaf, np.max(fenergy) + 0.05 * deltaf)
    ax5.set_xlabel(r"s")
    ax5.set_ylabel(r"F(s)")
    ax5.set_title(r"Minimum Energy Path")

    #gs.tight_layout(fig, rect=[0, 0, 1, 1], h_pad=0.005, w_pad=0.005)

    return fig


def complex_plot(x, y, psi_abs, psi_theta):

    N = psi_abs.shape[0]
    M = psi_abs.shape[1]
    hsb = np.zeros((N, M, 3))

    hsb[:, :, 0] = (psi_theta.T / 3.14 + 1) / 2.0
    hsb[:, :, 1] = (1.03 ** 6 - psi_abs.T ** 6) ** (1.0 / 6.0)
    hsb[:, :, 2] = psi_abs.T

    rgb = colors.hsv_to_rgb(hsb.clip(0, 1))

    rgba = np.ones((N, M, 4))
    rgba[:, :, 0:3] = rgb
    nan_mask = np.nan_to_num(psi_abs.T * 0 + 1)
    rgba[:, :, 3] = nan_mask

    fig, ax = plt.subplots(dpi=300)
    ax.imshow(
        rgba,
        origin="lower",
        interpolation="nearest",
        extent=[x.min(), x.max(), y.min(), y.max()],
    )

    fig.tight_layout()

    return fig, ax


def complex_plot_3D(xx, yy, psi_abs, psi_theta):
    fig = plt.figure(dpi=300)
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    theta_deg = (psi_theta / np.pi + 1) / 2
    # cmap = phase(theta_deg)
    cmap = cm.hsv(theta_deg)
    ax.plot_surface(xx, yy, psi_abs, facecolors=cmap)
    ax.set_zlim([0, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def vector_potential_plot(x, y, ax, ay, a_max, a_min, title=""):
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=False, dpi=300, figsize=(11, 7)
    )

    fig.suptitle(title, fontsize="x-large")

    im = axes.flat[0].set_title(r"$A_x$")
    im = axes.flat[1].set_title(r"$A_y$")

    im = axes.flat[0].set_aspect("equal")
    im = axes.flat[1].set_aspect("equal")

    levels = np.linspace(a_min, a_max, 32)
    im = axes.flat[0].contourf(x, y, ax, levels=levels)
    im = axes.flat[1].contourf(x, y, ay, levels=levels)

    cax, kw = colorbar.make_axes([ax for ax in axes.flat], orientation="horizontal")
    plt.colorbar(im, cax=cax, **kw)

    fig.tight_layout()

    return fig, ax


def magnetic_field_plot(x, y, b, b_max, b_min):

    levels = np.linspace(b_min, b_max, 40)

    fig, ax = plt.subplots(dpi=300)

    im = ax.contourf(x, y, b, levels=levels, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)

    fig.colorbar(im, cax=cax, format="%1.2f")
    fig.tight_layout()

    return fig, ax


def magnetic_field_plot_3D(x, y, b, b_max, b_min):
    fig = plt.figure(dpi=300)

    _x = x[:-1, :-1]
    _y = y[:-1, :-1]
    _b = b[:-1, :-1]

    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ax.plot_surface(_x, _y, _b)
    ax.set_zlim([b_min, b_max])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax


def current_plot(x, y, j_x, j_y, scale=None, borders=None):

    nx, ny = x.shape

    delta = int(nx / 50)

    X = x[::delta, ::delta]
    Y = y[::delta, ::delta]

    U = uniform_filter(j_x, delta)[::delta, ::delta]
    V = uniform_filter(j_y, delta)[::delta, ::delta]

    fig, ax = plt.subplots(dpi=300)
    if scale:
        ax.quiver(X, Y, U, V, scale=scale)
    else:
        ax.quiver(X, Y, U, V)

    if borders is not None:
        idx_x, idx_y = np.argwhere(borders == 1).T
        xx = x[idx_x, idx_y]
        yy = y[idx_x, idx_y]
        ax.scatter(xx, yy, 0.05, "r")

    fig.tight_layout()

    return fig, ax


################################## STATIC PLOT MULTICOMPONENT ##################################

def pseudospin_plot(x, y, psi_abs_1, psi_abs_2, theta_12):
    """
    Plot the pseudospin for a multicomponent simulation.
    """

    n = psi_abs_1**2 + psi_abs_2**2
    Sx = psi_abs_1*psi_abs_2/n * np.cos(theta_12)
    Sy = psi_abs_1*psi_abs_2/n * np.sin(theta_12)
    Sz = (psi_abs_1**2 - psi_abs_2**2)/n

    fig, ax = plt.subplots()
    
    nx, ny = x.shape

    delta = int(nx // 50)

    X = uniform_filter(x, delta)[::delta, ::delta]
    Y = uniform_filter(y, delta)[::delta, ::delta]
    U = uniform_filter(Sx, delta)[::delta, ::delta]
    V = uniform_filter(Sy, delta)[::delta, ::delta]
    W = uniform_filter(Sz, delta)[::delta, ::delta]

    norm = colors.Normalize(vmin=-1.0, vmax=1.0)

    im = ax.quiver(X, Y, U, V, W, scale=10, pivot='mid', cmap = 'turbo', norm=norm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(
        im, cax=cax, format="%1.1f", ticks=np.linspace(-1, 1, 3), label=r"$S_z$"
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    
    ax.set_xlim( 1, 5)
    ax.set_ylim(-2, 2)

    fig.tight_layout()

    return fig, ax


def pseudospin_plot_3D(x, y, psi_abs_1, psi_abs_2, theta_12):
    """
    Plot the pseudospin for a multicomponent simulation, 3D version.
    """

    n = psi_abs_1**2 + psi_abs_2**2
    Sx = psi_abs_1*psi_abs_2/n * np.cos(theta_12)
    Sy = psi_abs_1*psi_abs_2/n * np.sin(theta_12)
    Sz = (psi_abs_1**2 - psi_abs_2**2)/n

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    
    nx, ny = x.shape

    delta = int(nx / 16)

    X = x[nx//2::delta, ny//4:-ny//4:delta]
    Y = y[nx//2::delta, ny//4:-ny//4:delta]

    U = uniform_filter(Sx, delta)[nx//2::delta, ny//4:-ny//4:delta]
    V = uniform_filter(Sy, delta)[nx//2::delta, ny//4:-ny//4:delta]
    W = uniform_filter(Sz, delta)[nx//2::delta, ny//4:-ny//4:delta]

    L = np.sqrt(U**2 + V**2 + W**2)

    ax.quiver(X, Y, 0, U, V, W)
    ax.set_zlim([-2, 2])

    fig.tight_layout()

    return fig, ax


####################################### ANIMATIONS #######################################


def current_animation(F, x, y, j_x, j_y, scale, borders=None):

    nx, ny = x.shape

    delta = int(nx / 50)

    X = x[::delta, ::delta]
    Y = y[::delta, ::delta]

    U = j_x[:, ::delta, ::delta]
    V = j_y[:, ::delta, ::delta]

    fig = plt.figure(dpi=300, figsize=(6, 6))
    ax = plt.axes(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    qv = ax.quiver(X, Y, U[0], V[0])
    ax.set_title("0")

    if borders is not None:
        idx_x, idx_y = np.argwhere(borders == 1).T
        xx = x[idx_x, idx_y]
        yy = y[idx_x, idx_y]
        ax.scatter(xx, yy, 0.05, "r")

    def update_quiver(n):

        qv.set_UVC(U[n], V[n])
        ax.set_title(f"{n}")
        return (qv,)

    anim = animation.FuncAnimation(
        fig, update_quiver, frames=F, interval=250, blit=False
    )
    fig.tight_layout()

    return anim


def magnetic_field_animation(F, x, y, b, borders=None):

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    b_max = b.max()
    b_min = b.min()

    levels = np.linspace(b_min, b_max, 40)

    im = ax.contourf(x, y, b[0], levels=levels, cmap="jet")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])

    ax.set_title("0")

    plt.colorbar(im)

    if borders is not None:
        idx_x, idx_y = np.argwhere(borders == 1).T
        xx = x[idx_x, idx_y]
        yy = y[idx_x, idx_y]
        ax.scatter(xx, yy, 0.05, "k")

    def update_contourf(n):

        ax.contourf(x, y, b[n], levels=levels, cmap="jet")
        ax.set_title(f"{n}")

        return (im,)

    anim = animation.FuncAnimation(
        fig, update_contourf, frames=F, interval=250, blit=False
    )
    fig.tight_layout()

    return anim


def complex_field_animation(F, x, y, psi_abs, psi_theta, borders=None):

    N = psi_abs.shape[1]
    M = psi_abs.shape[2]
    hsb = np.zeros((F, N, M, 3))

    hsb[:, :, :, 0] = (psi_theta.transpose((0, 2, 1)) / 3.14 + 1) / 2.0
    hsb[:, :, :, 1] = (1.02 ** 6 - psi_abs.transpose((0, 2, 1)) ** 6) ** (1.0 / 6.0)
    hsb[:, :, :, 2] = psi_abs.transpose((0, 2, 1))

    rgb = colors.hsv_to_rgb(hsb.clip(0, 1))

    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(
        rgb[0],
        origin="lower",
        interpolation="lanczos",
        extent=[x.min(), x.max(), y.min(), y.max()],
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])

    ax.set_title("0")

    if borders is not None:
        idx_x, idx_y = np.argwhere(borders == 1).T
        xx = x[idx_x, idx_y]
        yy = y[idx_x, idx_y]
        ax.scatter(xx, yy, 0.05, "r")

    def update_cplot(n):
        im.set_data(rgb[n])
        ax.set_title(f"{n}")
        fig.canvas.draw()
        return (im,)

    anim = animation.FuncAnimation(
        fig, update_cplot, frames=F, interval=250, blit=False
    )
    fig.tight_layout()

    return anim

