"""
Initializing code for mepgl.
"""

import os
import shutil
import subprocess

import numpy as np

## Header functions


def generate_headers(
    N,
    dx,
    default_relaxation_step_number,
    multicomponent,
    thin_film=False,
    double_precision=False,
    dev_number=0,
):
    """
    Generate the headers.
    """

    new_headers = False

    # Real.cuh header

    doubl_precision_str = "" if double_precision else "//"
    real_cuh = (
        f"""#ifndef REAL_CUH                                \n"""
        f"""#define REAL_CUH                                \n"""
        f"""                                                \n"""
        f"""{doubl_precision_str}#define DOUBLE_PRECISION   \n"""
        f"""                                                \n"""
        f"""// Type definition                              \n"""
        f"""#ifdef DOUBLE_PRECISION                         \n"""
        f"""                                                \n"""
        f"""typedef double real;                            \n"""
        f"""                                                \n"""
        f"""#ifdef __CUDACC__                               \n"""
        f"""typedef double2 real2;                          \n"""
        f"""typedef double4 real4;                          \n"""
        f"""#endif                                          \n"""
        f"""                                                \n"""
        f"""#else                                           \n"""
        f"""                                                \n"""
        f"""typedef float real;                             \n"""
        f"""                                                \n"""
        f"""#ifdef __CUDACC__                               \n"""
        f"""typedef float2 real2;                           \n"""
        f"""typedef float4 real4;                           \n"""
        f"""#endif                                          \n"""
        f"""                                                \n"""
        f"""#endif                                          \n"""
        f"""                                                \n"""
        f"""#endif // REAL_CUH                              \n"""
    )

    old_real_cuh = ""
    if os.path.exists("./src/real.cuh"):
        with open("./src/real.cuh", "r") as real_cuh_file:
            old_real_cuh = real_cuh_file.read()

    if old_real_cuh != real_cuh:
        with open("./src/real.cuh", "w+") as real_cuh_file:
            real_cuh_file.write(real_cuh)
        new_headers = True

    # config.cuh header
    multicomponent_str = "" if multicomponent else "//"
    thin_film_str = "" if thin_film else "//"
    config_cuh = (
        f"""#ifndef CONFIG_CUH                                                                      \n"""
        f"""#define CONFIG_CUH                                                                      \n"""
        f"""/*****************************                                                          \n"""
        f""" * Autogenerated by config.py                                                           \n"""
        f""" *                                                                                      \n"""
        f""" * DO NOT MODIFY!                                                                       \n"""
        f""" ****************************/                                                          \n"""
        f"""                                                                                        \n"""
        f"""#include <string>                                                                       \n"""
        f"""                                                                                        \n"""
        f"""#include "real.cuh"                                                                     \n"""
        f"""                                                                                        \n"""
        f"""// Device number                                                                        \n"""
        f"""const int devNumber = {dev_number};                                                     \n"""
        f"""                                                                                        \n"""
        f"""// Multicomponent                                                                       \n"""
        f"""{multicomponent_str}#define MULTICOMPONENT                                              \n"""
        f"""                                                                                        \n"""
        f""" // Neglect self field (for thin films)                                                 \n"""
        f"""{thin_film_str}#define NO_SELF_FIELD                                                    \n"""
        f"""                                                                                        \n"""
        f"""// Number of point for side                                                             \n"""
        f"""constexpr int N = {N};                                                                  \n"""
        f"""                                                                                        \n"""
        f"""#ifdef DOUBLE_PRECISION                                                                 \n"""
        f"""constexpr int blockSizeX = 512;                                                         \n"""
        f"""#else                                                                                   \n"""
        f"""constexpr int blockSizeX = 1024;                                                        \n"""
        f"""#endif                                                                                  \n"""
        f"""                                                                                        \n"""
        f"""constexpr int gridSizeX = (int)((N*N)/blockSizeX + 1);                                  \n"""
        f"""constexpr real dx = {dx};                                                               \n"""
        f"""                                                                                        \n"""
        f"""constexpr int default_relaxation_step_number = {default_relaxation_step_number};        \n"""
        f"""                                                                                        \n"""
        f"""#endif // CONFIG_CUH                                                                    \n"""
    )

    old_config_cuh = ""
    if os.path.exists("./src/config.cuh"):
        with open("./src/config.cuh", "r") as config_cuh_file:
            old_config_cuh = config_cuh_file.read()

    if old_config_cuh != config_cuh:
        with open("./src/config.cuh", "w+") as config_cuh_file:
            config_cuh_file.write(config_cuh)
        new_headers = True

    return new_headers


# Launcher functions


def generate_launcher(simulation_name, F, iterations, modes):
    """
    Generate a bash launcher script (legacy).
    """

    P = F.shape[0]

    bashfile = "#!/bin/bash\n"

    for i in range(P - 1):
        command_string = f"./mepgl --simname {simulation_name} --mode {modes[i]} --Fin {F[i]} --Fout {F[i+1]} --iterations {iterations[i]}\n ./reload.py \n"
        bashfile += command_string

    command_string = f"./mepgl --simname {simulation_name} --mode {modes[-1]} --Fin {F[-1]} --Fout {F[-1]}  --iterations {iterations[-1]}\n"
    bashfile += command_string

    launcher_file = open("./launcher.sh", "w+")
    launcher_file.write(bashfile)
    launcher_file.close()


def launch_simulation(simulation_name, F, iterations, modes):
    """
    Launch the mepgl solver.
    """

    P = F.shape[0]
    for i in range(P - 1):
        subprocess.run(
            [
                "./mepgl",
                "--simname",
                simulation_name,
                "--mode",
                modes[i],
                "--Fin",
                F[i],
                "--Fout",
                F[i + 1],
                "--iterations",
                iterations[i],
            ]
        )
        subprocess.run(["python", "./reload.py"])

    subprocess.run(
        [
            "./mepgl",
            "--simname",
            simulation_name,
            "--mode",
            f"{modes[-1]}",
            "--Fin",
            f"{F[-1]}",
            "--Fout",
            f"{F[-1]}",
            "--iterations",
            f"{iterations[-1]}",
        ]
    )


# Init files


def generate_init(
    F,
    a_1,
    a_2,
    ax,
    ay,
    b_1,
    b_2,
    comp_domain,
    delta,
    eta,
    gamma,
    h,
    m_xx_1,
    m_xx_2,
    m_yy_1,
    m_yy_2,
    multicomponent,
    q_1,
    q_2,
    sc_domain,
    simulation_name,
    u_1,
    u_2,
    v_1,
    v_2,
    x,
    y,
    double_precision=False,
):

    if double_precision:
        data_type = np.float64
    else:
        data_type = np.float32

    sim_dir_name = f"./simulations/{simulation_name}/"
    os.makedirs(sim_dir_name, exist_ok=True)

    input_dir = sim_dir_name + "input_data/"

    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)

    np.save(input_dir + f"x.npy", x.astype(data_type))
    np.save(input_dir + f"y.npy", y.astype(data_type))
    np.save(input_dir + f"comp_domain.npy", comp_domain.astype(np.int32))
    np.save(input_dir + f"sc_domain.npy", sc_domain.astype(np.int32))
    np.save(input_dir + f"a_1.npy", a_1.astype(data_type))
    np.save(input_dir + f"b_1.npy", b_1.astype(data_type))
    np.save(input_dir + f"m_xx_1.npy", m_xx_1.astype(data_type))
    np.save(input_dir + f"m_yy_1.npy", m_yy_1.astype(data_type))
    np.save(input_dir + f"a_2.npy", a_2.astype(data_type))
    np.save(input_dir + f"b_2.npy", b_2.astype(data_type))
    np.save(input_dir + f"m_xx_2.npy", m_xx_2.astype(data_type))
    np.save(input_dir + f"m_yy_2.npy", m_yy_2.astype(data_type))
    np.save(input_dir + f"h.npy", h.astype(data_type))

    q = np.array([q_1, q_2, eta, gamma, delta])
    np.save(input_dir + f"q.npy", q.astype(data_type))

    for n in range(F[0]):
        os.makedirs(input_dir + f"{n}/", exist_ok=True)
        np.save(input_dir + f"{n}/ax.npy", ax[n].astype(data_type))
        np.save(input_dir + f"{n}/ay.npy", ay[n].astype(data_type))
        np.save(input_dir + f"{n}/u_1.npy", u_1[n].astype(data_type))
        np.save(input_dir + f"{n}/v_1.npy", v_1[n].astype(data_type))

        if multicomponent:
            np.save(input_dir + f"{n}/u_2.npy", u_2[n].astype(data_type))
            np.save(input_dir + f"{n}/v_2.npy", v_2[n].astype(data_type))
