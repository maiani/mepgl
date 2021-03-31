#ifndef GL_NLCG_CUH
#define GL_NLCG_CUH

/***************************************************************
 *
 * Here there are all the device kernels and host functions
 * needed to implemente Nonlinear Conjugate Gradient Descent
 * for a Ginzburg Landau system
 *
 ****************************************************************/

#include "real.cuh"
 
__global__ 
void buildNearestNeighbourMap(int *sc_domain, int *nn_map);


 __host__
void computeFreeEnergy( int F, int *dev_comp_nn_map, int *dev_sc_nn_map,
                        real *dev_a_1, real *dev_b_1, real *dev_m_xx_1, real *dev_m_yy_1, 
                        real *dev_a_2, real *dev_b_2, real *dev_m_xx_2, real *dev_m_yy_2, 
                        real *dev_h, real *dev_q_1, real *dev_q_2, real *dev_eta, real *dev_gamma, real *dev_delta,
                        real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
                        real* dev_fenergy);

__host__
void computeJB(int F, int *dev_comp_nn_map, int *dev_sc_nn_map,
               real *dev_a_1, real *dev_b_1, real *dev_m_xx_1, real *dev_m_yy_1, 
               real *dev_a_2, real *dev_b_2, real *dev_m_xx_2, real *dev_m_yy_2, 
               real *dev_h, real *dev_q_1, real *dev_q_2,
               real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
               real* dev_b, real2 *dev_j_1, real2 *dev_j_2);

__host__
void computeAbsGrad(int F, int *dev_sc_nn_map,
                    real *dev_q_1, real *dev_q_2,
                    real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
                    real* dev_psi_1_abs, real2 *dev_cd_1,
                    real* dev_psi_2_abs, real2 *dev_cd_2);

__host__
 void nlcgSteps(int iter_n, int F, int *dev_comp_nn_map, int* dev_sc_nn_map,
                real *dev_a_1, real *dev_b_1, real *dev_m_xx_1, real *dev_m_yy_1,
                real *dev_a_2, real *dev_b_2, real *dev_m_xx_2, real *dev_m_yy_2,
                real *dev_h, real *dev_q_1, real *dev_q_2, real *dev_eta, real *dev_gamma, real *dev_delta,
                real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2, 
                real2 *dev_a_g, real2 *dev_psi_1_g, real2 *dev_psi_2_g, 
                real2 *dev_a_g_old, real2 *dev_psi_1_g_old, real2 *dev_psi_2_g_old, 
                real2 *dev_a_d, real2 *dev_psi_1_d, real2 *dev_psi_2_d,
                real* dev_g_norm2_density, real* dev_g_delta_g_density,
                real *g_norm2, real *g_norm2_old, real *g_delta_g, real *alpha,
                real *dev_fenergy_density, real4 *dev_poly_coeff_density, real *fenergy, real4 *poly_coeff,
                bool nlcg = true, bool maxwell_solver = false
    );

#endif