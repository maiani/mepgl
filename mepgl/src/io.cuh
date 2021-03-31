#ifndef IO_CUH
#define IO_CUH

#include <string>

#include "real.cuh"

void loadInput(std::string simulation_name, unsigned int F, unsigned int N, real2 *r, int* comp_domain, int* sc_domain, 
                real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, real *h,
                real *q_1, real *q_2, real *eta, real *gamma, real *delta,
                real2 *psi_1, real2 *psi_2, real2 *a);

void saveOutput(std::string simulation_name, unsigned int F, unsigned int N, real2 *r, int* comp_domain, int* sc_domain, 
                real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, real *h,
                real *q_1, real *q_2, real *eta, real *gamma, real *delta,
                real2 *psi_1, real2 *psi_2, real2 *a,
                real2 *j_1, real2 *j_2, real *b,
                real *fenergy, real *s, real *s_jb);

void saveStats(std::string simulation_name, unsigned int F, unsigned int itermax, real *fenergy, real *g_norm2);

#endif