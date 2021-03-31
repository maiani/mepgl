#ifndef INTERPOLATION_CUH
#define INTERPOLATION_CUH

#include <cusparse_v2.h>

__host__
void arcLength(int F, int N, real*s, real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2);

__host__
void arcLengthJB(int F, int N, real*s_jb, real*b, real2 *j_1, real2 *j_2);

__host__
void arcLengthAbsGrad(int N, int F, real*s_ag, real*dev_psi_1_abs, real2 *dev_cd_1,
                                                 real*dev_psi_2_abs, real2 *dev_cd_2);
                                                 
__host__
void redistributeLinear(cusparseHandle_t handle, int N, 
                        int F, real*s, real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
                        int F_new, real2 *dev_a_new, real2 *dev_psi_1_new, real2 *dev_psi_2_new);

__host__
void redistributeCubic(cusparseHandle_t handle, int N, 
                        int F, real*s, real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
                        int F_new, real2 *dev_a_new, real2 *dev_psi_1_new, real2 *dev_psi_2_new);

#endif