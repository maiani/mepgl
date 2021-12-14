/***************************************************************
 *
 * Here there are all the device kernels and host functions
 * needed to implemente Nonlinear Conjugate Gradient Descent
 * for a Ginzburg Landau system
 *
 ****************************************************************/
#include <cmath>

#include "cuda_errors.cuh"

#include "real.cuh"
#include "parallel_sum.cuh"
 
#include "config.cuh"

// Define the formula for beta:
// 0 -> Stepest descent without conjugate gradient
// 1 -> Polak–Ribière
// 2 -> Fletcher–Reeves
#define BETA 1

// If defined turn off linesearch and use the constant step length given
// #define ALPHA 1e-1

// Enable bilinear bilinear Josephson coupling
#define BILINEAR_JOSEPHSON_COUPLING

// Enable biquadratic density coupling
#define DENSITY_COUPLING

// Enable biquadratic Josephson coupling
#define BIQUADRATIC_JOSEPHSON_COUPLING

// Neglect self field (for thin field)
#define NO_SELF_FIELD

// DEBUG
// #define NO_NL
// #define NO_EM
// #define NO_KINETIC

/***************************** Functions needed to compute covariant derivatives *********************************************/

__device__ inline real sqr(real x) { return x*x; }
__device__ inline int int2bit(int x, int n){ return (x >> n) & 1; }

__global__ void buildNearestNeighbourMap(int *domain, int *nn_map){
    
    /* 
     * Build nearest neighbour map, the direction/bit position map is the following
     *
     *    3 2 1 
     *    4 X 0
     *    5 6 7
     *
     */

    int idx =  blockDim.x*blockIdx.x + threadIdx.x;

    if(idx >= N*N){
        return;
    }

    int nn = 0;

    if(domain[idx]!=0){
        
        if((idx % N != N-1))
            if(domain[idx+1  ]!=0)
                nn += 1 << 0;

        if((idx % N != N-1)&&(idx / N != N-1))
            if(domain[idx+1+N]!=0)
                nn += 1 << 1;

        if((idx / N != N-1))
            if(domain[idx+N  ]!=0)
                nn += 1 << 2;
        
        if((idx / N != N-1)&&(idx % N != 0))
            if(domain[idx-1+N]!=0)
                nn += 1 << 3;

        if((idx % N != 0))
            if(domain[idx-1  ]!=0)
                nn += 1 << 4;
        
        if((idx % N != 0)&&(idx / N != 0))
                if(domain[idx-1-N]!=0)
            nn += 1 << 5;
        
        if((idx / N != 0))
            if(domain[idx-N  ]!=0)
                nn += 1 << 6;

        if((idx / N != 0)&&(idx % N != N-1))
            if(domain[idx-N+1]!=0)
            nn += 1 << 7;
        }

        nn_map[idx] = nn;
}

__global__
void freeEnergyDensity( int* comp_nn_map, int* sc_nn_map,
                        real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, 
                        real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, 
                        real *h, real *q_1, real *q_2, real *eta, real *gamma, real *delta,
                        real2 *a, real2 *psi_1, real2 *psi_2,
                        real* fenergy_density){

    int idx_0 =  blockDim.x*blockIdx.x + threadIdx.x;

    if(idx_0 >= N*N){
        return;
    }

    int idx = idx_0 + N*N*blockIdx.y;

    real nl_term = b_1[idx_0]/2.0*sqr(a_1[idx_0]/b_1[idx_0] + sqr(psi_1[idx].x) + sqr(psi_1[idx].y));
    #ifdef MULTICOMPONENT

        nl_term += b_2[idx_0]/2.0*sqr(a_2[idx_0]/b_2[idx_0] + sqr(psi_2[idx].x) + sqr(psi_2[idx].y));
        
        #ifdef BILINEAR_JOSEPHSON_COUPLING
        nl_term += eta[0] * (psi_1[idx].x*psi_2[idx].x + psi_1[idx].y*psi_2[idx].y);
        #endif

        #ifdef DENSITY_COUPLING
        nl_term += gamma[0]/2.0 * (sqr(psi_1[idx].x) + sqr(psi_1[idx].y)) * (sqr(psi_2[idx].x) + sqr(psi_2[idx].y));
        #endif

        #ifdef BIQUADRATIC_JOSEPHSON_COUPLING
        nl_term += delta[0]/2.0 * (+ sqr(psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y) 
                                   - sqr(psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x) );
        #endif
    #endif

    nl_term *= ( int2bit(sc_nn_map[idx_0], 0)*int2bit(sc_nn_map[idx_0], 1)*int2bit(sc_nn_map[idx_0], 2) + 
                 int2bit(sc_nn_map[idx_0], 2)*int2bit(sc_nn_map[idx_0], 3)*int2bit(sc_nn_map[idx_0], 4) + 
                 int2bit(sc_nn_map[idx_0], 4)*int2bit(sc_nn_map[idx_0], 5)*int2bit(sc_nn_map[idx_0], 6) + 
                 int2bit(sc_nn_map[idx_0], 6)*int2bit(sc_nn_map[idx_0], 7)*int2bit(sc_nn_map[idx_0], 0))/4.0f;
                 
    real kinetic_term_x = 0;
    real kinetic_term_y = 0;

    if(int2bit(sc_nn_map[idx_0], 0)){
        kinetic_term_x += 0.25/m_xx_1[idx_0]*(
                        sqr(-(psi_1[idx + 1].x - psi_1[idx].x)/dx - q_1[0]*a[idx    ].x*psi_1[idx    ].y) +
                        sqr(-(psi_1[idx + 1].x - psi_1[idx].x)/dx - q_1[0]*a[idx + 1].x*psi_1[idx + 1].y) +

                        sqr(+(psi_1[idx + 1].y - psi_1[idx].y)/dx - q_1[0]*a[idx    ].x*psi_1[idx    ].x) + 
                        sqr(+(psi_1[idx + 1].y - psi_1[idx].y)/dx - q_1[0]*a[idx + 1].x*psi_1[idx + 1].x)
        );
        #ifdef MULTICOMPONENT
        kinetic_term_x += 0.25/m_xx_2[idx_0]*(
            sqr(-(psi_2[idx + 1].x - psi_2[idx].x)/dx - q_2[0]*a[idx    ].x*psi_2[idx    ].y) +
            sqr(-(psi_2[idx + 1].x - psi_2[idx].x)/dx - q_2[0]*a[idx + 1].x*psi_2[idx + 1].y) +

            sqr(+(psi_2[idx + 1].y - psi_2[idx].y)/dx - q_2[0]*a[idx    ].x*psi_2[idx    ].x) + 
            sqr(+(psi_2[idx + 1].y - psi_2[idx].y)/dx - q_2[0]*a[idx + 1].x*psi_2[idx + 1].x)
        );
        #endif
    }
        
    if(int2bit(sc_nn_map[idx_0], 2)){
        kinetic_term_y += 0.25/m_yy_1[idx_0]*(
                        sqr(-(psi_1[idx + N].x - psi_1[idx].x)/dx - q_1[0]*a[idx    ].y*psi_1[idx    ].y) +
                        sqr(-(psi_1[idx + N].x - psi_1[idx].x)/dx - q_1[0]*a[idx + N].y*psi_1[idx + N].y) +

                        sqr(+(psi_1[idx + N].y - psi_1[idx].y)/dx - q_1[0]*a[idx    ].y*psi_1[idx    ].x) + 
                        sqr(+(psi_1[idx + N].y - psi_1[idx].y)/dx - q_1[0]*a[idx + N].y*psi_1[idx + N].x)
                    );

        #ifdef MULTICOMPONENT
        kinetic_term_y += 0.25/m_yy_2[idx_0]*(
                            sqr(-(psi_2[idx + N].x - psi_2[idx].x)/dx - q_2[0]*a[idx    ].y*psi_2[idx    ].y) +
                            sqr(-(psi_2[idx + N].x - psi_2[idx].x)/dx - q_2[0]*a[idx + N].y*psi_2[idx + N].y) +
    
                            sqr(+(psi_2[idx + N].y - psi_2[idx].y)/dx - q_2[0]*a[idx    ].y*psi_2[idx    ].x) + 
                            sqr(+(psi_2[idx + N].y - psi_2[idx].y)/dx - q_2[0]*a[idx + N].y*psi_2[idx + N].x)
                        );
        #endif
    }
    
    
    kinetic_term_x *= (int2bit(sc_nn_map[idx_0], 1)*int2bit(sc_nn_map[idx_0], 2) + int2bit(sc_nn_map[idx_0], 6)*int2bit(sc_nn_map[idx_0], 7))/2.0f;
    kinetic_term_y *= (int2bit(sc_nn_map[idx_0], 0)*int2bit(sc_nn_map[idx_0], 1) + int2bit(sc_nn_map[idx_0], 3)*int2bit(sc_nn_map[idx_0], 4))/2.0f;

    real em_term = 0;
    
    if(int2bit(comp_nn_map[idx_0], 0) && int2bit(comp_nn_map[idx_0], 1) && int2bit(comp_nn_map[idx_0], 2))
        em_term += 0.25*0.5*sqr((a[idx+1].y - a[idx].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);
    
    if(int2bit(comp_nn_map[idx_0], 2) && int2bit(comp_nn_map[idx_0], 3) && int2bit(comp_nn_map[idx_0], 4))
        em_term += 0.25*0.5*sqr((a[idx].y - a[idx-1].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);

    if(int2bit(comp_nn_map[idx_0], 4) && int2bit(comp_nn_map[idx_0], 5) && int2bit(comp_nn_map[idx_0], 6))
        em_term += 0.25*0.5*sqr((a[idx].y - a[idx-1].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);

    if(int2bit(comp_nn_map[idx_0], 6) && int2bit(comp_nn_map[idx_0], 7) && int2bit(comp_nn_map[idx_0], 0))
        em_term += 0.25*0.5*sqr((a[idx+1].y - a[idx].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);

    #ifndef NO_KINETIC
    fenergy_density[idx] += (kinetic_term_x + kinetic_term_y)*dx*dx;
    #endif
    #ifndef NO_NL
    fenergy_density[idx] += nl_term*dx*dx;
    #endif
    #if !defined(NO_EM) && !defined(NO_SELF_FIELD)
    fenergy_density[idx] += em_term*dx*dx;
    #endif
}


__host__
void computeFreeEnergy( int F, int *dev_comp_nn_map, int *dev_sc_nn_map,
                        real *dev_a_1, real *dev_b_1, real *dev_m_xx_1, real *dev_m_yy_1, 
                        real *dev_a_2, real *dev_b_2, real *dev_m_xx_2, real *dev_m_yy_2, 
                        real *dev_h, real *dev_q_1, real *dev_q_2, real *dev_eta, real *dev_gamma, real *dev_delta,
                        real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
                        real* dev_fenergy){

    real *dev_fenergy_density = nullptr;
    checkCudaErrors(cudaMalloc(&dev_fenergy_density, F*N*N*sizeof(real)));

    freeEnergyDensity<<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>( dev_comp_nn_map, dev_sc_nn_map,
                                                                     dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1,
                                                                     dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2,
                                                                     dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                                                                     dev_a, dev_psi_1, dev_psi_2,
                                                                     dev_fenergy_density);
    checkCudaErrors(cudaDeviceSynchronize());

    parallel_sum::sumArrayF(dev_fenergy_density, dev_fenergy, F, N*N);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(dev_fenergy_density));
}


__global__
void computeJBKernel(int *comp_nn_map, int *sc_nn_map,
    real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, 
    real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, 
    real *h, real *q_1, real *q_2, 
    real2 *a, real2 *psi_1, real2 *psi_2,
    real* b, real2 *j_1, real2 *j_2){
    
    // This kernel compute the gradient at each site of the bulk

    int idx_0 =  blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx_0 >= N*N){
        return;
    }

    int idx = idx_0 + N*N*blockIdx.y;


    b[idx] = 0;
    
    if(int2bit(comp_nn_map[idx_0], 0) && int2bit(comp_nn_map[idx_0], 1) && int2bit(comp_nn_map[idx_0], 2))
        b[idx] += 0.25*((a[idx+1].y - a[idx].y)/dx - (a[idx+N].x - a[idx].x)/dx);
    
    if(int2bit(comp_nn_map[idx_0], 2) && int2bit(comp_nn_map[idx_0], 3) && int2bit(comp_nn_map[idx_0], 4))
        b[idx] += 0.25*((a[idx].y - a[idx-1].y)/dx - (a[idx+N].x - a[idx].x)/dx);

    if(int2bit(comp_nn_map[idx_0], 4) && int2bit(comp_nn_map[idx_0], 5) && int2bit(comp_nn_map[idx_0], 6))
        b[idx] += 0.25*((a[idx].y - a[idx-1].y)/dx - (a[idx].x - a[idx-N].x)/dx);

    if(int2bit(comp_nn_map[idx_0], 6) && int2bit(comp_nn_map[idx_0], 7) && int2bit(comp_nn_map[idx_0], 0))
        b[idx] += 0.25*((a[idx+1].y - a[idx].y)/dx - (a[idx].x - a[idx-N].x)/dx); 
    
    if (b[idx]!=0) 
        b[idx] /= ( int2bit(comp_nn_map[idx_0], 0)*int2bit(comp_nn_map[idx_0], 1)*int2bit(comp_nn_map[idx_0], 2) + 
                    int2bit(comp_nn_map[idx_0], 2)*int2bit(comp_nn_map[idx_0], 3)*int2bit(comp_nn_map[idx_0], 4) + 
                    int2bit(comp_nn_map[idx_0], 4)*int2bit(comp_nn_map[idx_0], 5)*int2bit(comp_nn_map[idx_0], 6) + 
                    int2bit(comp_nn_map[idx_0], 6)*int2bit(comp_nn_map[idx_0], 7)*int2bit(comp_nn_map[idx_0], 0))/4.0f;


    j_1[idx].x = 0;
    j_1[idx].y = 0;
    
    if (int2bit(sc_nn_map[idx_0], 0)){
        j_1[idx].x += +0.5*q_1[0]/m_xx_1[idx_0]*(
            
            + psi_1[idx].y * (-(psi_1[idx+1].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx].y)  
            + psi_1[idx].x * (+(psi_1[idx+1].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx].x)            
            
        );
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        j_1[idx].x += +0.5*q_1[0]/m_xx_1[idx_0]*(

            + psi_1[idx].y * (-(psi_1[idx].x - psi_1[idx-1].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx].y)
            + psi_1[idx].x * (+(psi_1[idx].y - psi_1[idx-1].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx].x)      
            
        );
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        j_1[idx].y += +0.5*q_1[0]/m_yy_1[idx_0]*(
                         
            + psi_1[idx].y * (-(psi_1[idx+N].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx].y)
            + psi_1[idx].x * (+(psi_1[idx+N].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx].x) 
            
        );  
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        j_1[idx].y += +0.5*q_1[0]/m_yy_1[idx_0]*(
          
            + psi_1[idx].y * (-(psi_1[idx].x - psi_1[idx-N].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx].y)
            + psi_1[idx].x * (+(psi_1[idx].y - psi_1[idx-N].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx].x) 
            
        );
    }    
    
    #ifdef MULTICOMPONENT

    j_2[idx].x = 0;
    j_2[idx].y = 0;
    
    if (int2bit(sc_nn_map[idx_0], 0)){
        j_2[idx].x += +0.5*q_2[0]/m_xx_2[idx_0]*(
            
            + psi_2[idx].y * (-(psi_2[idx+1].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx].y)  
            + psi_2[idx].x * (+(psi_2[idx+1].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx].x)            
            
        );
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        j_2[idx].x += +0.5*q_2[0]/m_xx_2[idx_0]*(

            + psi_2[idx].y * (-(psi_2[idx].x - psi_2[idx-1].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx].y)
            + psi_2[idx].x * (+(psi_2[idx].y - psi_2[idx-1].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx].x)      
            
        );
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        j_2[idx].y += +0.5*q_2[0]/m_yy_2[idx_0]*(
                         
            + psi_2[idx].y * (-(psi_2[idx+N].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx].y)
            + psi_2[idx].x * (+(psi_2[idx+N].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx].x) 
            
        );  
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        j_2[idx].y += +0.5*q_2[0]/m_yy_2[idx_0]*(
          
            + psi_2[idx].y * (-(psi_2[idx].x - psi_2[idx-N].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx].y)
            + psi_2[idx].x * (+(psi_2[idx].y - psi_2[idx-N].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx].x) 
            
        );
    }

    #endif
}

__host__
void computeJB(int F, int *dev_comp_nn_map, int *dev_sc_nn_map,
               real *dev_a_1, real *dev_b_1, real *dev_m_xx_1, real *dev_m_yy_1, 
               real *dev_a_2, real *dev_b_2, real *dev_m_xx_2, real *dev_m_yy_2, 
               real *dev_h, real *dev_q_1, real *dev_q_2,
               real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
               real* dev_b, real2 *dev_j_1, real2 *dev_j_2){
                   
    computeJBKernel<<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_comp_nn_map, dev_sc_nn_map,
                                                             dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                                                             dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                                                             dev_h, dev_q_1, dev_q_2,
                                                             dev_a, dev_psi_1, dev_psi_2,
                                                             dev_b, dev_j_1, dev_j_2);
}


__global__
void computeAbsGradKernel(int *sc_nn_map,
    real *q_1, real *q_2, 
    real2 *a, real2 *psi_1, real2 *psi_2,
    real* psi_1_abs, real2 *cd_1,
    real* psi_2_abs, real2 *cd_2){
    
    // This kernel compute the gradient at each site of the bulk

    int idx_0 =  blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx_0 >= N*N){
        return;
    }

    int idx = idx_0 + N*N*blockIdx.y;


    psi_1_abs[idx] = std::sqrt(sqr(psi_1[idx].x) + sqr(psi_1[idx].y));

    cd_1[idx].x = 0;
    cd_1[idx].y = 0;
    
    if (int2bit(sc_nn_map[idx_0], 0)){
        cd_1[idx].x += (
                (+ psi_1[idx].x * (psi_1[idx+1].y - psi_1[idx].y)/dx 
                 - psi_1[idx].y * (psi_1[idx+1].x - psi_1[idx].x)/dx)
                 / (sqr(psi_1[idx].x) + sqr(psi_1[idx].y))
                 - q_1[0]*(a[idx+1].x + a[idx].x)/2         
        );
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        cd_1[idx].x += (
            (+ psi_1[idx].x * (psi_1[idx].y - psi_1[idx-1].y)/dx 
             - psi_1[idx].y * (psi_1[idx].x - psi_1[idx-1].x)/dx)
             / (sqr(psi_1[idx].x) + sqr(psi_1[idx].y))
             - q_1[0]*(a[idx].x + a[idx-1].x)/2         
        );
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        cd_1[idx].y += (
            (+ psi_1[idx].x * (psi_1[idx+N].y - psi_1[idx].y)/dx 
             - psi_1[idx].y * (psi_1[idx+N].x - psi_1[idx].x)/dx)
             / (sqr(psi_1[idx].x) + sqr(psi_1[idx].y))
             - q_1[0]*(a[idx+N].x - a[idx].y)/2         
        );
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        cd_1[idx].y += (
            (+ psi_1[idx].x * (psi_1[idx].y - psi_1[idx-N].y)/dx 
             - psi_1[idx].y * (psi_1[idx].x - psi_1[idx-N].x)/dx)
             / (sqr(psi_1[idx].x) + sqr(psi_1[idx].y))
             - q_1[0]*(a[idx].x - a[idx-N].y)/2        
        );
    }    
    
    #ifdef MULTICOMPONENT

    psi_2_abs[idx] = std::sqrt(sqr(psi_2[idx].x) + sqr(psi_2[idx].y));

    cd_2[idx].x = 0;
    cd_2[idx].y = 0;
    
    if (int2bit(sc_nn_map[idx_0], 0)){
        cd_2[idx].x += (
                (+ psi_2[idx].x * (psi_2[idx+1].y - psi_2[idx].y)/dx 
                 - psi_2[idx].y * (psi_2[idx+1].x - psi_2[idx].x)/dx)
                 / (sqr(psi_2[idx].x) + sqr(psi_2[idx].y))
                 - q_2[0]*(a[idx+1].x + a[idx].x)/2        
        );
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        cd_2[idx].x += (
            (+ psi_2[idx].x * (psi_2[idx].y - psi_2[idx-1].y)/dx 
             - psi_2[idx].y * (psi_2[idx].x - psi_2[idx-1].x)/dx)
             / (sqr(psi_2[idx].x) + sqr(psi_2[idx].y))
             - q_2[0]*(a[idx].x + a[idx-1].x)/2       
        );
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        cd_2[idx].y += (
            (+ psi_2[idx].x * (psi_2[idx+N].y - psi_2[idx].y)/dx 
             - psi_2[idx].y * (psi_2[idx+N].x - psi_2[idx].x)/dx)
             / (sqr(psi_2[idx].x) + sqr(psi_2[idx].y))
             - q_2[0]*(a[idx+N].x - a[idx].y)/2        
        );
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        cd_2[idx].y += (
            (+ psi_2[idx].x * (psi_2[idx].y - psi_2[idx-N].y)/dx 
             - psi_2[idx].y * (psi_2[idx].x - psi_2[idx-N].x)/dx)
             / (sqr(psi_2[idx].x) + sqr(psi_2[idx].y))
             - q_2[0]*(a[idx].x - a[idx-N].y)/2       
        );
    }    

    #endif
}

__host__
void computeAbsGrad(int F, int *dev_sc_nn_map,
               real *dev_q_1, real *dev_q_2,
               real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
               real* dev_psi_1_abs, real2 *dev_cd_1,
               real* dev_psi_2_abs, real2 *dev_cd_2){
                   
    computeAbsGradKernel<<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_sc_nn_map,
                                                             dev_q_1, dev_q_2, 
                                                             dev_a, dev_psi_1, dev_psi_2,
                                                             dev_psi_1_abs, dev_cd_1,
                                                             dev_psi_2_abs, dev_cd_2);
}

/******************************************* Functions needed to compute gradient ************************************************************/

__global__
void gradientPsi( int* comp_nn_map, int* sc_nn_map,
                  real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, 
                  real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, 
                  real *h, real *q_1, real *q_2, real *eta, real *gamma, real *delta,
                  real2 *a, real2 *psi_1, real2 *psi_2,
                  real2 *a_g, real2 *psi_1_g, real2 *psi_2_g){
    
    // This kernel compute the gradient at each site of the bulk

    int idx_0 =  blockDim.x*blockIdx.x + threadIdx.x;

    if(idx_0 >= N*N){
        return;
    }

    int idx = idx_0 + N*N*blockIdx.y;

    real area_factor = (   int2bit(sc_nn_map[idx_0], 0)*int2bit(sc_nn_map[idx_0], 1)*int2bit(sc_nn_map[idx_0], 2) + 
                           int2bit(sc_nn_map[idx_0], 2)*int2bit(sc_nn_map[idx_0], 3)*int2bit(sc_nn_map[idx_0], 4) + 
                           int2bit(sc_nn_map[idx_0], 4)*int2bit(sc_nn_map[idx_0], 5)*int2bit(sc_nn_map[idx_0], 6) + 
                           int2bit(sc_nn_map[idx_0], 6)*int2bit(sc_nn_map[idx_0], 7)*int2bit(sc_nn_map[idx_0], 0))/4.0f;
    
    psi_1_g[idx].x = (2*a_1[idx_0]*psi_1[idx].x + b_1[idx_0]*2*psi_1[idx].x*(sqr(psi_1[idx].x) + sqr(psi_1[idx].y)))*area_factor;
    psi_1_g[idx].y = (2*a_1[idx_0]*psi_1[idx].y + b_1[idx_0]*2*psi_1[idx].y*(sqr(psi_1[idx].x) + sqr(psi_1[idx].y)))*area_factor;

    if (int2bit(sc_nn_map[idx_0], 0)){
        psi_1_g[idx].x += 0.5/m_xx_1[idx_0]*(
            
            + 1.0/dx * (- (psi_1[idx+1].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].y) 
            + 1.0/dx * (- (psi_1[idx+1].x - psi_1[idx].x)/dx - q_1[0]*a[idx+1].x*psi_1[idx+1].y)
                              
            - q_1[0]*a[idx  ].x * (+ (psi_1[idx+1].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 1) * int2bit(sc_nn_map[idx_0], 2) +
             int2bit(sc_nn_map[idx_0], 6) * int2bit(sc_nn_map[idx_0], 7))/2.0f;

        psi_1_g[idx].y += 0.5/m_xx_1[idx_0]*(
            
            - 1.0/dx * (+ (psi_1[idx+1].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].x) 
            - 1.0/dx * (+ (psi_1[idx+1].y - psi_1[idx].y)/dx - q_1[0]*a[idx+1].x*psi_1[idx+1].x)
                            
            - q_1[0]*a[idx  ].x * (- (psi_1[idx+1].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 1) * int2bit(sc_nn_map[idx_0], 2) + 
             int2bit(sc_nn_map[idx_0], 6) * int2bit(sc_nn_map[idx_0], 7))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        psi_1_g[idx].x += 0.5/m_xx_1[idx_0]*(
            
            - 1.0/dx * (- (psi_1[idx].x - psi_1[idx-1].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].y) 
            - 1.0/dx * (- (psi_1[idx].x - psi_1[idx-1].x)/dx - q_1[0]*a[idx-1].x*psi_1[idx-1].y)
                            
            - q_1[0]*a[idx  ].x * (+ (psi_1[idx].y - psi_1[idx-1].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 2) * int2bit(sc_nn_map[idx_0], 3) + 
             int2bit(sc_nn_map[idx_0], 5) * int2bit(sc_nn_map[idx_0], 6))/2.0f;

        psi_1_g[idx].y += 0.5/m_xx_1[idx_0]*(
            
            + 1.0/dx * (+ (psi_1[idx].y - psi_1[idx-1].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].x) 
            + 1.0/dx * (+ (psi_1[idx].y - psi_1[idx-1].y)/dx - q_1[0]*a[idx-1].x*psi_1[idx-1].x)
                            
            - q_1[0]*a[idx  ].x * (- (psi_1[idx].x - psi_1[idx-1].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 2) * int2bit(sc_nn_map[idx_0], 3) +
             int2bit(sc_nn_map[idx_0], 5) * int2bit(sc_nn_map[idx_0], 6))/2.0f;
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        psi_1_g[idx].x += 0.5/m_yy_1[idx_0]*( 
            
            + 1.0/dx * (- (psi_1[idx+N].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].y) 
            + 1.0/dx * (- (psi_1[idx+N].x - psi_1[idx].x)/dx - q_1[0]*a[idx+N].y*psi_1[idx+N].y)
                            
            - q_1[0]*a[idx  ].y * (+ (psi_1[idx+N].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 0) * int2bit(sc_nn_map[idx_0], 1) + 
             int2bit(sc_nn_map[idx_0], 3) * int2bit(sc_nn_map[idx_0], 4))/2.0f;

        psi_1_g[idx].y += 0.5/m_yy_1[idx_0]*(
            
            - 1.0/dx * (+ (psi_1[idx+N].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].x) 
            - 1.0/dx * (+ (psi_1[idx+N].y - psi_1[idx].y)/dx - q_1[0]*a[idx+N].y*psi_1[idx+N].x)
                            
            - q_1[0]*a[idx  ].y * (- (psi_1[idx+N].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 0) * int2bit(sc_nn_map[idx_0], 1) + 
             int2bit(sc_nn_map[idx_0], 3) * int2bit(sc_nn_map[idx_0], 4))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        psi_1_g[idx].x += 0.5/m_yy_1[idx_0]*(
            
            - 1.0/dx * (- (psi_1[idx].x - psi_1[idx-N].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].y) 
            - 1.0/dx * (- (psi_1[idx].x - psi_1[idx-N].x)/dx - q_1[0]*a[idx-N].y*psi_1[idx-N].y)
                            
            - q_1[0]*a[idx  ].y * (+(psi_1[idx].y - psi_1[idx-N].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 4) * int2bit(sc_nn_map[idx_0], 5) +
             int2bit(sc_nn_map[idx_0], 7) * int2bit(sc_nn_map[idx_0], 0))/2.0f;

        psi_1_g[idx].y += 0.5/m_yy_1[idx_0]*(
            
            + 1.0/dx * (+ (psi_1[idx].y - psi_1[idx-N].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].x) 
            + 1.0/dx * (+ (psi_1[idx].y - psi_1[idx-N].y)/dx - q_1[0]*a[idx-N].y*psi_1[idx-N].x)
                            
            - q_1[0]*a[idx  ].y * (-(psi_1[idx].x - psi_1[idx-N].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 4) * int2bit(sc_nn_map[idx_0], 5) +
             int2bit(sc_nn_map[idx_0], 7) * int2bit(sc_nn_map[idx_0], 0))/2.0f;   
    }

    #ifdef MULTICOMPONENT
    psi_2_g[idx].x = (2*a_2[idx_0]*psi_2[idx].x + b_2[idx_0]*2*psi_2[idx].x*(sqr(psi_2[idx].x) + sqr(psi_2[idx].y)))*area_factor;
    psi_2_g[idx].y = (2*a_2[idx_0]*psi_2[idx].y + b_2[idx_0]*2*psi_2[idx].y*(sqr(psi_2[idx].x) + sqr(psi_2[idx].y)))*area_factor;

    #ifdef BILINEAR_JOSEPHSON_COUPLING
    psi_1_g[idx].x += eta[0]*psi_2[idx].x*area_factor;
    psi_1_g[idx].y += eta[0]*psi_2[idx].y*area_factor;

    psi_2_g[idx].x += eta[0]*psi_1[idx].x*area_factor;    
    psi_2_g[idx].y += eta[0]*psi_1[idx].y*area_factor;
    #endif 

    #ifdef DENSITY_COUPLING
    psi_1_g[idx].x += gamma[0]*psi_1[idx].x*(sqr(psi_2[idx].x) + sqr(psi_2[idx].y))*area_factor;
    psi_1_g[idx].y += gamma[0]*psi_1[idx].y*(sqr(psi_2[idx].x) + sqr(psi_2[idx].y))*area_factor;

    psi_2_g[idx].x += gamma[0]*psi_2[idx].x*(sqr(psi_1[idx].x) + sqr(psi_1[idx].y))*area_factor;    
    psi_2_g[idx].y += gamma[0]*psi_2[idx].y*(sqr(psi_1[idx].x) + sqr(psi_1[idx].y))*area_factor;
    #endif

    #ifdef BIQUADRATIC_JOSEPHSON_COUPLING
    psi_1_g[idx].x += delta[0] * ( + psi_2[idx].x * (psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y) 
                                   - psi_2[idx].y * (psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x) )*area_factor;
    psi_1_g[idx].y += delta[0] * ( + psi_2[idx].y * (psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y) 
                                   + psi_2[idx].x * (psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x) )*area_factor;    
    
    psi_2_g[idx].x += delta[0] * ( + psi_1[idx].x * (psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y) 
                                   + psi_1[idx].y * (psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x) )*area_factor;
    psi_2_g[idx].y += delta[0] * ( + psi_1[idx].y * (psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y) 
                                   - psi_1[idx].x * (psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x) )*area_factor;
    #endif

    if (int2bit(sc_nn_map[idx_0], 0)){
        psi_2_g[idx].x += 0.5/m_xx_2[idx_0]*(
            
            + 1.0/dx * (- (psi_2[idx+1].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].y) 
            + 1.0/dx * (- (psi_2[idx+1].x - psi_2[idx].x)/dx - q_2[0]*a[idx+1].x*psi_2[idx+1].y)
                              
            - q_2[0]*a[idx  ].x * (+(psi_2[idx+1].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 1) * int2bit(sc_nn_map[idx_0], 2) +
             int2bit(sc_nn_map[idx_0], 6) * int2bit(sc_nn_map[idx_0], 7))/2.0f;

        psi_2_g[idx].y += 0.5/m_xx_2[idx_0]*(
            
            - 1.0/dx * (+ (psi_2[idx+1].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].x) 
            - 1.0/dx * (+ (psi_2[idx+1].y - psi_2[idx].y)/dx - q_2[0]*a[idx+1].x*psi_2[idx+1].x)
                            
            - q_2[0]*a[idx  ].x   * (- (psi_2[idx+1].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 1) * int2bit(sc_nn_map[idx_0], 2) + 
             int2bit(sc_nn_map[idx_0], 6) * int2bit(sc_nn_map[idx_0], 7))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        psi_2_g[idx].x += 0.5/m_xx_2[idx_0]*(
            
            - 1.0/dx * (- (psi_2[idx].x - psi_2[idx-1].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].y) 
            - 1.0/dx * (- (psi_2[idx].x - psi_2[idx-1].x)/dx - q_2[0]*a[idx-1].x*psi_2[idx-1].y)
                            
            - q_2[0]*a[idx  ].x * (+ (psi_2[idx].y - psi_2[idx-1].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 2) * int2bit(sc_nn_map[idx_0], 3) + 
             int2bit(sc_nn_map[idx_0], 5) * int2bit(sc_nn_map[idx_0], 6))/2.0f;

        psi_2_g[idx].y += 0.5/m_xx_2[idx_0]*(
            
            + 1.0/dx * (+ (psi_2[idx].y - psi_2[idx-1].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].x) 
            + 1.0/dx * (+ (psi_2[idx].y - psi_2[idx-1].y)/dx - q_2[0]*a[idx-1].x*psi_2[idx-1].x)
                            
            - q_2[0]*a[idx  ].x * (- (psi_2[idx].x - psi_2[idx-1].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 2) * int2bit(sc_nn_map[idx_0], 3) +
             int2bit(sc_nn_map[idx_0], 5) * int2bit(sc_nn_map[idx_0], 6))/2.0f;
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        psi_2_g[idx].x += 0.5/m_yy_2[idx_0]*( 
            
            + 1.0/dx * (- (psi_2[idx+N].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].y) 
            + 1.0/dx * (- (psi_2[idx+N].x - psi_2[idx].x)/dx - q_2[0]*a[idx+N].y*psi_2[idx+N].y)
                            
            - q_2[0]*a[idx  ].y * (+ (psi_2[idx+N].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 0) * int2bit(sc_nn_map[idx_0], 1) + 
             int2bit(sc_nn_map[idx_0], 3) * int2bit(sc_nn_map[idx_0], 4))/2.0f;

        psi_2_g[idx].y += 0.5/m_yy_2[idx_0]*(
            
            - 1.0/dx * (+ (psi_2[idx+N].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].x) 
            - 1.0/dx * (+ (psi_2[idx+N].y - psi_2[idx].y)/dx - q_2[0]*a[idx+N].y*psi_2[idx+N].x)
                            
            - q_2[0]*a[idx  ].y * (- (psi_2[idx+N].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 0) * int2bit(sc_nn_map[idx_0], 1) + 
             int2bit(sc_nn_map[idx_0], 3) * int2bit(sc_nn_map[idx_0], 4))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        psi_2_g[idx].x += 0.5/m_yy_2[idx_0]*(
            
            - 1.0/dx * (- (psi_2[idx].x - psi_2[idx-N].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].y) 
            - 1.0/dx * (- (psi_2[idx].x - psi_2[idx-N].x)/dx - q_2[0]*a[idx-N].y*psi_2[idx-N].y)
                            
            - q_2[0]*a[idx  ].y * (+(psi_2[idx].y - psi_2[idx-N].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 4) * int2bit(sc_nn_map[idx_0], 5) +
             int2bit(sc_nn_map[idx_0], 7) * int2bit(sc_nn_map[idx_0], 0))/2.0f;

        psi_2_g[idx].y += 0.5/m_yy_2[idx_0]*(
            
            + 1.0/dx * (+ (psi_2[idx].y - psi_2[idx-N].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].x) 
            + 1.0/dx * (+ (psi_2[idx].y - psi_2[idx-N].y)/dx - q_2[0]*a[idx-N].y*psi_2[idx-N].x)
                            
            - q_2[0]*a[idx  ].y * (-(psi_2[idx].x - psi_2[idx-N].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx  ].y) 
            
        ) * (int2bit(sc_nn_map[idx_0], 4) * int2bit(sc_nn_map[idx_0], 5) +
             int2bit(sc_nn_map[idx_0], 7) * int2bit(sc_nn_map[idx_0], 0))/2.0f;   
    }
    #endif

    psi_1_g[idx].x *= dx*dx;
    psi_1_g[idx].y *= dx*dx;   

    psi_2_g[idx].x *= dx*dx;
    psi_2_g[idx].y *= dx*dx; 
}

__global__
void gradientA( int* comp_nn_map, int* sc_nn_map,
                real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, 
                real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, 
                real *h, real *q_1, real *q_2, real *eta, real *gamma, real *delta,
                real2 *a, real2 *psi_1, real2 *psi_2,
                real2 *a_g, real2 *psi_1_g, real2 *psi_2_g
                ){
    
    // This kernel compute the gradient at each site of the bulk

    int idx_0 =  blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx_0 >= N*N){
        return;
    }

    int idx = idx_0 + N*N*blockIdx.y;

    a_g[idx].x = 0;
    a_g[idx].y = 0;

    // Kinetic term
    if (int2bit(sc_nn_map[idx_0], 0)){
        a_g[idx].x += -0.5*q_1[0]/m_xx_1[idx_0]*(
            
            + psi_1[idx].y * (-(psi_1[idx+1].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx].y)  
            + psi_1[idx].x * (+(psi_1[idx+1].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx].x)            
            
        ) * ( int2bit(sc_nn_map[idx_0], 1) * int2bit(sc_nn_map[idx_0], 2) +
              int2bit(sc_nn_map[idx_0], 6) * int2bit(sc_nn_map[idx_0], 7))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        a_g[idx].x += -0.5*q_1[0]/m_xx_1[idx_0]*(

            + psi_1[idx].y * (-(psi_1[idx].x - psi_1[idx-1].x)/dx - q_1[0]*a[idx  ].x*psi_1[idx].y)
            + psi_1[idx].x * (+(psi_1[idx].y - psi_1[idx-1].y)/dx - q_1[0]*a[idx  ].x*psi_1[idx].x)      
            
        )  * (int2bit(sc_nn_map[idx_0], 2) * int2bit(sc_nn_map[idx_0], 3) + 
              int2bit(sc_nn_map[idx_0], 5) * int2bit(sc_nn_map[idx_0], 6))/2.0f;
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        a_g[idx].y += -0.5*q_1[0]/m_yy_1[idx_0]*(
                         
            + psi_1[idx].y * (-(psi_1[idx+N].x - psi_1[idx].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx].y)
            + psi_1[idx].x * (+(psi_1[idx+N].y - psi_1[idx].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx].x) 
            
        )  * (int2bit(sc_nn_map[idx_0], 0) * int2bit(sc_nn_map[idx_0], 1) + 
              int2bit(sc_nn_map[idx_0], 3) * int2bit(sc_nn_map[idx_0], 4))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        a_g[idx].y += -0.5*q_1[0]/m_yy_1[idx_0]*(
          
            + psi_1[idx].y * (-(psi_1[idx].x - psi_1[idx-N].x)/dx - q_1[0]*a[idx  ].y*psi_1[idx].y)
            + psi_1[idx].x * (+(psi_1[idx].y - psi_1[idx-N].y)/dx - q_1[0]*a[idx  ].y*psi_1[idx].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 4) * int2bit(sc_nn_map[idx_0], 5) +
             int2bit(sc_nn_map[idx_0], 7) * int2bit(sc_nn_map[idx_0], 0))/2.0f;
    }        
    
    #ifdef MULTICOMPONENT
    if (int2bit(sc_nn_map[idx_0], 0)){
        a_g[idx].x += -0.5*q_2[0]/m_xx_2[idx_0]*(
            
            + psi_2[idx].y * (-(psi_2[idx+1].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx].y)  
            + psi_2[idx].x * (+(psi_2[idx+1].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx].x)            
            
        ) * ( int2bit(sc_nn_map[idx_0], 1) * int2bit(sc_nn_map[idx_0], 2) +
              int2bit(sc_nn_map[idx_0], 6) * int2bit(sc_nn_map[idx_0], 7))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 4)){
        a_g[idx].x += -0.5*q_2[0]/m_xx_2[idx_0]*(

            + psi_2[idx].y * (-(psi_2[idx].x - psi_2[idx-1].x)/dx - q_2[0]*a[idx  ].x*psi_2[idx].y)
            + psi_2[idx].x * (+(psi_2[idx].y - psi_2[idx-1].y)/dx - q_2[0]*a[idx  ].x*psi_2[idx].x)      
            
        )  * (int2bit(sc_nn_map[idx_0], 2) * int2bit(sc_nn_map[idx_0], 3) + 
              int2bit(sc_nn_map[idx_0], 5) * int2bit(sc_nn_map[idx_0], 6))/2.0f;
    }   

    if (int2bit(sc_nn_map[idx_0], 2)){
        a_g[idx].y += -0.5*q_2[0]/m_yy_2[idx_0]*(
                         
            + psi_2[idx].y * (-(psi_2[idx+N].x - psi_2[idx].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx].y)
            + psi_2[idx].x * (+(psi_2[idx+N].y - psi_2[idx].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx].x) 
            
        )  * (int2bit(sc_nn_map[idx_0], 0) * int2bit(sc_nn_map[idx_0], 1) + 
              int2bit(sc_nn_map[idx_0], 3) * int2bit(sc_nn_map[idx_0], 4))/2.0f;
    }

    if (int2bit(sc_nn_map[idx_0], 6)){
        a_g[idx].y += -0.5*q_2[0]/m_yy_2[idx_0]*(
          
            + psi_2[idx].y * (-(psi_2[idx].x - psi_2[idx-N].x)/dx - q_2[0]*a[idx  ].y*psi_2[idx].y)
            + psi_2[idx].x * (+(psi_2[idx].y - psi_2[idx-N].y)/dx - q_2[0]*a[idx  ].y*psi_2[idx].x) 
            
        ) * (int2bit(sc_nn_map[idx_0], 4) * int2bit(sc_nn_map[idx_0], 5) +
             int2bit(sc_nn_map[idx_0], 7) * int2bit(sc_nn_map[idx_0], 0))/2.0f;
    }
    #endif


    // Curl term 
    if(int2bit(comp_nn_map[idx_0], 0)  && int2bit(comp_nn_map[idx_0], 1) && int2bit(comp_nn_map[idx_0], 2)){
        a_g[idx].x += +0.25/dx*( (a[idx+1].y - a[idx].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);
        a_g[idx].y += -0.25/dx*( (a[idx+1].y - a[idx].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);
        
        a_g[idx].y += -0.25/dx*((a[idx+1].y - a[idx].y    )/dx - (a[idx+N+1].x - a[idx+1].x)/dx - h[idx_0+1]);
        a_g[idx].x += +0.25/dx*((a[idx+1+N].y - a[idx+N].y)/dx - (a[idx+N].x - a[idx].x    )/dx - h[idx_0+N]);
    }

    if(int2bit(comp_nn_map[idx_0], 2) && int2bit(comp_nn_map[idx_0], 3) && int2bit(comp_nn_map[idx_0], 4)){
        a_g[idx].x += +0.25/dx*((a[idx].y - a[idx-1].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);
        a_g[idx].y += +0.25/dx*((a[idx].y - a[idx-1].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);

        a_g[idx].x += +0.25/dx*((a[idx+N].y - a[idx+N-1].y)/dx - (a[idx+N].x - a[idx].x    )/dx - h[idx_0+N]);
        a_g[idx].y += +0.25/dx*((a[idx].y - a[idx-1].y    )/dx - (a[idx+N-1].x - a[idx-1].x)/dx - h[idx_0-1]);
    }
        
    if(int2bit(comp_nn_map[idx_0], 4)  && int2bit(comp_nn_map[idx_0], 5) && int2bit(comp_nn_map[idx_0], 6)){
        a_g[idx].x += -0.25/dx*((a[idx].y - a[idx-1].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);
        a_g[idx].y += +0.25/dx*((a[idx].y - a[idx-1].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);
        
        a_g[idx].y += +0.25/dx*((a[idx].y - a[idx-1].y    )/dx - (a[idx-1].x - a[idx-1-N].x)/dx - h[idx_0-1]);
        a_g[idx].x += -0.25/dx*((a[idx-N].y - a[idx-N-1].y)/dx - (a[idx].x - a[idx-N].x    )/dx - h[idx_0-N]);
    }

    if(int2bit(comp_nn_map[idx_0], 6) && int2bit(comp_nn_map[idx_0], 7) && int2bit(comp_nn_map[idx_0], 0)){
        a_g[idx].x += -0.25/dx*((a[idx+1].y - a[idx].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);
        a_g[idx].y += -0.25/dx*((a[idx+1].y - a[idx].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);

        a_g[idx].x += -0.25/dx*((a[idx-N+1].y - a[idx-N].y)/dx - (a[idx].x - a[idx-N].x    )/dx - h[idx_0-N]);
        a_g[idx].y += -0.25/dx*((a[idx+1].y - a[idx].y    )/dx - (a[idx+1].x - a[idx+1-N].x)/dx - h[idx_0+1]);    
        }

    a_g[idx].x *= dx*dx;
    a_g[idx].y *= dx*dx;

}


// __global__
// void sumGradientAtomic(real2 *a_g, real2 *psi_g, real2 *a_g_old, real2 *psi_g_old, real *g_norm2, real *g_delta_g){
   
//     // This kernel computes the norm of the gradient and g\cdot\Delta g


//     int idx =  blockDim.x*blockIdx.x + threadIdx.x;
    

//     if(idx >= N*N){
//         return;
//     }

//     idx += N*N*blockIdx.y;

//     // TODO: Bottleneck
//     atomicAdd(&g_norm2[blockIdx.y],   sqr(a_g[idx].x) + 
//                                       sqr(a_g[idx].y) + 
//                                       sqr(psi_g[idx].x) + 
//                                       sqr(psi_g[idx].y));

//     atomicAdd(&g_delta_g[blockIdx.y], a_g[idx].x*(a_g[idx].x - a_g_old[idx].x) +
//                                       a_g[idx].y*(a_g[idx].y - a_g_old[idx].y) +  
//                                       psi_g[idx].x*(psi_g[idx].x - psi_g_old[idx].x) +
//                                       psi_g[idx].y*(psi_g[idx].y - psi_g_old[idx].y) );
// }


__global__
void gradientDensity(real2 *a_g, real2 *psi_1_g, real2 *psi_2_g, 
                     real2 *a_g_old, real2 *psi_1_g_old, real2 *psi_2_g_old, 
                     real *g_norm2_density, real *g_delta_g_density){
   
    // This kernel computes the norm of the gradient and g\cdot\Delta g


    int idx =  blockDim.x*blockIdx.x + threadIdx.x;
    

    if(idx >= N*N){
        return;
    }

    idx += N*N*blockIdx.y;

    g_norm2_density[idx] =  sqr(a_g[idx].x) + 
                            sqr(a_g[idx].y) + 
                            sqr(psi_1_g[idx].x) + 
                            sqr(psi_1_g[idx].y) +
                            sqr(psi_2_g[idx].x) + 
                            sqr(psi_2_g[idx].y);

    g_delta_g_density[idx] = a_g[idx].x*(a_g[idx].x - a_g_old[idx].x) +
                             a_g[idx].y*(a_g[idx].y - a_g_old[idx].y) +  
                             psi_1_g[idx].x*(psi_1_g[idx].x - psi_1_g_old[idx].x) +
                             psi_1_g[idx].y*(psi_1_g[idx].y - psi_1_g_old[idx].y) +
                             psi_2_g[idx].x*(psi_2_g[idx].x - psi_2_g_old[idx].x) +
                             psi_2_g[idx].y*(psi_2_g[idx].y - psi_2_g_old[idx].y);
}



/********************************************** Step *************************************************/

__global__
void computeStepDirection( real2 *a, real2 *psi_1, real2 *psi_2, 
                           real2 *a_g, real2 *psi_1_g, real2 *psi_2_g, 
                           real2 *a_d, real2 *psi_1_d, real2 *psi_2_d,  
                           real *g_norm2, real *g_norm2_old, real* g_delta_g, bool nlcg){

    // This routinte compute the optimal descend direction
    
    int idx =  blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx >= N*N){
        return;
    }

    idx += N*N*blockIdx.y;

    real beta = 0;
    
    if(nlcg){
        #if BETA == 1
        
        beta = g_delta_g[blockIdx.y] / g_norm2_old[blockIdx.y];
        beta = max(beta, 0.0f);
    
        #elif BETA == 2
        
        beta = g_norm2[blockIdx.y] / g_norm2_old[blockIdx.y];
        
        #endif
    }

    // if(idx == 0) printf("beta = %f\n", beta);

    a_d[idx].x = - a_g[idx].x + beta*a_d[idx].x;
    a_d[idx].y = - a_g[idx].y + beta*a_d[idx].y;
    psi_1_d[idx].x = - psi_1_g[idx].x + beta*psi_1_d[idx].x;
    psi_1_d[idx].y = - psi_1_g[idx].y + beta*psi_1_d[idx].y;
    psi_2_d[idx].x = - psi_2_g[idx].x + beta*psi_2_d[idx].x;
    psi_2_d[idx].y = - psi_2_g[idx].y + beta*psi_2_d[idx].y;
}


__global__
void polynomialExpansion(int* comp_nn_map, int* sc_nn_map, 
                         real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, 
                         real *a_2, real *b_2, real *m_xx_2, real *m_yy_2,
                         real* h, real *q_1, real *q_2, real *eta, real *gamma, real *delta,
                         real2 *a, real2 *psi_1, real2 *psi_2, 
                         real2 *a_d, real2 *psi_1_d, real2 *psi_2_d, 
                         real* fenergy_density, real4* poly_coeff_density){
        
    // Given a state and a direction compute coefficients of polynomial expansion at each site

    int idx_0 =  blockDim.x*blockIdx.x + threadIdx.x;

    if(idx_0 >= N*N){
        return;
    }

    int idx = idx_0 + N*N*blockIdx.y;


     fenergy_density[idx]      = 0;
     poly_coeff_density[idx].x = 0;
     poly_coeff_density[idx].y = 0;
     poly_coeff_density[idx].z = 0;
     poly_coeff_density[idx].w = 0;
        
    // Nonlinear term
    #ifndef NO_NL
    real area_factor =  (   int2bit(sc_nn_map[idx_0], 0)*int2bit(sc_nn_map[idx_0], 1)*int2bit(sc_nn_map[idx_0], 2) + 
                            int2bit(sc_nn_map[idx_0], 2)*int2bit(sc_nn_map[idx_0], 3)*int2bit(sc_nn_map[idx_0], 4) + 
                            int2bit(sc_nn_map[idx_0], 4)*int2bit(sc_nn_map[idx_0], 5)*int2bit(sc_nn_map[idx_0], 6) + 
                            int2bit(sc_nn_map[idx_0], 6)*int2bit(sc_nn_map[idx_0], 7)*int2bit(sc_nn_map[idx_0], 0))/4.0f;

    real nl_term[3];

    nl_term[0] =  (a_1[idx_0]/b_1[idx_0] + sqr(psi_1[idx].x) + sqr(psi_1[idx].y));
    nl_term[1] = +2*(psi_1[idx].x*psi_1_d[idx].x + psi_1[idx].y*psi_1_d[idx].y);
    nl_term[2] = +(psi_1_d[idx].x*psi_1_d[idx].x + psi_1_d[idx].y*psi_1_d[idx].y);
 
    fenergy_density[idx]      += b_1[idx_0]*( 0.5*sqr(nl_term[0])                         ) * area_factor*dx*dx;
    poly_coeff_density[idx].x += b_1[idx_0]*(                       nl_term[0]*nl_term[1] ) * area_factor*dx*dx;
    poly_coeff_density[idx].y += b_1[idx_0]*( 0.5*sqr(nl_term[1]) + nl_term[0]*nl_term[2] ) * area_factor*dx*dx;
    poly_coeff_density[idx].z += b_1[idx_0]*(                       nl_term[1]*nl_term[2] ) * area_factor*dx*dx;
    poly_coeff_density[idx].w += b_1[idx_0]*( 0.5*sqr(nl_term[2])                         ) * area_factor*dx*dx;

    #ifdef MULTICOMPONENT

    nl_term[0] =  (a_2[idx_0]/b_2[idx_0] + sqr(psi_2[idx].x) + sqr(psi_2[idx].y));
    nl_term[1] = +2*(psi_2[idx].x*psi_2_d[idx].x + psi_2[idx].y*psi_2_d[idx].y);
    nl_term[2] = +(psi_2_d[idx].x*psi_2_d[idx].x + psi_2_d[idx].y*psi_2_d[idx].y);
 
    fenergy_density[idx]      += b_2[idx_0]*( 0.5*sqr(nl_term[0])                         ) * area_factor*dx*dx;
    poly_coeff_density[idx].x += b_2[idx_0]*(                       nl_term[0]*nl_term[1] ) * area_factor*dx*dx;
    poly_coeff_density[idx].y += b_2[idx_0]*( 0.5*sqr(nl_term[1]) + nl_term[0]*nl_term[2] ) * area_factor*dx*dx;
    poly_coeff_density[idx].z += b_2[idx_0]*(                       nl_term[1]*nl_term[2] ) * area_factor*dx*dx;
    poly_coeff_density[idx].w += b_2[idx_0]*( 0.5*sqr(nl_term[2])                         ) * area_factor*dx*dx;
    
    #ifdef BILINEAR_JOSEPHSON_COUPLING
    fenergy_density[idx]      += eta[0] * (psi_1[idx].x*psi_2[idx].x + psi_1[idx].y*psi_2[idx].y) * area_factor*dx*dx;
    poly_coeff_density[idx].x += eta[0] * (psi_1_d[idx].x*psi_2[idx].x + psi_1_d[idx].y*psi_2[idx].y + 
                                           psi_1[idx].x*psi_2_d[idx].x + psi_1[idx].y*psi_2_d[idx].y ) * area_factor*dx*dx;
    poly_coeff_density[idx].y += eta[0] * (psi_1_d[idx].x*psi_2_d[idx].x + psi_1_d[idx].y*psi_2_d[idx].y) * area_factor*dx*dx;
    #endif

    #ifdef DENSITY_COUPLING
    fenergy_density[idx]      += gamma[0]/2.0 * (sqr(psi_1[idx].x) + sqr(psi_1[idx].y)) * 
                                                (sqr(psi_2[idx].x) + sqr(psi_2[idx].y)) * area_factor*dx*dx;

    poly_coeff_density[idx].x += gamma[0]/2.0 * (2.0 * (psi_1[idx].x*psi_1_d[idx].x + psi_1[idx].y*psi_1_d[idx].y) * (sqr(psi_2[idx].x) + sqr(psi_2[idx].y)) +  
                                                 2.0 * (psi_2[idx].x*psi_2_d[idx].x + psi_2[idx].y*psi_2_d[idx].y) * (sqr(psi_1[idx].x) + sqr(psi_1[idx].y)) ) * area_factor*dx*dx;

    poly_coeff_density[idx].y += gamma[0]/2.0 * (4.0 * (psi_1[idx].x*psi_1_d[idx].x + psi_1[idx].y*psi_1_d[idx].y) * (psi_2[idx].x*psi_2_d[idx].x + psi_2[idx].y*psi_2_d[idx].y) +
                                                       (sqr(psi_1[idx].x) + sqr(psi_1[idx].y)) * (sqr(psi_2_d[idx].x) + sqr(psi_2_d[idx].y)) + 
                                                       (sqr(psi_2[idx].x) + sqr(psi_2[idx].y)) * (sqr(psi_1_d[idx].x) + sqr(psi_1_d[idx].y)) ) * area_factor*dx*dx;  
                                                         
    poly_coeff_density[idx].z += gamma[0]/2.0 * (2.0 * (psi_1[idx].x*psi_1_d[idx].x + psi_1[idx].y*psi_1_d[idx].y) * (sqr(psi_2_d[idx].x) + sqr(psi_2_d[idx].y)) +
                                                 2.0 * (psi_2[idx].x*psi_2_d[idx].x + psi_2[idx].y*psi_2_d[idx].y) * (sqr(psi_1_d[idx].x) + sqr(psi_1_d[idx].y)) ) * area_factor*dx*dx;

    poly_coeff_density[idx].w += gamma[0]/2.0 * ( (sqr(psi_1_d[idx].x) + sqr(psi_1_d[idx].y)) * (sqr(psi_2_d[idx].x) + sqr(psi_2_d[idx].y)) ) * area_factor*dx*dx;
    #endif

    #ifdef BIQUADRATIC_JOSEPHSON_COUPLING
    fenergy_density[idx]      +=  delta[0]/2.0 * (+ sqr(psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y) 
                                                  - sqr(psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x) 
                                                 ) * area_factor*dx*dx;

    poly_coeff_density[idx].x +=  delta[0]/2.0 * ( + 2.0 * (+ psi_1[idx].x*psi_2_d[idx].x + psi_1_d[idx].x*psi_2[idx].x 
                                                         + psi_1[idx].y*psi_2_d[idx].y + psi_1_d[idx].y*psi_2[idx].y) * (psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y)

                                                   - 2.0 * (+ psi_1[idx].x*psi_2_d[idx].y + psi_1_d[idx].x*psi_2[idx].y 
                                                            - psi_1[idx].y*psi_2_d[idx].x - psi_1_d[idx].y*psi_2[idx].x) * (psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x)
                                                 ) * area_factor*dx*dx;

    poly_coeff_density[idx].y +=  delta[0]/2.0 * ( + sqr(psi_1[idx].x*psi_2_d[idx].x + psi_1_d[idx].x*psi_2[idx].x + psi_1[idx].y*psi_2_d[idx].y + psi_1_d[idx].y*psi_2[idx].y) 
                                                   - sqr(psi_1[idx].x*psi_2_d[idx].y + psi_1_d[idx].x*psi_2[idx].y - psi_1[idx].y*psi_2_d[idx].x - psi_1_d[idx].y*psi_2[idx].x)

                                                   + 2.0 * (psi_1[idx].x * psi_2[idx].x + psi_1[idx].y * psi_2[idx].y) * (psi_1_d[idx].x * psi_2_d[idx].x + psi_1_d[idx].y * psi_2_d[idx].y)
                                                   - 2.0 * (psi_1[idx].x * psi_2[idx].y - psi_1[idx].y * psi_2[idx].x) * (psi_1_d[idx].x * psi_2_d[idx].y - psi_1_d[idx].y * psi_2_d[idx].x)
                                                 ) * area_factor*dx*dx;  
                                                         
    poly_coeff_density[idx].z +=  delta[0]/2.0 * ( + 2.0 * (psi_1[idx].x*psi_2_d[idx].x + psi_1_d[idx].x*psi_2[idx].x + psi_1[idx].y*psi_2_d[idx].y + psi_1_d[idx].y*psi_2[idx].y) * (psi_1_d[idx].x * psi_2_d[idx].x + psi_1_d[idx].y * psi_2_d[idx].y) 
                                                   - 2.0 * (psi_1[idx].x*psi_2_d[idx].y + psi_1_d[idx].x*psi_2[idx].y - psi_1[idx].y*psi_2_d[idx].x - psi_1_d[idx].y*psi_2[idx].x) * (psi_1_d[idx].x * psi_2_d[idx].y - psi_1_d[idx].y * psi_2_d[idx].x) 
                                                 ) * area_factor*dx*dx; 

    poly_coeff_density[idx].w +=  delta[0]/2.0 * (+ sqr(psi_1_d[idx].x * psi_2_d[idx].x + psi_1_d[idx].y * psi_2_d[idx].y) 
                                                  - sqr(psi_1_d[idx].x * psi_2_d[idx].y - psi_1_d[idx].y * psi_2_d[idx].x) 
                                                 ) * area_factor*dx*dx;
    #endif
    #endif
    #endif

    // EM Term
    #if !defined(NO_EM) && !defined(NO_SELF_FIELD)
    real em_term[2];
    if(int2bit(comp_nn_map[idx_0], 0) && int2bit(comp_nn_map[idx_0], 1) && int2bit(comp_nn_map[idx_0], 2)){
        em_term[0] = ((a[idx+1].y - a[idx].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);
        em_term[1] = ((a_d[idx+1].y - a_d[idx].y)/dx - (a_d[idx+N].x - a_d[idx].x)/dx);

        fenergy_density[idx] +=      ( 0.5*sqr(em_term[0])                         )*0.25*dx*dx;
        poly_coeff_density[idx].x += (                       em_term[0]*em_term[1] )*0.25*dx*dx;
        poly_coeff_density[idx].y += ( 0.5*sqr(em_term[1])                         )*0.25*dx*dx;
    }
    
    if(int2bit(comp_nn_map[idx_0], 2) && int2bit(comp_nn_map[idx_0],3) && int2bit(comp_nn_map[idx_0], 4)){
        em_term[0] = ((a[idx].y - a[idx-1].y)/dx - (a[idx+N].x - a[idx].x)/dx - h[idx_0]);
        em_term[1] = ((a_d[idx].y - a_d[idx-1].y)/dx - (a_d[idx+N].x - a_d[idx].x)/dx);

        fenergy_density[idx] +=      ( 0.5*sqr(em_term[0])                         )*0.25*dx*dx;
        poly_coeff_density[idx].x += (                       em_term[0]*em_term[1] )*0.25*dx*dx;
        poly_coeff_density[idx].y += ( 0.5*sqr(em_term[1])                         )*0.25*dx*dx;
    }

    if(int2bit(comp_nn_map[idx_0], 4) && int2bit(comp_nn_map[idx_0], 5) && int2bit(comp_nn_map[idx_0], 6)){
        em_term[0] = ((a[idx].y - a[idx-1].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);
        em_term[1] = ((a_d[idx].y - a_d[idx-1].y)/dx - (a_d[idx].x - a_d[idx-N].x)/dx);

        fenergy_density[idx] +=      ( 0.5*sqr(em_term[0])                         )*0.25*dx*dx;
        poly_coeff_density[idx].x += (                       em_term[0]*em_term[1] )*0.25*dx*dx;
        poly_coeff_density[idx].y += ( 0.5*sqr(em_term[1])                         )*0.25*dx*dx;
    }

    if(int2bit(comp_nn_map[idx_0], 6) && int2bit(comp_nn_map[idx_0], 7) &&  int2bit(comp_nn_map[idx_0], 0)){
        em_term[0] = ((a[idx+1].y - a[idx].y)/dx - (a[idx].x - a[idx-N].x)/dx - h[idx_0]);
        em_term[1] = ((a_d[idx+1].y - a_d[idx].y)/dx - (a_d[idx].x - a_d[idx-N].x)/dx);

        fenergy_density[idx] +=      ( 0.5*sqr(em_term[0])                         )*0.25*dx*dx;
        poly_coeff_density[idx].x += (                       em_term[0]*em_term[1] )*0.25*dx*dx;
        poly_coeff_density[idx].y += ( 0.5*sqr(em_term[1])                         )*0.25*dx*dx;
    }
    #endif 
    
    // Kinetic term
    #ifndef NO_KINETIC
    real coeff_x = (int2bit(sc_nn_map[idx_0], 1)*int2bit(sc_nn_map[idx_0], 2) + int2bit(sc_nn_map[idx_0], 6)*int2bit(sc_nn_map[idx_0], 7))/2.0f;
    real coeff_y = (int2bit(sc_nn_map[idx_0], 0)*int2bit(sc_nn_map[idx_0], 1) + int2bit(sc_nn_map[idx_0], 3)*int2bit(sc_nn_map[idx_0], 4))/2.0f;

    real kinetic_term_x[4][3];
    real kinetic_term_y[4][3];
    
    if(int2bit(sc_nn_map[idx_0], 0)){
        kinetic_term_x[0][0] = (-(psi_1[idx + 1].x - psi_1[idx].x)/dx - q_1[0]*a[idx    ].x*psi_1[idx    ].y) ;
        kinetic_term_x[1][0] = (-(psi_1[idx + 1].x - psi_1[idx].x)/dx - q_1[0]*a[idx + 1].x*psi_1[idx + 1].y) ;
        kinetic_term_x[2][0] = (+(psi_1[idx + 1].y - psi_1[idx].y)/dx - q_1[0]*a[idx    ].x*psi_1[idx    ].x) ; 
        kinetic_term_x[3][0] = (+(psi_1[idx + 1].y - psi_1[idx].y)/dx - q_1[0]*a[idx + 1].x*psi_1[idx + 1].x) ;

        kinetic_term_x[0][1] = (-(psi_1_d[idx + 1].x - psi_1_d[idx].x)/dx - q_1[0]*a[idx    ].x*psi_1_d[idx    ].y + a_d[idx    ].x*psi_1[idx    ].y) ;
        kinetic_term_x[1][1] = (-(psi_1_d[idx + 1].x - psi_1_d[idx].x)/dx - q_1[0]*a[idx + 1].x*psi_1_d[idx + 1].y + a_d[idx + 1].x*psi_1[idx + 1].y) ;
        kinetic_term_x[2][1] = (+(psi_1_d[idx + 1].y - psi_1_d[idx].y)/dx - q_1[0]*a[idx    ].x*psi_1_d[idx    ].x + a_d[idx    ].x*psi_1[idx    ].x) ; 
        kinetic_term_x[3][1] = (+(psi_1_d[idx + 1].y - psi_1_d[idx].y)/dx - q_1[0]*a[idx + 1].x*psi_1_d[idx + 1].x + a_d[idx + 1].x*psi_1[idx + 1].x) ;

        kinetic_term_x[0][2] = -q_1[0]*a_d[idx    ].x*psi_1_d[idx    ].y;
        kinetic_term_x[1][2] = -q_1[0]*a_d[idx + 1].x*psi_1_d[idx + 1].y;
        kinetic_term_x[2][2] = -q_1[0]*a_d[idx    ].x*psi_1_d[idx    ].x; 
        kinetic_term_x[3][2] = -q_1[0]*a_d[idx + 1].x*psi_1_d[idx + 1].x;
        
        fenergy_density[idx] += (
            + 0.5*sqr(kinetic_term_x[0][0])
            + 0.5*sqr(kinetic_term_x[1][0])
            + 0.5*sqr(kinetic_term_x[2][0])
            + 0.5*sqr(kinetic_term_x[3][0])
            )/(2*m_xx_1[idx_0])*coeff_x*dx*dx;

        poly_coeff_density[idx].x += (
                + (kinetic_term_x[0][0])*(kinetic_term_x[0][1])
                + (kinetic_term_x[1][0])*(kinetic_term_x[1][1])
                + (kinetic_term_x[2][0])*(kinetic_term_x[2][1])
                + (kinetic_term_x[3][0])*(kinetic_term_x[3][1])
                )/(2*m_xx_1[idx_0])*coeff_x*dx*dx;

        poly_coeff_density[idx].y += (
                + 0.5*sqr(kinetic_term_x[0][1]) + (kinetic_term_x[0][0])*(kinetic_term_x[0][2])
                + 0.5*sqr(kinetic_term_x[1][1]) + (kinetic_term_x[1][0])*(kinetic_term_x[1][2])
                + 0.5*sqr(kinetic_term_x[2][1]) + (kinetic_term_x[2][0])*(kinetic_term_x[2][2])
                + 0.5*sqr(kinetic_term_x[3][1]) + (kinetic_term_x[3][0])*(kinetic_term_x[3][2])
                )/(2*m_xx_1[idx_0])*coeff_x*dx*dx;
    
        poly_coeff_density[idx].z += (
                + (kinetic_term_x[0][1])*(kinetic_term_x[0][2])
                + (kinetic_term_x[1][1])*(kinetic_term_x[1][2])
                + (kinetic_term_x[2][1])*(kinetic_term_x[2][2])
                + (kinetic_term_x[3][1])*(kinetic_term_x[3][2])
                )/(2*m_xx_1[idx_0])*coeff_x*dx*dx;

        poly_coeff_density[idx].w += (
                + 0.5*sqr(kinetic_term_x[0][2])
                + 0.5*sqr(kinetic_term_x[1][2])
                + 0.5*sqr(kinetic_term_x[2][2])
                + 0.5*sqr(kinetic_term_x[3][2])
                )/(2*m_xx_1[idx_0])*coeff_x*dx*dx;                
    }

        
    if(int2bit(sc_nn_map[idx_0], 2)){
        kinetic_term_y[0][0] = (-(psi_1[idx + N].x - psi_1[idx].x)/dx - q_1[0]*a[idx    ].y*psi_1[idx    ].y) ;
        kinetic_term_y[1][0] = (-(psi_1[idx + N].x - psi_1[idx].x)/dx - q_1[0]*a[idx + N].y*psi_1[idx + N].y) ;
        kinetic_term_y[2][0] = (+(psi_1[idx + N].y - psi_1[idx].y)/dx - q_1[0]*a[idx    ].y*psi_1[idx    ].x) ; 
        kinetic_term_y[3][0] = (+(psi_1[idx + N].y - psi_1[idx].y)/dx - q_1[0]*a[idx + N].y*psi_1[idx + N].x) ;

        kinetic_term_y[0][1] = (-(psi_1_d[idx + N].x - psi_1_d[idx].x)/dx - q_1[0]*a[idx    ].y*psi_1_d[idx    ].y - q_1[0]*a_d[idx    ].y*psi_1[idx    ].y) ;
        kinetic_term_y[1][1] = (-(psi_1_d[idx + N].x - psi_1_d[idx].x)/dx - q_1[0]*a[idx + N].y*psi_1_d[idx + N].y - q_1[0]*a_d[idx + N].y*psi_1[idx + N].y) ;
        kinetic_term_y[2][1] = (+(psi_1_d[idx + N].y - psi_1_d[idx].y)/dx - q_1[0]*a[idx    ].y*psi_1_d[idx    ].x - q_1[0]*a_d[idx    ].y*psi_1[idx    ].x) ; 
        kinetic_term_y[3][1] = (+(psi_1_d[idx + N].y - psi_1_d[idx].y)/dx - q_1[0]*a[idx + N].y*psi_1_d[idx + N].x - q_1[0]*a_d[idx + N].y*psi_1[idx + N].x) ;

        kinetic_term_y[0][2] = -q_1[0]*a_d[idx    ].y*psi_1_d[idx    ].y;
        kinetic_term_y[1][2] = -q_1[0]*a_d[idx + N].y*psi_1_d[idx + N].y;
        kinetic_term_y[2][2] = -q_1[0]*a_d[idx    ].y*psi_1_d[idx    ].x; 
        kinetic_term_y[3][2] = -q_1[0]*a_d[idx + N].y*psi_1_d[idx + N].x;

        fenergy_density[idx] += (
            + 0.5*sqr(kinetic_term_y[0][0])
            + 0.5*sqr(kinetic_term_y[1][0])
            + 0.5*sqr(kinetic_term_y[2][0])
            + 0.5*sqr(kinetic_term_y[3][0])
            )/(2*m_yy_1[idx_0])*coeff_y*dx*dx;

        poly_coeff_density[idx].x += (
            + (kinetic_term_y[0][0])*(kinetic_term_y[0][1])
            + (kinetic_term_y[1][0])*(kinetic_term_y[1][1])
            + (kinetic_term_y[2][0])*(kinetic_term_y[2][1])
            + (kinetic_term_y[3][0])*(kinetic_term_y[3][1])
            )/(2*m_yy_1[idx_0])*coeff_y*dx*dx;

        poly_coeff_density[idx].y += (
            + 0.5*sqr(kinetic_term_y[0][1]) + (kinetic_term_y[0][0])*(kinetic_term_y[0][2])
            + 0.5*sqr(kinetic_term_y[1][1]) + (kinetic_term_y[1][0])*(kinetic_term_y[1][2])
            + 0.5*sqr(kinetic_term_y[2][1]) + (kinetic_term_y[2][0])*(kinetic_term_y[2][2])
            + 0.5*sqr(kinetic_term_y[3][1]) + (kinetic_term_y[3][0])*(kinetic_term_y[3][2])
            )/(2*m_yy_1[idx_0])*coeff_y*dx*dx;
    
        poly_coeff_density[idx].z += (
                + (kinetic_term_y[0][1])*(kinetic_term_y[0][2])
                + (kinetic_term_y[1][1])*(kinetic_term_y[1][2])
                + (kinetic_term_y[2][1])*(kinetic_term_y[2][2])
                + (kinetic_term_y[3][1])*(kinetic_term_y[3][2])
                )/(2*m_yy_1[idx_0])*coeff_y*0.5*dx*dx;   
        
        poly_coeff_density[idx].w += (
                + 0.5*sqr(kinetic_term_y[0][2])
                + 0.5*sqr(kinetic_term_y[1][2])
                + 0.5*sqr(kinetic_term_y[2][2])
                + 0.5*sqr(kinetic_term_y[3][2])
                )/(2*m_yy_1[idx_0])*coeff_y*dx*dx;   
    }

    #ifdef MULTICOMPONENT

    if(int2bit(sc_nn_map[idx_0], 0)){
        kinetic_term_x[0][0] = (-(psi_2[idx + 1].x - psi_2[idx].x)/dx - q_2[0]*a[idx    ].x*psi_2[idx    ].y) ;
        kinetic_term_x[1][0] = (-(psi_2[idx + 1].x - psi_2[idx].x)/dx - q_2[0]*a[idx + 1].x*psi_2[idx + 1].y) ;
        kinetic_term_x[2][0] = (+(psi_2[idx + 1].y - psi_2[idx].y)/dx - q_2[0]*a[idx    ].x*psi_2[idx    ].x) ; 
        kinetic_term_x[3][0] = (+(psi_2[idx + 1].y - psi_2[idx].y)/dx - q_2[0]*a[idx + 1].x*psi_2[idx + 1].x) ;

        kinetic_term_x[0][1] = (-(psi_2_d[idx + 1].x - psi_2_d[idx].x)/dx - q_2[0]*a[idx    ].x*psi_2_d[idx    ].y + a_d[idx    ].x*psi_2[idx    ].y) ;
        kinetic_term_x[1][1] = (-(psi_2_d[idx + 1].x - psi_2_d[idx].x)/dx - q_2[0]*a[idx + 1].x*psi_2_d[idx + 1].y + a_d[idx + 1].x*psi_2[idx + 1].y) ;
        kinetic_term_x[2][1] = (+(psi_2_d[idx + 1].y - psi_2_d[idx].y)/dx - q_2[0]*a[idx    ].x*psi_2_d[idx    ].x + a_d[idx    ].x*psi_2[idx    ].x) ; 
        kinetic_term_x[3][1] = (+(psi_2_d[idx + 1].y - psi_2_d[idx].y)/dx - q_2[0]*a[idx + 1].x*psi_2_d[idx + 1].x + a_d[idx + 1].x*psi_2[idx + 1].x) ;

        kinetic_term_x[0][2] = -q_2[0]*a_d[idx    ].x*psi_2_d[idx    ].y;
        kinetic_term_x[1][2] = -q_2[0]*a_d[idx + 1].x*psi_2_d[idx + 1].y;
        kinetic_term_x[2][2] = -q_2[0]*a_d[idx    ].x*psi_2_d[idx    ].x; 
        kinetic_term_x[3][2] = -q_2[0]*a_d[idx + 1].x*psi_2_d[idx + 1].x;
        
        fenergy_density[idx] += (
            + 0.5*sqr(kinetic_term_x[0][0])
            + 0.5*sqr(kinetic_term_x[1][0])
            + 0.5*sqr(kinetic_term_x[2][0])
            + 0.5*sqr(kinetic_term_x[3][0])
            )/(2*m_xx_2[idx_0])*coeff_x*dx*dx;

        poly_coeff_density[idx].x += (
                + (kinetic_term_x[0][0])*(kinetic_term_x[0][1])
                + (kinetic_term_x[1][0])*(kinetic_term_x[1][1])
                + (kinetic_term_x[2][0])*(kinetic_term_x[2][1])
                + (kinetic_term_x[3][0])*(kinetic_term_x[3][1])
                )/(2*m_xx_2[idx_0])*coeff_x*dx*dx;

        poly_coeff_density[idx].y += (
                + 0.5*sqr(kinetic_term_x[0][1]) + (kinetic_term_x[0][0])*(kinetic_term_x[0][2])
                + 0.5*sqr(kinetic_term_x[1][1]) + (kinetic_term_x[1][0])*(kinetic_term_x[1][2])
                + 0.5*sqr(kinetic_term_x[2][1]) + (kinetic_term_x[2][0])*(kinetic_term_x[2][2])
                + 0.5*sqr(kinetic_term_x[3][1]) + (kinetic_term_x[3][0])*(kinetic_term_x[3][2])
                )/(2*m_xx_2[idx_0])*coeff_x*dx*dx;
    
        poly_coeff_density[idx].z += (
                + (kinetic_term_x[0][1])*(kinetic_term_x[0][2])
                + (kinetic_term_x[1][1])*(kinetic_term_x[1][2])
                + (kinetic_term_x[2][1])*(kinetic_term_x[2][2])
                + (kinetic_term_x[3][1])*(kinetic_term_x[3][2])
                )/(2*m_xx_2[idx_0])*coeff_x*dx*dx;

        poly_coeff_density[idx].w += (
                + 0.5*sqr(kinetic_term_x[0][2])
                + 0.5*sqr(kinetic_term_x[1][2])
                + 0.5*sqr(kinetic_term_x[2][2])
                + 0.5*sqr(kinetic_term_x[3][2])
                )/(2*m_xx_2[idx_0])*coeff_x*dx*dx;                
    }

        
    if(int2bit(sc_nn_map[idx_0], 2)){
        kinetic_term_y[0][0] = (-(psi_2[idx + N].x - psi_2[idx].x)/dx - q_2[0]*a[idx    ].y*psi_2[idx    ].y) ;
        kinetic_term_y[1][0] = (-(psi_2[idx + N].x - psi_2[idx].x)/dx - q_2[0]*a[idx + N].y*psi_2[idx + N].y) ;
        kinetic_term_y[2][0] = (+(psi_2[idx + N].y - psi_2[idx].y)/dx - q_2[0]*a[idx    ].y*psi_2[idx    ].x) ; 
        kinetic_term_y[3][0] = (+(psi_2[idx + N].y - psi_2[idx].y)/dx - q_2[0]*a[idx + N].y*psi_2[idx + N].x) ;

        kinetic_term_y[0][1] = (-(psi_2_d[idx + N].x - psi_2_d[idx].x)/dx - q_2[0]*a[idx    ].y*psi_2_d[idx    ].y - q_2[0]*a_d[idx    ].y*psi_2[idx    ].y) ;
        kinetic_term_y[1][1] = (-(psi_2_d[idx + N].x - psi_2_d[idx].x)/dx - q_2[0]*a[idx + N].y*psi_2_d[idx + N].y - q_2[0]*a_d[idx + N].y*psi_2[idx + N].y) ;
        kinetic_term_y[2][1] = (+(psi_2_d[idx + N].y - psi_2_d[idx].y)/dx - q_2[0]*a[idx    ].y*psi_2_d[idx    ].x - q_2[0]*a_d[idx    ].y*psi_2[idx    ].x) ; 
        kinetic_term_y[3][1] = (+(psi_2_d[idx + N].y - psi_2_d[idx].y)/dx - q_2[0]*a[idx + N].y*psi_2_d[idx + N].x - q_2[0]*a_d[idx + N].y*psi_2[idx + N].x) ;

        kinetic_term_y[0][2] = -q_2[0]*a_d[idx    ].y*psi_2_d[idx    ].y;
        kinetic_term_y[1][2] = -q_2[0]*a_d[idx + N].y*psi_2_d[idx + N].y;
        kinetic_term_y[2][2] = -q_2[0]*a_d[idx    ].y*psi_2_d[idx    ].x; 
        kinetic_term_y[3][2] = -q_2[0]*a_d[idx + N].y*psi_2_d[idx + N].x;

        fenergy_density[idx] += (
            + 0.5*sqr(kinetic_term_y[0][0])
            + 0.5*sqr(kinetic_term_y[1][0])
            + 0.5*sqr(kinetic_term_y[2][0])
            + 0.5*sqr(kinetic_term_y[3][0])
            )/(2*m_yy_2[idx_0])*coeff_y*dx*dx;

        poly_coeff_density[idx].x += (
            + (kinetic_term_y[0][0])*(kinetic_term_y[0][1])
            + (kinetic_term_y[1][0])*(kinetic_term_y[1][1])
            + (kinetic_term_y[2][0])*(kinetic_term_y[2][1])
            + (kinetic_term_y[3][0])*(kinetic_term_y[3][1])
            )/(2*m_yy_2[idx_0])*coeff_y*dx*dx;

        poly_coeff_density[idx].y += (
            + 0.5*sqr(kinetic_term_y[0][1]) + (kinetic_term_y[0][0])*(kinetic_term_y[0][2])
            + 0.5*sqr(kinetic_term_y[1][1]) + (kinetic_term_y[1][0])*(kinetic_term_y[1][2])
            + 0.5*sqr(kinetic_term_y[2][1]) + (kinetic_term_y[2][0])*(kinetic_term_y[2][2])
            + 0.5*sqr(kinetic_term_y[3][1]) + (kinetic_term_y[3][0])*(kinetic_term_y[3][2])
            )/(2*m_yy_2[idx_0])*coeff_y*dx*dx;
    
        poly_coeff_density[idx].z += (
                + (kinetic_term_y[0][1])*(kinetic_term_y[0][2])
                + (kinetic_term_y[1][1])*(kinetic_term_y[1][2])
                + (kinetic_term_y[2][1])*(kinetic_term_y[2][2])
                + (kinetic_term_y[3][1])*(kinetic_term_y[3][2])
                )/(2*m_yy_2[idx_0])*coeff_y*0.5*dx*dx;   
        
        poly_coeff_density[idx].w += (
                + 0.5*sqr(kinetic_term_y[0][2])
                + 0.5*sqr(kinetic_term_y[1][2])
                + 0.5*sqr(kinetic_term_y[2][2])
                + 0.5*sqr(kinetic_term_y[3][2])
                )/(2*m_yy_2[idx_0])*coeff_y*dx*dx;   
    }

    #endif
    #endif
}

__global__
void updateFields(real2 *a, real2 *psi_1, real2 *psi_2,
                  real2 *a_d, real2 *psi_1_d, real2 *psi_2_d,
                  real *alpha){

    // Update all the field moving in the computed direction with computed step 

    int idx =  blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx >= N*N){
        return;
    }

    idx += N*N*blockIdx.y;

    a[idx].x = a[idx].x + alpha[blockIdx.y] * a_d[idx].x;
    a[idx].y = a[idx].y + alpha[blockIdx.y] * a_d[idx].y;
    psi_1[idx].x = psi_1[idx].x + alpha[blockIdx.y] * psi_1_d[idx].x;
    psi_1[idx].y = psi_1[idx].y + alpha[blockIdx.y] * psi_1_d[idx].y;
    
    #ifdef MULTICOMPONENT
    psi_2[idx].x = psi_2[idx].x + alpha[blockIdx.y] * psi_2_d[idx].x;
    psi_2[idx].y = psi_2[idx].y + alpha[blockIdx.y] * psi_2_d[idx].y;
    #endif
}
 
__global__
void lineSearch(real* fenergy, real4* poly_coeff, real *alpha){

    // Perform line search and compute system energy

    int idx = blockIdx.y;

    #ifndef ALPHA
    
    real A = fenergy[idx];
    real B = poly_coeff[idx].x;
    real C = poly_coeff[idx].y;
    real D = poly_coeff[idx].z;
    real E = poly_coeff[idx].w;
    
    // Iterative search of optimal step length
    // We compute free energy along the chosen direction as a function of the step length until we get to the minimum
    
    // Start with a very small alpha
    real x = 1e-10;
    // Every iteration multiply it by 
    const real multiplier = 1.025f;
    // Maximum numer of try
    const int max_iter = 2000; 

    real f_old = E*pow(x, 4) + D*x*x*x + C*x*x + B*x + A;
    real f_new = 0; 

    // Update alpha until f(alpha) is decreasing
    for(int p = 0; p < max_iter; p++){
        x *= multiplier;
        f_new = E*pow(x, 4) + D*x*x*x + C*x*x + B*x + A;
        if (f_new > f_old)
            break;
        else
            f_old = f_new;
    }

    alpha[idx] = x/multiplier; 

    #else
    
    alpha[idx] = ALPHA;

    #endif

    // printf("B = %.10e\n", B);
}

__global__
void setCountersToZero(real *g_norm2, real *g_delta_g, real *fenergy, real4 *poly_coeff){
    
    int n = blockIdx.y;

    g_norm2[n] = 0;
    g_delta_g[n] = 0;
    fenergy[n] = 0;
    poly_coeff[n].x = 0;
    poly_coeff[n].y = 0;
    poly_coeff[n].z = 0;
    poly_coeff[n].w = 0;

}

__host__
void nlcgSteps(
                int iter_n, int F, int *dev_comp_nn_map, int* dev_sc_nn_map,
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
                ){

    for(int k = 0; k < iter_n; k++){

        // Print the substep number
        printf(" %d ", k + 1);

        // Swap pointer of gradients
        real* temp = g_norm2_old;
        g_norm2_old = g_norm2;
        g_norm2 = temp;
        
        real2* temp_g = dev_a_g_old;
        dev_a_g_old = dev_a_g;
        dev_a_g = temp_g;

        temp_g = dev_psi_1_g_old;
        dev_psi_1_g_old = dev_psi_1_g;
        dev_psi_1_g = temp_g;        
            
        temp_g = dev_psi_2_g_old;
        dev_psi_2_g_old = dev_psi_2_g;
        dev_psi_2_g = temp_g;        

        // Set counters to zero 
        setCountersToZero<<< dim3(1, F), dim3(1, 1) >>>(g_norm2, g_delta_g, fenergy, poly_coeff);
        checkCudaErrors(cudaDeviceSynchronize());  

        
        // Compute Free Energy 
        // computeFreeEnergy(F, dev_comp_nn_map, dev_sc_nn_map, 
        //     dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
        //     dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
        //     dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma,
        //     dev_a, dev_psi_1, dev_psi_2,
        //     fenergy);

        //     printf("System |  Free Energy   | \n");        
        //     printf("------------------------- \n");
        //     for(int n = 0; n<F; n++) 
        //         printf("   %3.3d | %.8e | \n", n, fenergy[n]);
        // setCountersToZero<<< dim3(1, F), dim3(1, 1) >>>(g_norm2, g_delta_g, fenergy, poly_coeff);
        // checkCudaErrors(cudaDeviceSynchronize());  

        // Compute gradient
        if(! maxwell_solver )
            gradientPsi <<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_comp_nn_map, dev_sc_nn_map, 
                                                                  dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                                                                  dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                                                                  dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                                                                  dev_a, dev_psi_1, dev_psi_2,
                                                                  dev_a_g, dev_psi_1_g, dev_psi_2_g);
                    
    
        #ifndef NO_SELF_FIELD
        gradientA <<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_comp_nn_map, dev_sc_nn_map, 
                                                            dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                                                            dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                                                            dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                                                            dev_a, dev_psi_1, dev_psi_2,
                                                            dev_a_g, dev_psi_1_g, dev_psi_2_g);
        #endif

        checkCudaErrors(cudaDeviceSynchronize());  

        // Compute gradient norm
        // sumGradientAtomic<<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_a_g, dev_psi_g, dev_a_g_old, dev_psi_g_old, g_norm2, g_delta_g);
        gradientDensity<<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_a_g, dev_psi_1_g, dev_psi_2_g, 
                                                                 dev_a_g_old, dev_psi_1_g_old, dev_psi_2_g_old, 
                                                                 dev_g_norm2_density, dev_g_delta_g_density);
        checkCudaErrors(cudaDeviceSynchronize());
        parallel_sum::sumArrayF(dev_g_norm2_density, g_norm2, F, N*N);
        parallel_sum::sumArrayF(dev_g_delta_g_density, g_delta_g, F, N*N);
        checkCudaErrors(cudaDeviceSynchronize());

        // Compute direction
        computeStepDirection<<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_a, dev_psi_1, dev_psi_2, 
                                                                      dev_a_g, dev_psi_1_g, dev_psi_2_g, 
                                                                      dev_a_d, dev_psi_1_d, dev_psi_2_d, 
                                                                      g_norm2, g_norm2_old, g_delta_g, nlcg);
        checkCudaErrors(cudaDeviceSynchronize());

        // Polynomial expansion
        polynomialExpansion<<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>(dev_comp_nn_map, dev_sc_nn_map,
                                                                     dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                                                                     dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                                                                     dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                                                                     dev_a, dev_psi_1, dev_psi_2,
                                                                     dev_a_d, dev_psi_1_d, dev_psi_2_d,
                                                                     dev_fenergy_density, dev_poly_coeff_density);
        checkCudaErrors(cudaDeviceSynchronize());
        parallel_sum::sumArrayF(dev_fenergy_density, fenergy, F, N*N);
        parallel_sum::sumArrayF4(dev_poly_coeff_density, poly_coeff, F, N*N);
        checkCudaErrors(cudaDeviceSynchronize()); 
        // printf("System |  Free Energy   | \n");        
        // printf("------------------------- \n");
        // for(int n = 0; n<F; n++) 
        //     printf("   %3.3d | %.8e | \n", n, fenergy[n]);

        // Linesearch
        lineSearch<<< dim3(1, F), dim3(1, 1) >>>(fenergy, poly_coeff, alpha);
        checkCudaErrors(cudaDeviceSynchronize());  
        
        // Update step
        updateFields <<< dim3(gridSizeX, F), dim3(blockSizeX, 1) >>>( dev_a, dev_psi_1, dev_psi_2, 
                                                                dev_a_d, dev_psi_1_d, dev_psi_2_d, alpha);  
        checkCudaErrors(cudaDeviceSynchronize());         
    }

    printf("\n");
}
