/************************************************
 *
 * Cubic spline interpolation code
 *
 * For all the code below please refer to
 * "Computer methods for mathematical computations" by G. E. Forsythe sec 4.4
 *
 * - h[n] =s[n]-s[n-1] are the delta x of the book
 * - H and Delta are the interpolation matricies defined at page 74
 * - Sigma are the derived coefficients which can be used to generate new points
 *
 * N.B.
 * Delta has to be built in column-major format in order to have a linear system to be solved
 * by cusparseSgtsv2 but the fields are stored in row-major format.
 * 
 * Therefore there is a transposition operations to be done implicitly in computeDelta functions
 *
 * idx   = i + j * N + n * N * N = idx + n * N * N
 * idx'  = n + idx*F
 *********************************************************************************/

#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "cuda_errors.cuh"

#include "real.cuh"
#include "interpolation.cuh"
#include "config.cuh"

__device__ inline real sqr(real x) { return x*x; }
__device__ inline real cub(real x) { return x*x*x; }

/***************************** Arc Length code ********************************/

__global__
void arcLengthParametrization(int N, int F, real2 *a, real2 *psi_1, real2 *psi_2, real* delta_s2){

    // For each point in each frame compute the delta_s^2 (modulus squared of the difference)
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if ((idx >= N*N)||(blockIdx.y == 0)){
        return;
    }

    idx += N*N*blockIdx.y;

    #ifdef MULTICOMPONENT
    atomicAdd(&delta_s2[blockIdx.y], sqr((a[idx].x - a[idx - N*N].x)) +
                                     sqr((a[idx].y - a[idx - N*N].y)) +
                                     sqr((psi_1[idx].x - psi_1[idx - N*N].x)) +
                                     sqr((psi_1[idx].y - psi_1[idx - N*N].y)) +
                                     sqr((psi_2[idx].x - psi_2[idx - N*N].x)) +
                                     sqr((psi_2[idx].y - psi_2[idx - N*N].y)) ); 
    #else
    atomicAdd(&delta_s2[blockIdx.y], sqr((a[idx].x - a[idx - N*N].x)) +
                                     sqr((a[idx].y - a[idx - N*N].y)) +
                                     sqr((psi_1[idx].x - psi_1[idx - N*N].x)) +
                                     sqr((psi_1[idx].y - psi_1[idx - N*N].y)));  
    #endif
}


__global__
void arcLengthParametrizationJB(int N, int F, real *b, real2 *j_1, real2 *j_2, real* delta_s2){

    // For each point in each frame compute the delta_s^2 (modulus squared of the difference)
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if ((idx >= N*N)||(blockIdx.y == 0)){
        return;
    }

    idx += N*N*blockIdx.y;

    #ifdef MULTICOMPONENT
    atomicAdd(&delta_s2[blockIdx.y], sqr((b[idx] - b[idx - N*N])) +
                                     sqr((j_1[idx].x - j_1[idx - N*N].x)) +
                                     sqr((j_1[idx].y - j_1[idx - N*N].y)) +
                                     sqr((j_2[idx].x - j_2[idx - N*N].x)) +
                                     sqr((j_2[idx].y - j_2[idx - N*N].y)) ); 
    #else
    atomicAdd(&delta_s2[blockIdx.y], sqr((b[idx] - b[idx - N*N])) +
                                     sqr((j_1[idx].x - j_1[idx - N*N].x)) +
                                     sqr((j_1[idx].y - j_1[idx - N*N].y)) );  
    #endif
}


__global__
void arcLengthParametrizationAbsGrad( int N, int F,
                                      real* psi_1_abs, real2 *cd_1,
                                      real* psi_2_abs, real2 *cd_2,
                                      real* delta_s2){

    // For each point in each frame compute the delta_s^2 (modulus squared of the difference)
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if ((idx >= N*N)||(blockIdx.y == 0)){
        return;
    }

    idx += N*N*blockIdx.y;

    #ifdef MULTICOMPONENT
    atomicAdd(&delta_s2[blockIdx.y], sqr((psi_1_abs[idx] - psi_1_abs[idx - N*N])) +
                                     sqr((cd_1[idx].x - cd_1[idx - N*N].x)) +
                                     sqr((cd_1[idx].y - cd_1[idx - N*N].y)) +
                                     sqr((psi_2_abs[idx] - psi_2_abs[idx - N*N])) +
                                     sqr((cd_2[idx].x - cd_2[idx - N*N].x)) +
                                     sqr((cd_2[idx].y - cd_2[idx - N*N].y)) 
                                     ); 
    #else
    atomicAdd(&delta_s2[blockIdx.y], sqr((psi_1_abs[idx] - psi_1_abs[idx - N*N])) +
                                     sqr((cd_1[idx].x - cd_1[idx - N*N].x)) +
                                     sqr((cd_1[idx].y - cd_1[idx - N*N].y)) );  
    #endif
}

__global__
void computeArcLength(int F, real *s){
    // In s there is actually (Delta s)^2

    //Compute s, the arclength.
    s[0] = 0;
    for(int n = 1; n < F; n++){
        s[n] = s[n-1] + sqrt(s[n]);
    }
    
    // Reparametrization in order to have the have unitary curve length (s_F = 1)
    for(int n = 0; n < F; n++){
        s[n] /= s[F-1];
    }
}


__host__
void arcLength(int N, int F, real *s, real2 *dev_a, real2 *dev_psi_1,  real2 *dev_psi_2){

    // Setting to zero s vector
    checkCudaErrors(cudaMemset(s, 0, F*sizeof(real))); 

    // Filling s with delta_s^2
    arcLengthParametrization<<< dim3(N, F), dim3(N, 1) >>>(N, F, dev_a, dev_psi_1, dev_psi_2, s);
    checkCudaErrors(cudaDeviceSynchronize()); 

    computeArcLength <<< 1,1 >>>(F, s);
    checkCudaErrors(cudaDeviceSynchronize()); 
}

__host__
void arcLengthJB(int N, int F, real *s_jb, real *b, real2 *j_1, real2 *j_2){

    // Setting to zero s vector
    checkCudaErrors(cudaMemset(s_jb, 0, F*sizeof(real))); 

    // Filling s with delta_s^2
    arcLengthParametrizationJB<<< dim3(N, F), dim3(N, 1) >>>(N, F, b, j_1, j_2, s_jb);
    checkCudaErrors(cudaDeviceSynchronize()); 

    computeArcLength <<< 1,1 >>>(F, s_jb);
    checkCudaErrors(cudaDeviceSynchronize()); 
}

__host__
void arcLengthAbsGrad(int N, int F, real *s_ag, real *dev_psi_1_abs, real2 *dev_cd_1,
                                                 real *dev_psi_2_abs, real2 *dev_cd_2){

    // Setting to zero s vector
    checkCudaErrors(cudaMemset(s_ag, 0, F*sizeof(real))); 

    // Filling s with delta_s^2
    arcLengthParametrizationAbsGrad<<< dim3(N, F), dim3(N, 1) >>>(N, F, dev_psi_1_abs, dev_cd_1,
                                                                        dev_psi_2_abs, dev_cd_2,
                                                                        s_ag);
    checkCudaErrors(cudaDeviceSynchronize()); 

    computeArcLength <<< 1,1 >>>(F, s_ag);
    checkCudaErrors(cudaDeviceSynchronize()); 
}

/*********************** Redistribution code ********************************/

__global__
void generateNewFramesLinear(int N, int F, real *s, real2 *f, real F_new, real2 *f_new){

    // This kernel redistribute the value of the x component of f at point idx
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= N*N){
        return;
    }

    // Number of the segment in which we are working
    int k = 1;
    real segment_start = 0;
    real segment_end = s[1];

    for(int n = 0; n < F_new ; n++){

         // New s point at wich we want to generate a new point
        real s_new = (real)n/(F_new - 1);

        // Find the node AT THE END of the segment we are working in
        while(segment_end<s_new){
            k++;
            segment_start = s[k-1];
            segment_end = s[k];
        }
        if(k>=F) k = F-1;
    
    real w = (s_new - segment_start)/(s[k] - s[k-1]);

    f_new[idx + n*N*N].x =  w * f[idx + k*N*N].x +
            (1-w) * f[idx + (k-1)*N*N].x;

    f_new[idx + n*N*N].y =  w * f[idx + k*N*N].y +
            (1-w) * f[idx + (k-1)*N*N].y;
    }
}

__host__
void redistributeLinear(cusparseHandle_t handle, int N, 
                        int F, real *s, real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2,
                        int F_new, real2 *dev_a_new, real2 *dev_psi_1_new,real2 *dev_psi_2_new){

       // Define a function which redistribute the images for each field
    auto redistributeField = [&] (real2 *f, real2 *f_new) -> void {

        const int gridSize = (int)((N*N)/1024 + 1);

        generateNewFramesLinear <<< gridSize, 1024 >>> (N, F, s, f, F_new, f_new);
        checkCudaErrors(cudaDeviceSynchronize());

    };

    // Apply the above function
    redistributeField(dev_a, dev_a_new);
    redistributeField(dev_psi_1, dev_psi_1_new);
    #ifdef MULTICOMPONENT
    redistributeField(dev_psi_2, dev_psi_2_new);
    #endif
}   

__global__ 
void buildHMatrix(int F, real *s, real *h, real *dl, real *d, real *du){

    int idx = threadIdx.x;

    if(idx >= F){
        return;
    }

    /*********************** Compute h vector *******************************/
    
    if(idx == 0){
        h[0] = 0;
    }
    else{
        h[idx] = s[idx] - s[idx-1];
    }
    __syncthreads();

    /*********************** Build the H matrix ****************************/

    if(idx == 0){
        // First row
        du[0] = +h[1];  
        d[0]  = -h[1];
        dl[0] = 0;

        // Last row
        du[F-1] = 0;
        d[F-1]  = -h[F-1];  
        dl[F-1] = +h[F-1];  
    }
    else if(idx<F-1){
        // Central part of the matrix
        dl[idx] = h[idx];
        d[idx]  = 2 * ( h[idx] + h[idx+1] );
        du[idx] = h[idx+1];
    }
}

__global__
void computeDeltaX(int F, int N, real *Delta, real* h, real2 *f){

    // This kernel compute the Delta vector for point idx of the x component of f
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= N*N){
        return;
    }

    // Put in Delta the finite difference
    for(int n = 1; n<F; n++){
        Delta[n + idx*F] = (f[idx + n*N*N].x - f[idx + (n-1)*N*N].x)/h[n];
    }

    // Compute the third devided difference at begging and end of the chain
    real delta_3_start = ((Delta[3 + idx*F] - Delta[2 + idx*F])/(h[3] + h[2]) -
                           (Delta[2 + idx*F] - Delta[1 + idx*F])/(h[2] + h[1]))/(h[3] + h[2] + h[1]);

    real delta_3_end   = ((Delta[(F-1) + idx*F] - Delta[(F-2) + idx*F])/(h[F-1] + h[F-2]) -
                           (Delta[(F-2) + idx*F] - Delta[(F-3) + idx*F])/(h[F-2] + h[F-3]))/(h[F-1] + h[F-2] + h[F-3]);

    // Compute the central part of the Delta vector
    for(int n = 1; n<F-1; n++){
        Delta[n + F*idx] = Delta[(n+1) + F*idx] - Delta[n + F*idx];
    }

    // Set the first and last element
    Delta[0 + F*idx] = sqr(h[1])*delta_3_start;
    Delta[(F-1) + F*idx] = - sqr(h[F-1])*delta_3_end;
}


__global__
void computeDeltaY(int F, int N, real *Delta, real* h, real2 *f){

    // This kernel compute the Delta vector for point idx of the y component of f
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= N*N){
        return;
    }

    // Put in Delta the finite difference
    for(int n = 1; n<F; n++){
        Delta[n + idx*F] = (f[idx + n*N*N].y - f[idx + (n-1)*N*N].y)/h[n];
    }

    // Compute the third devided difference at begging and end of the chain
    real delta_3_start = ((Delta[3 + idx*F] - Delta[2 + idx*F])/(h[3] + h[2]) -
                           (Delta[2 + idx*F] - Delta[1 + idx*F])/(h[2] + h[1]))/(h[3] + h[2] + h[1]);

    real delta_3_end   = ((Delta[(F-1) + idx*F] - Delta[(F-2) + idx*F])/(h[F-1] + h[F-2]) -
                           (Delta[(F-2) + idx*F] - Delta[(F-3) + idx*F])/(h[F-2] + h[F-3]))/(h[F-1] + h[F-2] + h[F-3]);

    // Compute the central part of the Delta vector
    for(int n = 1; n<F-1; n++){
        Delta[n + F*idx] = Delta[(n+1) + F*idx] - Delta[n + F*idx];
    }

    // Set the first and last element
    Delta[0 + F*idx] = sqr(h[1])*delta_3_start;
    Delta[(F-1) + F*idx] = - sqr(h[F-1])*delta_3_end;
}

#ifndef DOUBLE_PRECISION

__global__
void generateNewFramesCubic(int N, int F, real *h, real *SigmaX, real *SigmaY, real2 *f, real F_new, real2 *f_new){

    // This kernel redistribute the value of the x component of f at point idx
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= N*N){
        return;
    }

    // Number of the segment in which we are working
    int k = 1;
    real segment_start = 0;
    real segment_end = h[1];

    for(int n = 0; n < F_new ; n++){

         // New s point at wich we want to generate a new point
        real s_new = (real)n/(F_new - 1);

        // Find the node AT THE END of the segment we are working in
        while(segment_end<s_new){
            k++;
            segment_start += h[k-1];
            segment_end += h[k];
        }
        if(k>=F) k = F-1;
    
    real w = (s_new - segment_start)/h[k];

    f_new[idx + n*N*N].x =  w * f[idx + k*N*N].x +
            (1-w) * f[idx + (k-1)*N*N].x + sqr(h[k]) * 
            ((cub(w  ) -  w   ) * SigmaX[k     + F*idx] +
             (cub(w-1) - (w-1)) * SigmaX[(k-1) + F*idx] );

    f_new[idx + n*N*N].y =  w * f[idx + k*N*N].y +
              (1-w) * f[idx + (k-1)*N*N].y + sqr(h[k]) * 
             ((cub(w  ) -  w   ) * SigmaY[k     + F*idx] +
              (cub(w-1) - (w-1)) * SigmaY[(k-1) + F*idx] );
    }
}

__host__
void redistributeCubic(cusparseHandle_t handle, int N, 
                        int F, real *s, real2 *dev_a, real2 *dev_psi_1, real2 *dev_psi_2, 
                        int F_new, real2 *dev_a_new, real2 *dev_psi_1_new, real2 *dev_psi_2_new){

    // Allocate memory for the H matrix
    real *dev_h = nullptr;
    real *dev_dl = nullptr;
    real *dev_d  = nullptr;
    real *dev_du = nullptr;

    checkCudaErrors(cudaMalloc(&dev_h,  F*sizeof(real))); 
    checkCudaErrors(cudaMalloc(&dev_d,  F*sizeof(real))); 
    checkCudaErrors(cudaMalloc(&dev_dl, F*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_du, F*sizeof(real))); 

    buildHMatrix<<< 1, F >>>(F, s, dev_h, dev_dl, dev_d, dev_du);
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate space for Delta Matrix 
    real *DeltaX = nullptr;
    checkCudaErrors(cudaMalloc(&DeltaX, F*N*N*sizeof(real))); 
    real *DeltaY = nullptr;
    checkCudaErrors(cudaMalloc(&DeltaY, F*N*N*sizeof(real))); 

    // Define a function which redistribute the images for each field
    auto redistributeField = [&] (real2 *f, real2 *f_new) -> void {

        const int gridSize = (int)((N*N)/1024 + 1);

        // Compute Delta matrix
        computeDeltaX <<< gridSize, 1024 >>> (F, N, DeltaX, dev_h, f);
        computeDeltaY <<< gridSize, 1024 >>> (F, N, DeltaY, dev_h, f);
        checkCudaErrors(cudaDeviceSynchronize());

        // Estimate requested buffer size and allocate it
        size_t bufferSizeInBytesX;
        size_t bufferSizeInBytesY;
        
        checkCudaErrors(cusparseSgtsv2_bufferSizeExt(handle, F, N*N, dev_dl, dev_d, dev_du, DeltaX, F, &bufferSizeInBytesX));
        checkCudaErrors(cusparseSgtsv2_bufferSizeExt(handle, F, N*N, dev_dl, dev_d, dev_du, DeltaY, F, &bufferSizeInBytesY));

        real* pBufferX = nullptr;
        real* pBufferY = nullptr;
        
        checkCudaErrors(cudaMalloc(&pBufferX, bufferSizeInBytesX)); 
        checkCudaErrors(cudaMalloc(&pBufferY, bufferSizeInBytesY)); 

        // Solve the system
        checkCudaErrors(cusparseSgtsv2(handle, F, N*N, dev_dl, dev_d, dev_du, DeltaX, F, pBufferX));
        checkCudaErrors(cusparseSgtsv2(handle, F, N*N, dev_dl, dev_d, dev_du, DeltaY, F, pBufferY));
        checkCudaErrors(cudaDeviceSynchronize());
        // Now in Delta there is the Sigma matrix, the solution of the linear system

        // Generate new fields
        generateNewFramesCubic <<< gridSize, 1024 >>> (N, F, dev_h, DeltaX, DeltaY, f, F_new, f_new);
        checkCudaErrors(cudaDeviceSynchronize());

        // Release buffer
        checkCudaErrors(cudaFree(pBufferX));
        checkCudaErrors(cudaFree(pBufferY)); 
    };

    // Apply the above function
    redistributeField(dev_a, dev_a_new);
    redistributeField(dev_psi_1, dev_psi_1_new);
    #ifdef MULTICOMPONENT
    redistributeField(dev_psi_2, dev_psi_2_new);
    #endif

    // Cleaning
    checkCudaErrors(cudaFree(dev_h)); 
    checkCudaErrors(cudaFree(dev_d)); 
    checkCudaErrors(cudaFree(dev_dl));
    checkCudaErrors(cudaFree(dev_du));  
    checkCudaErrors(cudaFree(DeltaX));
    checkCudaErrors(cudaFree(DeltaY));  
}   

#endif