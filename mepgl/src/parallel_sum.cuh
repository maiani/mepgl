#ifndef PARALLEL_SUM_CUH
#define PARALLEL_SUM_CUH

/************************************************************
 * Parallel sum functions
 ************************************************************/

#include "real.cuh"

namespace parallel_sum{

    #ifdef DOUBLE_PRECISION
    static const int blockSizeX = 512;
    #else
    static const int blockSizeX = 1024;
    #endif

    static const int gridSizeX = 24; // This number is hardware-dependent; usually #SM*2 is a good number.

    __global__ 
    void sumMultiBlockF(const real *array, const int arraySize, real *out);

    __host__ 
    void sumArrayF(real* dev_array, real* dev_results, int F, const int arraySize);

    __global__ 
    void sumMultiBlockF2(const real2 *array, const int arraySize, real2 *out);

    __host__ 
    void sumArrayF2(real2* dev_array, real2* dev_results, int F, const int arraySize);

    __global__ 
    void sumMultiBlockF4(const real4 *array, const int arraySize, real4 *out);

    __host__ 
    void sumArrayF4(real4* dev_array, real4* dev_results, int F, const int arraySize);

}

#endif