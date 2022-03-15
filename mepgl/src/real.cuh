#ifndef REAL_CUH
#define REAL_CUH

// #define DOUBLE_PRECISION

// Type definition
#ifdef DOUBLE_PRECISION

typedef double real;

#ifdef __CUDACC__
typedef double2 real2;
typedef double4 real4;
#endif

#else

typedef float real;

#ifdef __CUDACC__
typedef float2 real2;
typedef float4 real4;
#endif

#endif

#endif // REAL_CUH
