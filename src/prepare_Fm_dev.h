#include "hip/hip_runtime.h"
#ifndef  PREPARE_FM_DEV_H_COMPILE_GUARD
#define PREPARE_FM_DEV_H_COMPILE_GUARD
__global__ void prepare_Fm_batched_gpu_low_private(
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const int Ng );

#endif // COMPILE_GUARD
