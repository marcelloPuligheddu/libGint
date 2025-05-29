#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
/*
Copyright (c) 2023 Science and Technology Facilities Council

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef COMPUTE_FM_H_COMPILE_GUARD
#define COMPUTE_FM_H_COMPILE_GUARD

#include <vector>

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

__global__ void compute_Fm_batched_gpu_low_private( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ bias_and_weight_by_patch,   
      int potential_type, const int Ng );


__global__ void compute_Vm_batched_gpu_low_private( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ bias_and_weight_by_patch,   
      int potential_type, const int Ng );


__global__ void prepare_Fm_batched_gpu_low_shared(
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const int Ng );

__global__ void compute_Fm_batched_gpu_low_shared( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ bias_and_weight_by_patch,   
      int potential_type, const int Ng );



#endif // COMPUTE_FM_H_COMPILE_GUARD




