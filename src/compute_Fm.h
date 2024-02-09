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

void compute_Fm_batched(
      const std::vector<unsigned int>& FVH, const std::vector<unsigned int>& PMI,
      const std::vector<double>& data, std::vector<double>& Fm, int NFm, int L, bool periodic, double* cell,
      const double* const __restrict__ ftable, const int ftable_ld );

void compute_Fm_batched_low(
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ PMI,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ ftable, const int ftable_ld );

__global__ void compute_Fm_batched_low_gpu(
      unsigned int* __restrict__ FVH,
      unsigned int*  __restrict__ PMI,
      double* __restrict__ data,
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ cell,
      double* __restrict__ ftable, int ftable_ld );


#endif // COMPUTE_FM_H_COMPILE_GUARD




