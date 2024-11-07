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

#ifndef T_C_G0_N_H_COMPILE_GUARD
#define T_C_G0_N_H_COMPILE_GUARD


double* read_c0( int Nder, FILE* stream, int* ldc0_ptr );

__device__ __host__ void PD2VAL( double* res, int Nder, double TG1, double TG2, const double* C0_row );

__device__ __host__ bool t_c_g0_n( double* res, double R, double T, int Nder, const double* C0, int ldc0 );

__host__ void fill_x12_to_patch( const int N1, const int N2, int * x12_to_patch_low_R, int * x12_to_patch_high_R );

__device__ __host__ bool t_c_g0_n_v2(
      double* res, double R, double T, int Nder, const double* C0, int ld_C0, int N1, int N2, 
      const   int*  const __restrict__ x12_to_patch_low_R, const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ BW_by_patch, int iw);

#endif
