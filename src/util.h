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

#ifndef UTIL_H_COMPILE_GUARD
#define UTIL_H_COMPILE_GUARD



unsigned int encode4( int a, int b, int c, int d );

//#pragma omp declare target
__device__ __host__ void decode4(
      unsigned int abcd, unsigned int* a, unsigned int* b,
      unsigned int* c, unsigned int* d );

unsigned int encodeL( int la, int lb, int lc, int ld );
__device__ __host__ void decodeL( unsigned int L, int* la, int* lb, int* lc, int* ld );

unsigned int encode_ipabcd_n123( int ipa, int ipb, int ipc, int ipd, int n1, int n2, int n3 );

__device__ __host__ void decode_ipabcd_none(
      unsigned int ipzn,
      unsigned int* __restrict__ ipa, unsigned int* __restrict__ ipb,
      unsigned int* __restrict__ ipc, unsigned int* __restrict__ ipd);


__device__ __host__ void decode_ipabcd_n123(
      unsigned int ipzn,
      unsigned int* __restrict__ ipa, unsigned int* __restrict__ ipb,
      unsigned int* __restrict__ ipc, unsigned int* __restrict__ ipd,
      int* __restrict__ n1, int* __restrict__ n2, int* __restrict__ n3 );


__device__ __host__ int compute_Nc( int la, int lb=0, int lc=0, int ld=0 );


__device__ __host__ int compute_Ns( int la, int lb=0, int lc=0, int ld=0 );


__device__ __host__ void compute_weighted_distance(
      double X12[3], const double X1[3], const double X2[3],
      const double c1, const double c2, const double c12 );



__device__ __host__ int NLco_dev( int L );
// Returns L-lx
// essentialy is using the pattern:
// s = 0 0 0                                     -> L-lx = 0
// p = 1 0 0, 0 1 0, 0 0 1                       -> L-Lx = 0 1 1 
// d = 2 0 0, 1 1 0, 1 0 1, 0 2 0, 0 1 1, 0 0 2  -> L-lx = 0 1 1 2 2 2
// and noting that L-lx does not really depend on L, but on the index of the cartesian direction. Good up to L=8
// compute (cartesian) moment on x axis for a given total moment.
__device__ __host__ int L_lx(const int i );
__device__ __host__ int lx( const int i, const int L );
__device__ __host__ int lz( const int i, const int L );
__device__ __host__ int ly( const int i, const int L );
//#pragma omp end declare target



#endif
