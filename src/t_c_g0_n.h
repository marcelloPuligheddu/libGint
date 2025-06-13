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


//double* read_c0( int Nder, FILE* stream, int* ldc0_ptr );

//__device__ __host__ void PD2VAL( double* res, int Nder, double TG1, double TG2, const double* C0_row );

//__device__ __host__ bool t_c_g0_n( double* res, double R, double T, int Nder, const double* C0, int ldc0 );

//__host__ void fill_x12_to_patch( const int N1, const int N2, int * x12_to_patch_low_R, int * x12_to_patch_high_R );

//__device__ __host__ bool t_c_g0_n_v2(
//      double* res, double R, double T, int Nder, const double* C0, int ld_C0, int N1, int N2, 
//      const   int*  const __restrict__ x12_to_patch_low_R, const   int*  const __restrict__ x12_to_patch_high_R,
//      const double* const __restrict__ BW_by_patch, int iw, const double Kfac);

//__device__ bool t_c_g0_n_v3(
//      double* res, double R, double T, int Nder, const double* C0, int ld_C0, int N1, int N2, 
//      const   int*  const __restrict__ x12_to_patch_low_R, const   int*  const __restrict__ x12_to_patch_high_R,
//      const double* const __restrict__ BW_by_patch, int iw);



#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>



//#include "BW_by_patch.h"

#include "locate_patch_low_R.cpp"
#include "locate_patch_high_R.cpp"

#define SQRT2 1.4142135623730950488016887242096980785696718753
#define DEGREE 13
#define target_error 0.100000E-08
#define NDERIV_MAX 21

/*
double* read_c0( int Nder, FILE* stream, int* ldc0_ptr );
__device__ __host__ void PD2VAL( double* res, int Nder, double TG1, double TG2, const double* C0_row );
__device__ __host__ bool t_c_g0_n( double* res, double R, double T, int Nder, const double* C0, int ldc0 );
__host__ void fill_x12_to_patch( const int N1, const int N2, int * x12_to_patch_low_R, int * x12_to_patch_high_R );
__device__ __host__ bool t_c_g0_n_v2(
      double* res, double R, double T, int Nder, const double* C0, int ld_C0, int N1, int N2, 
      const   int*  const __restrict__ x12_to_patch_low_R, const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ BW_by_patch );
*/

inline double * read_c0( int Nder, FILE* stream, int* ldc0_ptr ){
   const int patches = 207;
   assert( Nder <= NDERIV_MAX );
//   int nderiv_init = Nder;
   int size1 = ((Nder       + 1)*(DEGREE + 1)*(DEGREE + 2))/2;
   int size2 = ((NDERIV_MAX + 1)*(DEGREE + 1)*(DEGREE + 2))/2;
   int ldc0 = size1;
   double* C0_ptr = (double*)malloc( patches*size1*sizeof(double) );
   for( int i=0; i < patches; i++ ){
      for ( int j=0; j < size2 ; j++ ){
         double temp;
         int nscanned = fscanf( stream , "%lE", &temp);
         assert ( nscanned == 1 );
         if ( j < ldc0 ){
            C0_ptr[i*ldc0+j] = temp;
         }
//         printf("%d %d %d %d %lE \n", i, j, i*size2+j, i*ldc0+j, C0_ptr[i*ldc0+j] );
      }
   }
//   printf("Done reading\n");
   (*ldc0_ptr) = ldc0;
   return C0_ptr;
}

__host__ inline void fill_x12_to_patch(
      const int N1, const int N2, 
      int * x12_to_patch_low_R, int * x12_to_patch_high_R ){
   for ( int i1 = 0 ; i1 < N1 ; i1 ++ ){
      double X1 = ((double)i1)/double(N1-1.0);
      for ( int i2 = 0 ; i2 < N2 ; i2 ++ ){
         double X2 = ((double)i2)/double(N2-1.0);
         x12_to_patch_low_R[ i1*N2+i2] = locate_patch_low_R ( X1,X2 );
         x12_to_patch_high_R[i1*N2+i2] = locate_patch_high_R( X1,X2 );
      }
   }
}


__device__ __host__ inline bool t_c_g0_n_v2( 
      double* res, double R, double T, int Nder, const double* C0, int ld_C0, 
      int N1, int N2,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R, 
      const double* const __restrict__ BW_by_patch,
      int iw, const double Kfac ){

//   printf(" computing tcg R: %lg T: %lg ", R, T );
//   bool use_gamma = false;

   double upper = R*R + 11.0*R + 50.0;
   double lower = R*R - 11.0*R +  0.0;
   double X1, X2;
   int patch = 255;

   if (T > upper) { 
      for ( int n = 0; n <= Nder ; n++ ){
         res[n] = 0.0;
      }
      return false;
   }

   if (R <= 11.0) { 
      X2 = R/11.0;
      upper = R*R + 11.0*R + 50.0;
      lower = 0.0;
      X1 = (T - lower)/(upper - lower);

      int i1 = (X1 * N1);
      int i2 = (X2 * N2);
      
      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

//      assert( i1 >= 0 );
//      assert( i1 < N1 );
//      assert( i2 >= 0 );
//      assert( i2 < N2 );

      patch = x12_to_patch_low_R[ i1*N2+i2 ];
   } else {
      if ( T < lower ) {
         // why even zero? Res is going to get overwritten by gamma
         for ( int n = 0; n <= Nder ; n++ ){
            res[n] = 0.0;
         }
         return true;
      }
      X2 = 11.0/R;
      X1 = (T-lower)/(upper-lower);
      
      int i1 = (X1 * N1);
      int i2 = (X2 * N2);

      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

//      assert( i1 >= 0 );
//      assert( i1 < N1 );
//      assert( i2 >= 0 );
//      assert( i2 < N2 );

      patch = x12_to_patch_high_R[ i1*N2+i2 ];
   }

   const double B1 = BW_by_patch[ patch*4 + 0 ];
   const double W1 = BW_by_patch[ patch*4 + 1 ];
   const double B2 = BW_by_patch[ patch*4 + 2 ];
   const double W2 = BW_by_patch[ patch*4 + 3 ];

//   if ( iw > 0 ){
//      printf(" Computing tcg @ %lg %lg -> X12p = %lg %lg %d | BW = %lg %lg %lg %lg \n", R,T,X1,X2,patch,B1,W1,B2,W2 );
//   }
   
   const double * const C0_row = &C0[ld_C0*patch];

   double TG1 = (2.*X1-B1)*W1;
   double TG2 = (2.*X2-B2)*W2;

   double T1[14];
   double T2[14];
   T1[0] = 1.0;
   T2[0] = 1.0;
   T1[1] = SQRT2*TG1;
   T2[1] = SQRT2*TG2;
   T1[2] = 2.*TG1*T1[1] - SQRT2;
   T2[2] = 2.*TG2*T2[1] - SQRT2;
   for ( int i=3; i < 14; i++ ) {
      // NOTE: this is the recurrence relation for Chebishev polynomial of the first kind
      T1[i] = 2.*TG1*T1[i-1] - T1[i-2];
      T2[i] = 2.*TG2*T2[i-1] - T2[i-2];
   }

   // NOTE: this horror has the structure v1(k) @ L(k) @ v2(k).T[::-1]
   // where v1 and v2 are vector and L is a (flattened) Triangular matrix
   for ( int k=0; k <= Nder; k++ ){
      double tmp = 0.0;
      int jl = 0; // unlike l, jl does not get reset after the l loop
      for ( int j=0; j < 14; j++ ){
         double dot = 0.0;
         for ( int l=0; l < 14-j; l++){
            dot += T1[l] * C0_row[k*105+jl];
//            printf(" T1[l]: %lg C0_kl : %lg || k j l jl kjl %d %d %d %d %d || \n", T1[l], C0_row[k*105+jl], k, j, l, jl, k*105+jl );
            jl++;
         }
//         printf(" T2[j]: %lg dot : %lg || j k %d %d || \n", T2[j], dot, j, k );
         tmp += dot * T2[j];

      }
      res[k] = tmp*Kfac;
   }
   return false;
}


/*
__device__ bool t_c_g0_n_v3(
      double* res, double R, double T, int Nder, const double* C0, int ld_C0,
      int N1, int N2,
      const   int*  const __restrict__ x12_to_patch_low_R,
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ BW_by_patch,
      int iw=0 ){

   constexpr int NFT =   2;
   constexpr int SFT =  32;
   constexpr int NPT =   1;

   assert( NFT*SFT/NPT == blockDim.x );

   int my_fm_rank = threadIdx.x % SFT;
   int my_fm_team = threadIdx.x / SFT;
 
   double upper = R*R + 11.0*R + 50.0;
   double lower = R*R - 11.0*R +  0.0;
   double X1, X2;
   int patch = 255;

//   printf(" %d.%d v3 %lg %lg \n", blockIdx.x, threadIdx.x, T,R );

   if (T > upper) {
      for ( int n = 0; n <= Nder ; n++ ){
         res[n] = 0.0;
      }
      return false;
   }

   if (R <= 11.0) {
      X2 = R/11.0;
      upper = R*R + 11.0*R + 50.0;
      lower = 0.0;
      X1 = (T - lower)/(upper - lower);

      int i1 = (X1 * N1);
      int i2 = (X2 * N2);

      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

      patch = x12_to_patch_low_R[ i1*N2+i2 ];
   } else {
      if ( T < lower ) { // if R > 11 and T < R2 - 11 R use gamma
         // why even zero? Res is going to get overwritten by gamma
         for ( int n = 0; n <= Nder ; n++ ){
            res[n] = 0.0;
         }
         return true;
      }
      X2 = 11.0/R;
      X1 = (T-lower)/(upper-lower);

      int i1 = (X1 * N1);
      int i2 = (X2 * N2);

      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

      patch = x12_to_patch_high_R[ i1*N2+i2 ];
   }

   const double B1 = BW_by_patch[ patch*4 + 0 ];
   const double W1 = BW_by_patch[ patch*4 + 1 ];
   const double B2 = BW_by_patch[ patch*4 + 2 ];
   const double W2 = BW_by_patch[ patch*4 + 3 ];

   double TG1 = (2.*X1-B1)*W1;
   double TG2 = (2.*X2-B2)*W2;

//   printf(" Computing tcg @ %lg %lg -> X12p = %lg %lg %d | BW = %lg %lg %lg %lg \n", T,R, X1,X2,patch, B1,W1,B2,W2 );

   double T1[16];
   double T2[16];

   constexpr int s_ld = SFT+8;

   __shared__ double s_tmp[NFT*s_ld];
   __shared__ double s_dot_jt[NFT*s_ld]; // TODO reuse s_tmp (?)
   __shared__ double dot[NFT*SFT]; // TODO not shared (?)

//   unsigned int tid = threadIdx.x;

   T1[0] = 1.0;
   T2[0] = 1.0;
   T1[1] = SQRT2*TG1;
   T2[1] = SQRT2*TG2;
   T1[2] = 2.*TG1*T1[1] - SQRT2;
   T2[2] = 2.*TG2*T2[1] - SQRT2;
   for ( int i=3; i < 14; i++ ) {
      // NOTE: this is the recurrence relation for Chebishev polynomial of the first kind
      T1[i] = 2.*TG1*T1[i-1] - T1[i-2];
      T2[i] = 2.*TG2*T2[i-1] - T2[i-2];
   }
   T1[14] = 0.0;
   T2[14] = 0.0;
   T1[15] = 0.0;
   T2[15] = 0.0;


   // NOTE: this horror has the structure v1(k) @ L(k) @ v2(k).T[::-1]
   // where v1 and v2 are vector and L is a (flattened) Triangular matrix

   // Zero the extra 8 doubles at the end of the shared memory assigned to this team
   s_dot_jt[my_fm_team*s_ld+my_fm_rank+8] = 0.0;
   s_tmp[my_fm_team*s_ld+my_fm_rank+8] = 0.0;

   for ( int k=0; k <= Nder; k++ ){
      int jl = 0;
      for ( int j=0; j < 14; j++ ){
         // Step 1: load C and multiply by T1 into shared memory
         // TODO: reshape C0 into 16*16 square matrix
         // NOTE: each thread will only ever use T1[my_fm_rank]
         // TODO: compute multiple T1 and T2 for different n3 in the same team and share
         int l = my_fm_rank;
         if ( l < 14-j ) {
            s_dot_jt[my_fm_team*s_ld+l] = C0[ld_C0*patch + k*105 + jl + l] * T1[l];
//            printf(" %d.%d.%d | C0.T1 = %lg %lg %lg \n", blockIdx.x, threadIdx.x, j, C0[ld_C0*patch + k*105 + jl + l] * T1[l], C0[ld_C0*patch + k*105 + jl + l], T1[l] );
         } else {
            s_dot_jt[my_fm_team*s_ld+l] = 0.0;
         }
         __syncwarp();

         // Step 2: sum over l to compute dot[j]
         const unsigned int sid = my_fm_team*s_ld+l;
         s_dot_jt[sid] += s_dot_jt[sid + 8];__syncwarp();
         s_dot_jt[sid] += s_dot_jt[sid + 4];__syncwarp();
         s_dot_jt[sid] += s_dot_jt[sid + 2];__syncwarp();
         s_dot_jt[sid] += s_dot_jt[sid + 1];__syncwarp();

         // s_dot_jt[0] now contains the sum of C0 * T1 for this j and this idx_t
         dot[my_fm_team*16+j] = s_dot_jt[my_fm_team*s_ld+0];
//         printf(" %d.%d.%d | C0@T1 = %lg \n", blockIdx.x, threadIdx.x, j, dot[my_fm_team*16+j] );
         __syncwarp();
         jl += 14-j;
      }
      // Zero extra term in dot. Dot is the size 14 vector product of C(14x14,Triangular) and T1(14)
      // TODO: Recast as a (16x16,Square) @ 16 product
      dot[my_fm_team*16+14] = 0.0;
      dot[my_fm_team*16+15] = 0.0;
      // Now we have all the C0 @ T1 for all j for this given idx_t and k
      int j = my_fm_rank;
      const unsigned int sid = my_fm_team*s_ld+j;
      // Step 3: multiply by T2 and sum over j
      s_tmp[my_fm_team*s_ld+j] = dot[my_fm_team*16+j] * T2[j];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 8];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 4];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 2];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 1];__syncwarp();

      if( my_fm_rank == 0 ) {
         res[k] = s_tmp[my_fm_team*s_ld+0];
//         printf(" %d.%d | R %lg \n", blockIdx.x, threadIdx.x, res[k] );
      }
      __syncwarp();
   }
   return false;
}
*/

#endif
