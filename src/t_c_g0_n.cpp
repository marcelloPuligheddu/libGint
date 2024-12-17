#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>

#include "t_c_g0_n.h"

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

double* read_c0( int Nder, FILE* stream, int* ldc0_ptr ){
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

__host__ void fill_x12_to_patch(
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


__device__ __host__ bool t_c_g0_n_v2( 
      double* res, double R, double T, int Nder, const double* C0, int ld_C0, 
      int N1, int N2,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R, 
      const double* const __restrict__ BW_by_patch,
      int iw=0 ){
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
//   
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
      res[k] = tmp;
   }

   return false;
}

