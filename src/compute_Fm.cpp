#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include <cmath>
#include <omp.h>
#include <vector>
#include "define.h"
#include "util.h"
#include "fgamma.h"
#include "t_c_g0_n.h"
#include "compute_Fm.h"
#include <iostream>

#define SQRT2 1.4142135623730950488016887242096980785696718753
#define BLKIDX 1

using std::cout;
using std::endl;

// Computes T and the foundamental ssss(m) integrals
__global__ void compute_Fm_batched_gpu_low_private( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ bias_and_weight_by_patch,   
      int potential_type, const int Ng ){

   int F_size = Fsize(L);

   int p = blockIdx.x;

   const int Of0 = (p * Ng + 0 ) * F_size ;

   double P[3], Q[3], PQ[3];
   double rpq2, T;
   double Kfac,invz,zc,zd,zab,zcd,rho,R;

//   PA[0] = Fm[Of0+L+1];
//   PA[1] = Fm[Of0+L+2];
//   PA[2] = Fm[Of0+L+3];
   P[0] = Fm[Of0+L+4];
   P[1] = Fm[Of0+L+5];
   P[2] = Fm[Of0+L+6];
//   QC[0] = Fm[Of0+L+7];
//   QC[1] = Fm[Of0+L+8];
//   QC[2] = Fm[Of0+L+9];
   Q[0] =  Fm[Of0+L+10];
   Q[1] =  Fm[Of0+L+11];
   Q[2] =  Fm[Of0+L+12];

   Kfac = Fm[Of0+L+13];
   invz = Fm[Of0+L+14];
   zc   = Fm[Of0+L+15];
   zd   = Fm[Of0+L+16];
   zab  = Fm[Of0+L+17];

   zcd  = zc + zd;
   rho  = zab * zcd * invz;
   R    = R_cut * sqrt(rho);

   for ( int n3 = threadIdx.x; n3 < Ng ; n3 += blockDim.x ){

      int Of = Of0 + n3 * F_size ;
      PQ[0] = P[0]-Q[0]-neighs[n3*3+0];
      PQ[1] = P[1]-Q[1]-neighs[n3*3+1];
      PQ[2] = P[2]-Q[2]-neighs[n3*3+2];
      rpq2 = (PQ[0]*PQ[0] + PQ[1]*PQ[1] + PQ[2]*PQ[2]);
      T = rho * rpq2 ;
//      printf("%d.%d %d.%d TR: %lg %lg %lg \n", blockIdx.x, threadIdx.x, p, n3, T, R, Kfac );
//      for( int m=0; m<L;m++){ Fm[Of+m] = T * Kfac + R; }
//         switch ( potential_type ){
//            case COULOMB :
//               fgamma0( L, T, &Fm[Of], ftable, ftable_ld );
//            break;
//            case TRUNCATED :

               bool use_gamma = t_c_g0_n_v2(
                  &Fm[Of], R, T, L, C0, ld_C0,
                  POT_TRUNC_N1, POT_TRUNC_N2,
                  x12_to_patch_low_R, x12_to_patch_high_R, bias_and_weight_by_patch, 0, Kfac );
               if (use_gamma) { fgamma0( L, T, &Fm[Of], ftable, ftable_ld, Kfac ); }

//            break;
//         } // end switch potential_type
   }
   __syncthreads();
}

// Fills Fm with the vectors and values required by VVR
__global__ void compute_Vm_batched_gpu_low_private( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ bias_and_weight_by_patch,   
      int potential_type, const int Ng ){

   int F_size = Fsize(L);
   int p = blockIdx.x;

   const int Of0 = (p * Ng + 0 ) * F_size ;

   double P[3], QC[3], Q[3], W[3], PA[3], Qp[3];
   double zc,zd,zcd,zab,invz,rho,izab,i2ab,mrab,izcd,i2cd,mrcd,i2z;

   PA[0] = Fm[Of0+L+1];
   PA[1] = Fm[Of0+L+2];
   PA[2] = Fm[Of0+L+3];
   P[0] = Fm[Of0+L+4];
   P[1] = Fm[Of0+L+5];
   P[2] = Fm[Of0+L+6];
   QC[0] = Fm[Of0+L+7];
   QC[1] = Fm[Of0+L+8];
   QC[2] = Fm[Of0+L+9];
   Q[0] =  Fm[Of0+L+10];
   Q[1] =  Fm[Of0+L+11];
   Q[2] =  Fm[Of0+L+12];

//////////
   invz = Fm[Of0+L+14];
   zc   = Fm[Of0+L+15];
   zd   = Fm[Of0+L+16];
   zab  = Fm[Of0+L+17];

   __syncthreads(); // sync before Fm[Of0] is wrote over by tid 0

   zcd  = zc + zd;
   rho  = zab * zcd * invz;

   izab = 1. / zab;
   i2ab = 0.5 * izab;
   mrab =  -1. * rho * izab * i2ab;
   izcd = 1. / zcd;
   i2cd = 0.5 * izcd;
   mrcd =  -1. * rho * izcd * i2cd;
   i2z  = 0.5 * invz;

   for ( int n3 = threadIdx.x; n3 < Ng ; n3 += blockDim.x ){

      int Of = Of0 + n3 * F_size ;

      Qp[0] = Q[0] + neighs[n3*3+0];
      Qp[1] = Q[1] + neighs[n3*3+1];
      Qp[2] = Q[2] + neighs[n3*3+2];

      compute_weighted_distance( W, P, Qp, zab,zcd,invz );
      if ( Fm[Of+0] > 0.0 ){

         Fm[Of+L+ 1] = PA[0];
         Fm[Of+L+ 2] = PA[1];
         Fm[Of+L+ 3] = PA[2];
         Fm[Of+L+ 4] = W[0]-P[0];
         Fm[Of+L+ 5] = W[1]-P[1];
         Fm[Of+L+ 6] = W[2]-P[2];
         Fm[Of+L+ 7] = QC[0];
         Fm[Of+L+ 8] = QC[1];
         Fm[Of+L+ 9] = QC[2];
         Fm[Of+L+10] = W[0]-Qp[0];
         Fm[Of+L+11] = W[1]-Qp[1];
         Fm[Of+L+12] = W[2]-Qp[2];
         Fm[Of+L+13] = i2ab;
         Fm[Of+L+14] = mrab;
         Fm[Of+L+15] = i2cd;
         Fm[Of+L+16] = mrcd;
         Fm[Of+L+17] = i2z ;
      }
//      if ( L == 1 and blockIdx.x == 1024){
//         printf("%d.%d %d.%d TR: %lg %lg %lg \n", blockIdx.x, threadIdx.x, p, n3, i2ab, mrab, i2z );
//      }

   }
}

