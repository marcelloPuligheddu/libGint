#include <cmath>
#include <omp.h>
#include <vector>
#include "define.h"
#include "util.h"
#include "fgamma.h"
#include "t_c_g0_n.h"
#include "compute_Fm.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define SQRT2 1.4142135623730950488016887242096980785696718753
#define BLKIDX 1

using std::cout;
using std::endl;

// Fills Fm with intermediate vectors
__global__ void prepare_Fm_batched_gpu_low_private(
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const int Ng ){

   for( int p = threadIdx.x + blockIdx.x*blockDim.x ; p < NFm ; p += blockDim.x*gridDim.x ){

   unsigned int i    =  OF[p];
   unsigned int ipzn = PMX[p*PMX_SIZE+PMX_OFFSET_IPZN];
   uint8_t ipabcd[4];

   decode4( ipzn, ipabcd );

   unsigned int idx_A  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_A];
   unsigned int idx_B  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_B];
   unsigned int idx_C  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_C];
   unsigned int idx_D  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_D];
   unsigned int idx_za = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZA] + ipabcd[0];
   unsigned int idx_zb = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZB] + ipabcd[1];
   unsigned int idx_zc = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZC] + ipabcd[2];
   unsigned int idx_zd = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZD] + ipabcd[3];
   unsigned int encoded_elabcd_12 = FVH[i*FVH_SIZE+FVH_OFFSET_ELABCD];

   uint8_t n12[2];
   decode_shell( encoded_elabcd_12, ipabcd, n12);
   uint8_t n1 = n12[0];
   uint8_t n2 = n12[1];

   // original position of the atoms before *any* pbc is applied
   const double* Ao = &data[idx_A];
   const double* Bo = &data[idx_B];
   const double* Co = &data[idx_C];
   const double* Do = &data[idx_D];

   const double za = data[idx_za];
   const double zb = data[idx_zb];
   const double zc = data[idx_zc];
   const double zd = data[idx_zd];


   const double zab = za+zb;
   const double inv_zab = 1. / zab;
   const double zcd = zc+zd;
   const double inv_zcd = 1. / zcd;
   const double z = zab + zcd;
   const double inv_z = 1./z;

   int F_size = Fsize(L);

   double A[3], B[3], C[3], D[3], Cs[3], Ds[3];
   double ABs[3], CDs[3], PQs[3];
   double P[3], Q[3], shift[3];

   compute_pbc( Ao, Bo, cell, ABs );
   A[0] = Ao[0];
   A[1] = Ao[1];
   A[2] = Ao[2];
   B[0] = Ao[0] + ABs[0] + neighs[n1*3+0];
   B[1] = Ao[1] + ABs[1] + neighs[n1*3+1];
   B[2] = Ao[2] + ABs[2] + neighs[n1*3+2];
   compute_weighted_distance( P, A,B,za,zb,inv_zab );

   compute_pbc( Co, Do, cell, CDs );  
   C[0] = Co[0];
   C[1] = Co[1];
   C[2] = Co[2];
   D[0] = Co[0] + CDs[0] + neighs[n2*3+0];
   D[1] = Co[1] + CDs[1] + neighs[n2*3+1];
   D[2] = Co[2] + CDs[2] + neighs[n2*3+2];
   compute_weighted_distance( Q, C,D,zc,zd,inv_zcd );

   compute_pbc( Q, P, cell, PQs );
   shift[0] = P[0] - Q[0] - PQs[0]; 
   shift[1] = P[1] - Q[1] - PQs[1]; 
   shift[2] = P[2] - Q[2] - PQs[2];

   double normf = 2. * pow(M_PI,5./2.) * inv_zab * inv_zcd ;

   double dx = A[0]-B[0];
   double dy = A[1]-B[1];
   double dz = A[2]-B[2];
   double RAB2 = dx*dx + dy*dy + dz*dz ;
   double expon_ab = - (za*zb)/(za+zb) * RAB2;

   dx = C[0]-D[0];
   dy = C[1]-D[1];
   dz = C[2]-D[2];
   double RCD2 = dx*dx + dy*dy + dz*dz ;
   double expon_cd = - (zc*zd)/(zc+zd) * RCD2;

//   double R1 = R_pgf[ipa*npb*2+ipb*2+0] * RAB2 + R_pgf[ipa*npb*2+ipb*2+1];
//   double R2 = R_pgf[ipc*npd*2+ipd*2+0] * RCD2 + R_pgf[ipc*npd*2+ipd*2+1];
//   double R12 = R1 + R2;

//   double Zn = 1./sqrt(z)/16./M_PI/M_PI; // libcint norm
   double Zn = 1./sqrt(z); // cp2k uses the correct norm so we can use OS86 eq 44

   double Kfactor = Zn * normf * exp(expon_ab+expon_cd);

   int Of = (p * Ng + 0 ) * F_size ;

   Cs[0] = Co[0]                           + shift[0];
   Cs[1] = Co[1]                           + shift[1];
   Cs[2] = Co[2]                           + shift[2];


   Ds[0] = Co[0] + CDs[0] + neighs[n2*3+0] + shift[0];
   Ds[1] = Co[1] + CDs[1] + neighs[n2*3+1] + shift[1];
   Ds[2] = Co[2] + CDs[2] + neighs[n2*3+2] + shift[2];

   compute_weighted_distance( Q, Cs,Ds,zc ,zd ,inv_zcd );

   Fm[Of+L+ 1] = P[0]-A[0];
   Fm[Of+L+ 2] = P[1]-A[1];
   Fm[Of+L+ 3] = P[2]-A[2];

   Fm[Of+L+ 4] = P[0];
   Fm[Of+L+ 5] = P[1];
   Fm[Of+L+ 6] = P[2];

   Fm[Of+L+ 7] = Q[0]-Cs[0];
   Fm[Of+L+ 8] = Q[1]-Cs[1];
   Fm[Of+L+ 9] = Q[2]-Cs[2];

   Fm[Of+L+10] = Q[0];
   Fm[Of+L+11] = Q[1];
   Fm[Of+L+12] = Q[2];

   Fm[Of+L+13] = Kfactor;
   Fm[Of+L+14] = inv_z;
//   Fm[Of+L+14] = R12;
   Fm[Of+L+15] = zc;
   Fm[Of+L+16] = zd;
   Fm[Of+L+17] = zab;

//   printf(" p %d | Ao: %lg %lg %lg    Bo: %lg %lg %lg || Co %lg %lg %lg     Do %lg %lg %lg    | n12: %d %d | \n",
//            p,     Ao[0],Ao[1],Ao[2], Bo[0],Bo[1],Bo[2], Co[0],Co[1],Co[2], Do[0],Do[1],Do[2], (int)n1, (int)n2);
//   printf(" p %d | B: %lg %lg %lg  C: %lg %lg %lg || S: %lg %lg %lg | Cs %lg %lg %lg | Kfactor %lg \n", 
//            p,     B[0],B[1],B[2], C[0],C[1],C[2],   shift[0],shift[1],shift[2], Fm[Of+L+1],Fm[Of+L+2],Fm[Of+L+3], Kfactor );
   }
}

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

