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
using std::cout;
using std::endl;

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
   unsigned int ipzn = PMX[p];
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
   unsigned int encoded_nlabcd_12 = FVH[i*FVH_SIZE+FVH_OFFSET_NLABCD];

   uint8_t n12[2];
   decode_shell( encoded_nlabcd_12, ipabcd, n12);
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

   double A[3], B[3], C[3], D[3];
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

   Fm[Of+L+ 1] = Co[0]                           + shift[0];
   Fm[Of+L+ 2] = Co[1]                           + shift[1];
   Fm[Of+L+ 3] = Co[2]                           + shift[2];

   Fm[Of+L+ 4] = Co[0] + CDs[0] + neighs[n2*3+0] + shift[0];
   Fm[Of+L+ 5] = Co[1] + CDs[1] + neighs[n2*3+1] + shift[1];
   Fm[Of+L+ 6] = Co[2] + CDs[2] + neighs[n2*3+2] + shift[2];

   Fm[Of+L+ 7] = P[0];
   Fm[Of+L+ 8] = P[1];
   Fm[Of+L+ 9] = P[2];

   Fm[Of+L+10] = P[0]-A[0];
   Fm[Of+L+11] = P[1]-A[1];
   Fm[Of+L+12] = P[2]-A[2];

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

   double P[3], C[3], D[3], Q[3], PQ[3];
   double rpq2, T;
   double Kfac,invz,zc,zd,zab,zcd,rho,R,izcd;

   C[0] = Fm[Of0+L+1+CDPA_COSx];
   C[1] = Fm[Of0+L+1+CDPA_COSy];
   C[2] = Fm[Of0+L+1+CDPA_COSz];
   D[0] = Fm[Of0+L+1+CDPA_DOSx];
   D[1] = Fm[Of0+L+1+CDPA_DOSy];
   D[2] = Fm[Of0+L+1+CDPA_DOSz];
   P[0] = Fm[Of0+L+1+CDPA_Px  ];
   P[1] = Fm[Of0+L+1+CDPA_Py  ];
   P[2] = Fm[Of0+L+1+CDPA_Pz  ];

   Kfac = Fm[Of0+L+1+CDPA_Kfac];
   invz = Fm[Of0+L+1+CDPA_invz];
   zc   = Fm[Of0+L+1+CDPA_zc  ];
   zd   = Fm[Of0+L+1+CDPA_zd  ];
   zab  = Fm[Of0+L+1+CDPA_zab ];

   zcd  = zc + zd;
   rho  = zab * zcd * invz;
   R    = R_cut * sqrt(rho);
   izcd = 1. / zcd;

   compute_weighted_distance( Q, C,D,zc ,zd ,izcd );

   for ( int n3 = threadIdx.x; n3 < Ng ; n3 += blockDim.x ){

      int Of = Of0 + n3 * F_size ;
      PQ[0] = P[0]-Q[0]-neighs[n3*3+0];
      PQ[1] = P[1]-Q[1]-neighs[n3*3+1];
      PQ[2] = P[2]-Q[2]-neighs[n3*3+2];
      rpq2 = (PQ[0]*PQ[0] + PQ[1]*PQ[1] + PQ[2]*PQ[2]);
      T = rho * rpq2 ;
      for( int m=0; m<L;m++){ Fm[Of+m] = T * Kfac + R; }
//         switch ( potential_type ){
//            case COULOMB :
//               fgamma0( L, T, &Fm[Of], ftable, ftable_ld );
//            break;
//            case TRUNCATED :

//               bool use_gamma = t_c_g0_n_v2(
//                  &Fm[Of], R, T, L, C0, ld_C0,
//                  POT_TRUNC_N1, POT_TRUNC_N2,
//                  x12_to_patch_low_R, x12_to_patch_high_R, bias_and_weight_by_patch, 0, Kfac );
//               if (use_gamma) { fgamma0( L, T, &Fm[Of], ftable, ftable_ld, Kfac ); }

//            break;
//         } // end switch potential_type
   }
   __syncthreads();
}


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
   __shared__ double cdpa[CDPA_SIZE];
   int p = blockIdx.x;

   const int Of0 = (p * Ng + 0 ) * F_size ;

   double P[3], C[3], D[3], Q[3], W[3], PA[3];
   double zc,zd,zcd,zab,invz,rho,izab,i2ab,mrab,izcd,i2cd,mrcd,i2z;

   C[0] = Fm[Of0+L+1+CDPA_COSx];
   C[1] = Fm[Of0+L+1+CDPA_COSy];
   C[2] = Fm[Of0+L+1+CDPA_COSz];
   D[0] = Fm[Of0+L+1+CDPA_DOSx];
   D[1] = Fm[Of0+L+1+CDPA_DOSy];
   D[2] = Fm[Of0+L+1+CDPA_DOSz];
   P[0] = Fm[Of0+L+1+CDPA_Px  ];
   P[1] = Fm[Of0+L+1+CDPA_Py  ];
   P[2] = Fm[Of0+L+1+CDPA_Pz  ];
   PA[0] = Fm[Of0+L+1+CDPA_PmAx];
   PA[1] = Fm[Of0+L+1+CDPA_PmAy];
   PA[2] = Fm[Of0+L+1+CDPA_PmAz];


   invz = Fm[Of0+L+1+CDPA_invz];
   zc   = Fm[Of0+L+1+CDPA_zc  ];
   zd   = Fm[Of0+L+1+CDPA_zd  ];
   zab  = Fm[Of0+L+1+CDPA_zab ];

   zcd  = zc + zd;
   rho  = zab * zcd * invz;

   izab = 1. / zab;
   i2ab = 0.5 * izab;
   mrab =  -1. * rho * izab * i2ab;
   izcd = 1. / zcd;
   i2cd = 0.5 * izcd;
   mrcd =  -1. * rho * izcd * i2cd;
   i2z  = 0.5 * invz;

   compute_weighted_distance( Q, C,D,zc ,zd ,izcd );

   for ( int n3 = threadIdx.x; n3 < Ng ; n3 += blockDim.x ){

      int Of = Of0 + n3 * F_size ;

      C[0] = cdpa[CDPA_COSx] + neighs[n3*3+0];
      C[1] = cdpa[CDPA_COSy] + neighs[n3*3+1];
      C[2] = cdpa[CDPA_COSz] + neighs[n3*3+2];

      D[0] = cdpa[CDPA_DOSx] + neighs[n3*3+0];
      D[1] = cdpa[CDPA_DOSy] + neighs[n3*3+1];
      D[2] = cdpa[CDPA_DOSz] + neighs[n3*3+2];

      compute_weighted_distance( Q, C,D,zc ,zd ,izcd );
      compute_weighted_distance( W, P,Q,zab,zcd,invz );

      Fm[Of+L+ 1] = PA[0];
      Fm[Of+L+ 2] = PA[1];
      Fm[Of+L+ 3] = PA[2];
      Fm[Of+L+ 4] = W[0]-P[0];
      Fm[Of+L+ 5] = W[1]-P[1];
      Fm[Of+L+ 6] = W[2]-P[2];
      Fm[Of+L+ 7] = Q[0]-C[0];
      Fm[Of+L+ 8] = Q[1]-C[1];
      Fm[Of+L+ 9] = Q[2]-C[2];
      Fm[Of+L+10] = W[0]-Q[0];
      Fm[Of+L+11] = W[1]-Q[1];
      Fm[Of+L+12] = W[2]-Q[2];
      Fm[Of+L+13] = i2ab;
      Fm[Of+L+14] = mrab;
      Fm[Of+L+15] = i2cd;
      Fm[Of+L+16] = mrcd;
      Fm[Of+L+17] = i2z ;
   }
}

