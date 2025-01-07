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

   if ( threadIdx.x == 0 ){
      cdpa[CDPA_COSx] = Fm[Of0+L+1+CDPA_COSx];
      cdpa[CDPA_COSy] = Fm[Of0+L+1+CDPA_COSy];
      cdpa[CDPA_COSz] = Fm[Of0+L+1+CDPA_COSz];
      cdpa[CDPA_DOSx] = Fm[Of0+L+1+CDPA_DOSx];
      cdpa[CDPA_DOSy] = Fm[Of0+L+1+CDPA_DOSy];
      cdpa[CDPA_DOSz] = Fm[Of0+L+1+CDPA_DOSz];
      cdpa[CDPA_Px  ] = Fm[Of0+L+1+CDPA_Px  ];
      cdpa[CDPA_Py  ] = Fm[Of0+L+1+CDPA_Py  ];
      cdpa[CDPA_Pz  ] = Fm[Of0+L+1+CDPA_Pz  ];
      cdpa[CDPA_PmAx] = Fm[Of0+L+1+CDPA_PmAx];
      cdpa[CDPA_PmAy] = Fm[Of0+L+1+CDPA_PmAy];
      cdpa[CDPA_PmAz] = Fm[Of0+L+1+CDPA_PmAz];

      cdpa[CDPA_Kfac] = Fm[Of0+L+1+CDPA_Kfac];
      cdpa[CDPA_invz] = Fm[Of0+L+1+CDPA_invz];
      cdpa[CDPA_zc  ] = Fm[Of0+L+1+CDPA_zc  ];
      cdpa[CDPA_zd  ] = Fm[Of0+L+1+CDPA_zd  ];
      cdpa[CDPA_zab ] = Fm[Of0+L+1+CDPA_zab ];

      cdpa[CDPA_zcd ] = cdpa[CDPA_zc] + cdpa[CDPA_zd];   
      cdpa[CDPA_rho ] = cdpa[CDPA_zab ] * cdpa[CDPA_zcd ] * cdpa[CDPA_invz];
      cdpa[CDPA_R   ] = R_cut * sqrt(cdpa[CDPA_rho ]);
      cdpa[CDPA_izab] =  1. / cdpa[CDPA_zab ];
      cdpa[CDPA_i2ab] = 0.5 * cdpa[CDPA_izab];
      cdpa[CDPA_mrab] = -1. * cdpa[CDPA_rho ] * cdpa[CDPA_izab] * cdpa[CDPA_i2ab];
      cdpa[CDPA_izcd] =  1. / cdpa[CDPA_zcd ];
      cdpa[CDPA_i2cd] = 0.5 * cdpa[CDPA_izcd];
      cdpa[CDPA_mrcd] = -1. * cdpa[CDPA_rho ] * cdpa[CDPA_izcd] * cdpa[CDPA_i2cd];
      cdpa[CDPA_i2z ] = 0.5 * cdpa[CDPA_invz];
   }
   __syncthreads();

   double R = cdpa[CDPA_R   ];
   double P[3], C[3], D[3], Q[3], W[3], PQ[3];
   double rpq2, T, Kfac;
   double zc,zd,zcd,izcd,zab,invz;

   zc   = cdpa[CDPA_zc  ];
   zd   = cdpa[CDPA_zd  ];

   izcd = cdpa[CDPA_izcd];
   zcd  = cdpa[CDPA_zcd ];
   zab  = cdpa[CDPA_zab ];
   invz = cdpa[CDPA_invz];

   P[0] = cdpa[CDPA_Px  ];
   P[1] = cdpa[CDPA_Py  ];
   P[2] = cdpa[CDPA_Pz  ];

   Kfac = cdpa[CDPA_Kfac];

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

      PQ[0] = P[0]-Q[0];
      PQ[1] = P[1]-Q[1];
      PQ[2] = P[2]-Q[2];
      Fm[Of+L+ 1] = cdpa[CDPA_PmAx];
      Fm[Of+L+ 2] = cdpa[CDPA_PmAy];
      Fm[Of+L+ 3] = cdpa[CDPA_PmAz];
      Fm[Of+L+ 4] = W[0]-cdpa[CDPA_Px  ];
      Fm[Of+L+ 5] = W[1]-cdpa[CDPA_Py  ];
      Fm[Of+L+ 6] = W[2]-cdpa[CDPA_Pz  ];
      Fm[Of+L+ 7] = Q[0]-C[0];
      Fm[Of+L+ 8] = Q[1]-C[1];
      Fm[Of+L+ 9] = Q[2]-C[2];
      Fm[Of+L+10] = W[0]-Q[0];
      Fm[Of+L+11] = W[1]-Q[1];
      Fm[Of+L+12] = W[2]-Q[2];
      Fm[Of+L+13] = cdpa[CDPA_i2ab];
      Fm[Of+L+14] = cdpa[CDPA_mrab];
      Fm[Of+L+15] = cdpa[CDPA_i2cd];
      Fm[Of+L+16] = cdpa[CDPA_mrcd];
      Fm[Of+L+17] = cdpa[CDPA_i2z ];
   }
   __syncthreads();
}



__global__ void prepare_Fm_batched_gpu_low__shared(
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

__device__ bool find_patch( 
      double T, double R, 
      int * patch, double * X1, double * X2, 
      const int * const __restrict__ x12_to_patch_low_R,
      const int * const __restrict__ x12_to_patch_high_R ){

   constexpr int N1 = POT_TRUNC_N1;
   constexpr int N2 = POT_TRUNC_N2;

   double upper = R*R + 11.0*R + 50.0;
   double lower = R*R - 11.0*R +  0.0;
   (*patch) = 255;

//   printf(" %d.%d v3 %lg %lg \n", blockIdx.x, threadIdx.x, T,R );

   if ( T < 0    ) { (*patch) = 253; return false; }
   if ( T > upper) { (*patch) = 254; return false; }

   if (R <= 11.0) {
      (*X2) = R/11.0;
      upper = R*R + 11.0*R + 50.0;
      lower = 0.0;

      (*X1) = (T - lower)/(upper - lower);
      int i1 = ((*X1) * N1);
      int i2 = ((*X2) * N2);
      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }
      (*patch) = x12_to_patch_low_R[ i1*N2+i2 ];
   } else {
      if ( T < lower ) { (*patch) = 252; return true; }
      (*X2) = 11.0/R;

      (*X1) = (T-lower)/(upper-lower);
      int i1 = ((*X1) * N1);
      int i2 = ((*X2) * N2);
      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }
      (*patch) = x12_to_patch_high_R[ i1*N2+i2 ];

   }
   return false;
}


__global__ void compute_Fm_batched_gpu_low_shared( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ BW_by_patch,   
      int potential_type, const int Ng ){

   int F_size = Fsize(L);

   constexpr int NFT =   2; // number of teams 
   constexpr int SFT =  32; // number of threads in a team
   constexpr int NPT =  32; // number of n3 assigned to each team

   assert( NFT*SFT == blockDim.x );
   assert( NPT <= SFT ); // guarantee single pass over T
   assert( SFT == 32 );  // no split warp / no multi warp

   int my_fm_rank = threadIdx.x % SFT;
   int my_fm_team = threadIdx.x / SFT;
  
   __shared__ double CDPA[CDPA_SIZE];

   int p = blockIdx.x;

   const int Of0 = (p * Ng + 0 ) * F_size ;

   // Note that we are using Fm[Of0] to comunicate,
   // so it is not allowed for two blocks to use the same Of0

   if ( threadIdx.x == 0 ){
      CDPA[CDPA_COSx] = Fm[Of0+L+1+CDPA_COSx];
      CDPA[CDPA_COSy] = Fm[Of0+L+1+CDPA_COSy];
      CDPA[CDPA_COSz] = Fm[Of0+L+1+CDPA_COSz];
      CDPA[CDPA_DOSx] = Fm[Of0+L+1+CDPA_DOSx];
      CDPA[CDPA_DOSy] = Fm[Of0+L+1+CDPA_DOSy];
      CDPA[CDPA_DOSz] = Fm[Of0+L+1+CDPA_DOSz];
      CDPA[CDPA_Px  ] = Fm[Of0+L+1+CDPA_Px  ];
      CDPA[CDPA_Py  ] = Fm[Of0+L+1+CDPA_Py  ];
      CDPA[CDPA_Pz  ] = Fm[Of0+L+1+CDPA_Pz  ];
      CDPA[CDPA_PmAx] = Fm[Of0+L+1+CDPA_PmAx];
      CDPA[CDPA_PmAy] = Fm[Of0+L+1+CDPA_PmAy];
      CDPA[CDPA_PmAz] = Fm[Of0+L+1+CDPA_PmAz];
      CDPA[CDPA_Kfac] = Fm[Of0+L+1+CDPA_Kfac];
      CDPA[CDPA_invz] = Fm[Of0+L+1+CDPA_invz];
      CDPA[CDPA_zc  ] = Fm[Of0+L+1+CDPA_zc  ];
      CDPA[CDPA_zd  ] = Fm[Of0+L+1+CDPA_zd  ];
      CDPA[CDPA_zab ] = Fm[Of0+L+1+CDPA_zab ]; 
      CDPA[CDPA_zcd ] = CDPA[CDPA_zc] + CDPA[CDPA_zd];   
      CDPA[CDPA_rho ] = CDPA[CDPA_zab ] * CDPA[CDPA_zcd ] * CDPA[CDPA_invz];
      CDPA[CDPA_R   ] = R_cut * sqrt(CDPA[CDPA_rho ]);
      CDPA[CDPA_izab] =  1. / CDPA[CDPA_zab ];
      CDPA[CDPA_i2ab] = 0.5 * CDPA[CDPA_izab];
      CDPA[CDPA_mrab] = -1. * CDPA[CDPA_rho ] * CDPA[CDPA_izab] * CDPA[CDPA_i2ab];
      CDPA[CDPA_izcd] =  1. / CDPA[CDPA_zcd ];
      CDPA[CDPA_i2cd] = 0.5 * CDPA[CDPA_izcd];
      CDPA[CDPA_mrcd] = -1. * CDPA[CDPA_rho ] * CDPA[CDPA_izcd] * CDPA[CDPA_i2cd];
      CDPA[CDPA_i2z ] = 0.5 * CDPA[CDPA_invz];
   }

   __syncthreads();


   double R = CDPA[CDPA_R   ];
//   __shared__ double s_T[NFT];
   double rpq2 =0.;
   double T = -1.0;

   double zc,zd,zcd,izcd,zab,invz;
   double P[3];

   zc   = CDPA[CDPA_zc  ];
   zd   = CDPA[CDPA_zd  ];

   izcd = CDPA[CDPA_izcd];
   zcd  = CDPA[CDPA_zcd ];
   zab  = CDPA[CDPA_zab ];
   invz = CDPA[CDPA_invz];

   P[0] = CDPA[CDPA_Px  ];
   P[1] = CDPA[CDPA_Py  ];
   P[2] = CDPA[CDPA_Pz  ];



   int my_n3_off = my_fm_rank % NPT;
   __shared__ double s_T1[ NFT*NPT*16 ];
   __shared__ double s_T2[ NFT*NPT*16 ];
   __shared__ int s_patch[NFT*NPT];
   // Make sure all threads in the same warp are looping together
   // even if some will skip later work
   for ( int n30 = my_fm_team*NPT; n30 < Ng ; n30 += NFT*NPT ){

      double C[3], D[3], Q[3], W[3], PQ[3];
      int n3 = n30 + my_n3_off;
      int Of = Of0 + n3 * F_size;

      double X1,X2;
      int patch = 255;
      bool use_gamma = false;

      // Each thread < NPT computes its own values of T and patch, then share T1, T2 and patch
      if ( my_fm_rank < NPT and n3 < Ng ){

         C[0] = CDPA[CDPA_COSx] + neighs[n3*3+0];
         C[1] = CDPA[CDPA_COSy] + neighs[n3*3+1];
         C[2] = CDPA[CDPA_COSz] + neighs[n3*3+2];
         D[0] = CDPA[CDPA_DOSx] + neighs[n3*3+0];
         D[1] = CDPA[CDPA_DOSy] + neighs[n3*3+1];
         D[2] = CDPA[CDPA_DOSz] + neighs[n3*3+2];
         compute_weighted_distance( Q, C,D,zc ,zd ,izcd );
         compute_weighted_distance( W, P,Q,zab,zcd,invz ); 
         PQ[0] = P[0]-Q[0];
         PQ[1] = P[1]-Q[1];
         PQ[2] = P[2]-Q[2];
         rpq2 = (PQ[0]*PQ[0] + PQ[1]*PQ[1] + PQ[2]*PQ[2]);
         T = CDPA[CDPA_rho ] * rpq2;
         use_gamma = find_patch( T,R, &patch, &X1,&X2, x12_to_patch_low_R,x12_to_patch_high_R );

         if ( patch <= 206 ){
            const double B1 = BW_by_patch[ patch*4 + 0 ];
            const double W1 = BW_by_patch[ patch*4 + 1 ];
            const double B2 = BW_by_patch[ patch*4 + 2 ];
            const double W2 = BW_by_patch[ patch*4 + 3 ];
            double TG1 = (2.*X1-B1)*W1;
            double TG2 = (2.*X2-B2)*W2;
            int loc = (my_fm_team*NPT+my_fm_rank)*16;
            s_T1[ loc+0 ] = 1.;
            s_T1[ loc+1 ] = SQRT2*TG1;
            s_T1[ loc+2 ] = 2.*SQRT2*TG1*TG1-SQRT2;
            for( int ii = 3; ii < 14; ii++ ){ s_T1[ loc+ii ] = 2.*TG1*s_T1[loc+ii-1]-s_T1[loc+ii-2]; }
            s_T1[ loc+14] = 0.0;
            s_T1[ loc+15] = 0.0;
            s_T2[ loc+0 ] = 1.;
            s_T2[ loc+1 ] = SQRT2*TG2;
            s_T2[ loc+2 ] = 2.*SQRT2*TG2*TG2-SQRT2;
            for( int ii = 3; ii < 14; ii++ ){ s_T2[ loc+ii ] = 2.*TG2*s_T2[loc+ii-1]-s_T2[loc+ii-2]; }
            s_T2[ loc+14] = 0.0;
            s_T2[ loc+15] = 0.0;
         }
      } else {
         T = -1.0; // T can only be positive, so we use -1 to flag n3 > Ng
      }

//      if ( blockIdx.x == BLKIDX ){
//         printf("%d.%d | TR: %lg + %lg -> %d \n", blockIdx.x, threadIdx.x, T, R, patch );
//      }   

      if (my_fm_rank < NPT){
         s_patch[my_fm_team*NPT+my_fm_rank] = patch;
      } __syncwarp();

      // Now that patch and T1,T2 are in shared memory we can use the full warp
      // for the calculation of Fm. This is why we force all threads to loop together
//      if ( blockIdx.x == BLKIDX and threadIdx.x == 0 ){
//         for( int ii=0; ii < NPT*NFT; ii++ ){
//            printf("%d.%d | %d s_p: %d \n", blockIdx.x, threadIdx.x, ii, s_patch[ii] );
//         }
//         for( int ii=0; ii < NPT*NFT; ii++ ){ 
//            for( int jj = 0; jj < 16; jj++ ){
//               printf("%d.%d | %d %d s_T: %lg \n", blockIdx.x, threadIdx.x, ii, jj, s_T1[ ii*16+jj ] );
//            }
//         }
//      } __syncwarp();

      constexpr int s_ld = SFT+8;
      __shared__ double s_tmp[NFT*s_ld];
      __shared__ double s_dot_jt[NFT*s_ld]; // TODO reuse s_tmp (?)


      // NOTE: this horror has the structure v1(k) @ L(k) @ v2(k).T[::-1]
      // where v1 and v2 are vector and L is a (flattened) Triangular matrix

      // Zero the extra 8 doubles at the end of the shared memory assigned to this team
      s_dot_jt[my_fm_team*s_ld+my_fm_rank+8] = 0.0;
      s_tmp[my_fm_team*s_ld+my_fm_rank+8] = 0.0;

      int my_l = my_fm_rank;
      int my_j = my_fm_rank;

      for( int idx_T = 0; idx_T < NPT ; idx_T++ ){

         int tmp_patch = s_patch[my_fm_team*NPT+idx_T];
         if ( tmp_patch > 206 ){ continue; } // Early exit 

//         const double * const C0_row = &C0[ld_C0*patch];

         double T1_l = 0.0;
         double T2_j = 0.0;
         if ( my_l < 16) { T1_l = s_T1[ (my_fm_team*NPT+idx_T)*16+my_l]; }
         if ( my_j < 16) { T2_j = s_T2[ (my_fm_team*NPT+idx_T)*16+my_j]; }

         for ( int k=0; k <= L; k++ ){
            int jl = 0;
            double dot = 0.0;
            for ( int j=0; j < 14; j++ ){
               // Step 1: load C and multiply by T1 into shared memory
               // TODO: reshape C0 into 16*16 square matrix
               // NOTE: each thread will only ever use T1[my_fm_rank]
               // TODO: compute multiple T1 and T2 for different n3 in the same team and share
               if ( my_l < 14-j ) {
                  s_dot_jt[my_fm_team*s_ld+my_l] = C0[ld_C0*tmp_patch + k*105 + jl + my_l] * T1_l;
//                  if ( blockIdx.x == BLKIDX ){ 
//                     printf(" %d.%d %d | C0 . T1 : %lg * %lg \n", blockIdx.x, threadIdx.x, tmp_patch, C0[ld_C0*tmp_patch + k*105 + jl + my_l], T1_l);
//                  }
               } else {
                  s_dot_jt[my_fm_team*s_ld+my_l] = 0.0;
               }
               __syncwarp();

               // Step 2: sum over l to compute dot[j]
               const unsigned int sid = my_fm_team*s_ld+my_l;
               s_dot_jt[sid] += s_dot_jt[sid + 8];__syncwarp();
               s_dot_jt[sid] += s_dot_jt[sid + 4];__syncwarp();
               s_dot_jt[sid] += s_dot_jt[sid + 2];__syncwarp();
               s_dot_jt[sid] += s_dot_jt[sid + 1];__syncwarp();

               // s_dot_jt[0] now contains the sum of C0 * T1 for this j and this idx_t
               if( j == my_j) { dot = s_dot_jt[my_fm_team*s_ld+0]; }

//               if ( blockIdx.x == BLKIDX ){ 
//                  printf(" %d.%d.%d | C0@T1 = %lg \n", blockIdx.x, threadIdx.x, j, dot );
//               }
               __syncwarp();
               jl += 14-j;
            }
            // Zero extra term in dot. Dot is the size 14 vector product of C(14x14,Triangular) and T1(14)
            // TODO: Recast as a (16x16,Square) @ 16 product


            // Now we have all the C0 @ T1 for all j for this given idx_t and k

            const unsigned int sid = my_fm_team*s_ld+my_j;
            // Step 3: multiply by T2 and sum over j
            s_tmp[my_fm_team*s_ld+my_j] = dot * T2_j;__syncwarp();
            s_tmp[sid] += s_tmp[sid + 8];__syncwarp();
            s_tmp[sid] += s_tmp[sid + 4];__syncwarp();
            s_tmp[sid] += s_tmp[sid + 2];__syncwarp();
            s_tmp[sid] += s_tmp[sid + 1];__syncwarp();

            if( my_j == 0 ) {
               int tmp_n3 = n30+idx_T;
               int tmp_Of = Of0 + tmp_n3 * F_size;
               Fm[tmp_Of+k] = s_tmp[my_fm_team*s_ld+0];
//               if ( Fm[Of+k]*Fm[Of+k] > 1.e-14 ){ 
//               if ( blockIdx.x == BLKIDX ){ 
//                  printf(" -> %d.%d %d n3: %d T: %lg R: %lg | Fm[%d+%d] %lg \n", blockIdx.x, threadIdx.x, idx_T, n3, T, R, Of, k, Fm[Of+k] );
//               }
            }
            __syncwarp();
         }
      }

      // Back to sequential work
      
      if ( patch > 206 and my_fm_rank < NPT and n3 < Ng ){ for( int k=0; k<=L; k++ ){ Fm[Of+k] = 0.0; } }
      if (    use_gamma and my_fm_rank < NPT and n3 < Ng ){ fgamma0( L, T, &Fm[Of], ftable, ftable_ld, 1.0 ); }

      if ( (use_gamma or patch <= 206) and my_fm_rank < NPT and n3 < Ng ){    
//         Fm[Of+0   ] = cdpa[CDPA_Kfac];
         Fm[Of+L+ 1] = CDPA[CDPA_PmAx];
         Fm[Of+L+ 2] = CDPA[CDPA_PmAy];
         Fm[Of+L+ 3] = CDPA[CDPA_PmAz];
         Fm[Of+L+ 4] = W[0]-CDPA[CDPA_Px  ];
         Fm[Of+L+ 5] = W[1]-CDPA[CDPA_Py  ];
         Fm[Of+L+ 6] = W[2]-CDPA[CDPA_Pz  ];
         Fm[Of+L+ 7] = Q[0]-C[0];
         Fm[Of+L+ 8] = Q[1]-C[1];
         Fm[Of+L+ 9] = Q[2]-C[2];
         Fm[Of+L+10] = W[0]-Q[0];
         Fm[Of+L+11] = W[1]-Q[1];
         Fm[Of+L+12] = W[2]-Q[2];
         Fm[Of+L+13] = CDPA[CDPA_i2ab];
         Fm[Of+L+14] = CDPA[CDPA_mrab];
         Fm[Of+L+15] = CDPA[CDPA_i2cd];
         Fm[Of+L+16] = CDPA[CDPA_mrcd];
         Fm[Of+L+17] = CDPA[CDPA_i2z ];
         // Don't forget to scale by Zn, Ka and Kb
         for( unsigned int m=0; m < L+1; m++ ){ Fm[Of+m] *= CDPA[CDPA_Kfac]; }
//         if ( Fm[Of+0]*Fm[Of+0] > 1.e-14 ){ 
//            printf(" -> %d.%d n3: %d T: %lg R: %lg | Fm[%d+%d] %lg \n", blockIdx.x, threadIdx.x, n3, T, R, Of, 0, Fm[Of+0] );
//         }
      }
      __syncwarp();
   }
   __syncthreads();
}

__global__ void compute_Fm_batched_gpu_low_single( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ BW_by_patch,   
      int potential_type, const int Ng ){

   int p = blockIdx.x;
   double C[3], D[3], P[3], PA[3], Q[3], W[3], PQ[3];
   double Kfac, invz,zc,zd,zab,zcd,rho,R,izab,i2ab,mrab,izcd,i2cd,mrcd,i2z;
   const int F_size = Fsize(L);
   double T1[14];
   double T2[14];
   const int Of0 = (p * Ng + 0 ) * F_size ;
   if ( threadIdx.x > 0 ) { return; }
   for( int n3 = 0; n3 < Ng; n3 ++ ){
      const int Of = Of0 + n3 * F_size ;
      C[0] = Fm[Of0+L+1+CDPA_COSx] + neighs[n3*3+0];
      C[1] = Fm[Of0+L+1+CDPA_COSy] + neighs[n3*3+1];
      C[2] = Fm[Of0+L+1+CDPA_COSz] + neighs[n3*3+2];
      D[0] = Fm[Of0+L+1+CDPA_DOSx] + neighs[n3*3+0];
      D[1] = Fm[Of0+L+1+CDPA_DOSy] + neighs[n3*3+1];
      D[2] = Fm[Of0+L+1+CDPA_DOSz] + neighs[n3*3+2];
      P[0] = Fm[Of0+L+1+CDPA_Px];
      P[1] = Fm[Of0+L+1+CDPA_Py];
      P[2] = Fm[Of0+L+1+CDPA_Pz];
      PA[0] = Fm[Of0+L+1+CDPA_PmAx];
      PA[1] = Fm[Of0+L+1+CDPA_PmAy];
      PA[2] = Fm[Of0+L+1+CDPA_PmAz];
      Kfac = Fm[Of0+L+1+CDPA_Kfac];
      invz = Fm[Of0+L+1+CDPA_invz];
      zc = Fm[Of0+L+1+CDPA_zc  ];
      zd = Fm[Of0+L+1+CDPA_zd  ];
      zab = Fm[Of0+L+1+CDPA_zab ]; 
      zcd = zc + zd;   
      rho = zab * zcd * invz;
      R = R_cut * sqrt(rho);
      izab =  1. / zab ;
      i2ab = 0.5 * izab;
      mrab = -1. * rho * izab * i2ab;
      izcd =  1. / zcd ;
      i2cd = 0.5 * izcd;
      mrcd = -1. * rho  * izcd * i2cd;
      i2z  = 0.5 * invz;

      double X1,X2;
      int patch = 255;
      bool use_gamma = false;
      double rpq2, T;

      compute_weighted_distance( Q, C,D,zc ,zd ,izcd );
      compute_weighted_distance( W, P,Q,zab,zcd,invz ); 
      PQ[0] = P[0]-Q[0];
      PQ[1] = P[1]-Q[1];
      PQ[2] = P[2]-Q[2];
      rpq2 = (PQ[0]*PQ[0] + PQ[1]*PQ[1] + PQ[2]*PQ[2]);
      T = rho * rpq2;
      use_gamma = find_patch( T,R, &patch, &X1,&X2, x12_to_patch_low_R,x12_to_patch_high_R );

      if ( patch <= 206 ){
         const double B1 = BW_by_patch[ patch*4 + 0 ];
         const double W1 = BW_by_patch[ patch*4 + 1 ];
         const double B2 = BW_by_patch[ patch*4 + 2 ];
         const double W2 = BW_by_patch[ patch*4 + 3 ];
         double TG1 = (2.*X1-B1)*W1;
         double TG2 = (2.*X2-B2)*W2;
         T1[0] = 1.0;
         T2[0] = 1.0;
         T1[1] = SQRT2*TG1;
         T2[1] = SQRT2*TG2;
         T1[2] = 2.*TG1*T1[1] - SQRT2;
         T2[2] = 2.*TG2*T2[1] - SQRT2;
         for ( int i=3; i < 14; i++ ) {
            // NOTE: this is the recurrence relation for Chebishev polynomial of the first kind
            // T[1] and T[2] make exception
            T1[i] = 2.*TG1*T1[i-1] - T1[i-2];
            T2[i] = 2.*TG2*T2[i-1] - T2[i-2];
         }

         for ( int k=0; k <= L; k++ ){
            int jl = 0;
            double sum = 0.0;
            for ( int j=0; j < 14; j++ ){
               double dot = 0.0;
               for ( int l=0; l < 14; l++ ){
                  dot += C0[ld_C0*patch + k*105 + jl + l] * T1[l];
                  jl ++;
               }
               sum += T2[j] * dot;
            }
            Fm[Of+k] = sum * Kfac;
         }
      }
     
      if ( patch > 206 ){ for( int k=0; k<=L; k++ ){ Fm[Of+k] = 0.0; } }
      if (   use_gamma ){
         fgamma0( L, T, &Fm[Of], ftable, ftable_ld, 1.0 );
         for( int k=0; k<=L; k++ ){ Fm[Of+k] *= Kfac; }
      }

//      if ( (use_gamma or patch <= 206) ){
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
//      }
   }
}


