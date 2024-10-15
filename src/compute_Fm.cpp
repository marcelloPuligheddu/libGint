#include <cmath>
#include <omp.h>
#include <vector>
#include "define.h"
#include "util.h"
#include "fgamma.h"
#include "t_c_g0_n.cpp"
#include "compute_Fm.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using std::cout;
using std::endl;

//#pragma omp declare target
__device__ __host__ double compute_K( const double z1, const double z2, const double R1[3], const double R2[3] ){
   double normf = sqrt(2.) * pow(M_PI,5./4.) / (z1+z2);
   double dx = R1[0]-R2[0];
   double dy = R1[1]-R2[1];
   double dz = R1[2]-R2[2];
   double R12_squared = dx*dx + dy*dy + dz*dz ;
   double expon = - (z1*z2)/(z1+z2) * R12_squared;
   return normf * exp(expon);
}
//#pragma omp end declare target

__device__ void compute_Fm_batched_single( int p, 
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const double* const __restrict__ ftable, 
      const int ftable_ld,
      const double R_cut, 
      const double * const __restrict__ C0,
      const int ld_C0,
      int potential_type, const int Ng ){

   unsigned int i    =  OF[p];
   unsigned int ipzn = PMX[p];
   unsigned int ipa,ipb,ipc,ipd;

   decode4( ipzn, &ipa, &ipb, &ipc, &ipd );

   unsigned int idx_A  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_A];
   unsigned int idx_B  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_B];
   unsigned int idx_C  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_C];
   unsigned int idx_D  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_D];
   unsigned int idx_za = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZA] + ipa;
   unsigned int idx_zb = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZB] + ipb;
   unsigned int idx_zc = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZC] + ipc;
   unsigned int idx_zd = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZD] + ipd;
   unsigned int encoded_nlabcd_12 = FVH[i*FVH_SIZE+FVH_OFFSET_NLABCD];

   // TODO ipa can be reused as the dummy for nla, and so on for bcd
   unsigned int nla,nlb,nlc,nld;
   unsigned int n1, n2;
   decode_shell( encoded_nlabcd_12, &nla,&nlb,&nlc,&nld,&n1,&n2);

//   if (p == 0 ){
//      printf(" Fm  FVH: " );
//      for ( int ii = 0 ; ii < FVH_SIZE ; ii++ ){
//         printf( " %u " , FVH[i*FVH_SIZE+ii] );
//      } printf("\n"); ; 
//   }

//   printf( " GPU px %d (%d.%d) : %u %u %u %u | %u %u %u \n" , p, threadIdx.x, blockIdx.x, ipa, ipb, ipc, ipd, n1, n2, n3 );

   // original position of the atoms before *any* pbc is applied
   const double* Ao = &data[idx_A];
   const double* Bo = &data[idx_B];
   const double* Co = &data[idx_C];
   const double* Do = &data[idx_D];

   double za = data[idx_za];
   double zb = data[idx_zb];
   double zc = data[idx_zc];
   double zd = data[idx_zd];


   double zab = za+zb;
   double inv_zab = 1. / zab;
   double zcd = zc+zd;
   double inv_zcd = 1. / zcd;
   double z = zab + zcd;
   double inv_z = 1./z;

//   printf( " p: %d | A: [ %d ]  %lf %lf %lf \n", p, idx_A, A[0], A[1], A[2] );
//   printf( " p: %d | B: [ %d ]  %lf %lf %lf \n", p, idx_B, B[0], B[1], B[2] );
//   printf( " p: %d | C: [ %d ]  %lf %lf %lf \n", p, idx_C, C[0], C[1], C[2] );
//   printf( " p: %d | D: [ %d ]  %lf %lf %lf \n", p, idx_D, D[0], D[1], D[2] );
//   printf( " z: [ %d ]  %lf %lf %lf %lf \n", idx_za, za, zb, zc, zd );
//   printf(" nla %d nlb %d nlc %d nld %d n1 %d n2 %d nlabcd %d \n" , nla,nlb,nlc,nld,n1,n2,nlabcd);     
//   printf(" shifting A %lf %lf %lf and B %lf %lf %lf by %lf %lf %lf \n", 
//       Ao[0], Ao[1], Ao[2], Bo[0], Bo[1], Bo[2], ABs[0], ABs[1], ABs[2] );

   int F_size = L + 1;
   if (L > 0 ) { F_size += 4*3+5; }

   // n1,n2 and n3 are the idx of the pbc cells for AB,CD and PQ \"
   // note that :
   // A does not move
   // B starts from the PERIODIC REDUCTION OF the AB pair and is it then moved by n1
   // C is moved by pq_shift and n3
   // D starts from the min.image of the CD pair and it is then moved by n2, PQ_shift and n3
   double A[3], B[3], C[3], D[3];
   double ABs[3], CDs[3], PQs[3];
   double P[3], Q[3], W[3], shift[3];

   compute_pbc( Ao, Bo, cell, ABs );
   A[0] = Ao[0];
   A[1] = Ao[1];
   A[2] = Ao[2];
   B[0] = Ao[0] + ABs[0] + neighs[n1*3+0];
   B[1] = Ao[1] + ABs[1] + neighs[n1*3+1];
   B[2] = Ao[2] + ABs[2] + neighs[n1*3+2];
   compute_weighted_distance( P, A,B,za,zb,zab );

   compute_pbc( Co, Do, cell, CDs );  
   C[0] = Co[0];
   C[1] = Co[1];
   C[2] = Co[2];
   D[0] = Co[0] + CDs[0] + neighs[n2*3+0];
   D[1] = Co[1] + CDs[1] + neighs[n2*3+1];
   D[2] = Co[2] + CDs[2] + neighs[n2*3+2];
   compute_weighted_distance( Q, C,D,zc,zd,zcd );

   compute_pbc( Q, P, cell, PQs );
   shift[0] = P[0] - Q[0] - PQs[0]; 
   shift[1] = P[1] - Q[1] - PQs[1]; 
   shift[2] = P[2] - Q[2] - PQs[2];
   double rho = zab*zcd*inv_z;
   double Kab;
   Kab = compute_K(za,zb,A,B);
   double Kcd; // Note Kcd does not depends on n3 !
   Kcd = compute_K(zc,zd,C,D);

//   double Zn = 1./sqrt(z)/16./M_PI/M_PI; // libcint norm
   double Zn = 1./sqrt(z); // cp2k uses the correct norm so we can use OS86 eq 44
   double Kfactor = Zn * Kab * Kcd;

   // END OF SHARED INFO BEFORE N3
   //
   for ( int n3 = 0; n3 < Ng ; n3++ ){

      int Of = (p * Ng + n3 ) * F_size ;

      C[0] = Co[0]                           + shift[0] + neighs[n3*3+0];
      C[1] = Co[1]                           + shift[1] + neighs[n3*3+1];
      C[2] = Co[2]                           + shift[2] + neighs[n3*3+2];
      D[0] = Co[0] + CDs[0] + neighs[n2*3+0] + shift[0] + neighs[n3*3+0];
      D[1] = Co[1] + CDs[1] + neighs[n2*3+1] + shift[1] + neighs[n3*3+1];
      D[2] = Co[2] + CDs[2] + neighs[n2*3+2] + shift[2] + neighs[n3*3+2];

   //   Q[0] = Q [0] + PQs[0] + neighs[n3*3+0];
   //   Q[1] = Q [1] + PQs[1] + neighs[n3*3+1];
   //   Q[2] = Q [2] + PQs[2] + neighs[n3*3+2];

      compute_weighted_distance( Q, C,D,zc,zd,zcd );
      compute_weighted_distance( W, P,Q,zab,zcd,z );
    
      double PQ[3] = { P[0]-Q[0], P[1]-Q[1], P[2]-Q[2] };
      double rpq2 = (PQ[0]*PQ[0] + PQ[1]*PQ[1] + PQ[2]*PQ[2]);
      double T = rho * rpq2 ;

   //   double AB[3] = { A[0]-B[0], A[1]-B[1], A[2]-B[2] };
   //   double rab2 = (AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2]);

   //   double CD[3] = { C[0]-D[0], C[1]-D[1], C[2]-D[2] };
   //   double rcd2 = (CD[0]*CD[0] + CD[1]*CD[1] + CD[2]*CD[2]);
      
   //   printf("PBC %lf %lf %lf %lf | %d %d %d -> %lf %lf %lf %lf -> | %lg %lg %lg | %lg %lg | %lg | \n", Ao[0],Bo[0],Co[0],Do[0], n1,n2,n3, A[0],B[0],C[0],D[0], ABs[0],CDs[0],shift[0], P[0],Q[0], rpq2 );
   //   printf("Y FM PBC A %lf B %lf C %lf D %lf n1 %d n2 %d n3 %d -> %lf %lf %lf %lf -> %lg %lg \n", Ao[1], Bo[1], Co[1], Do[1], n1,n2,n3, A[1], B[1], C[1], D[1], P[1], Q[1] );
   //   printf("Z FM PBC A %lf B %lf C %lf D %lf n1 %d n2 %d n3 %d -> %lf %lf %lf %lf -> %lg %lg \n", Ao[2], Bo[2], Co[2], Do[2], n1,n2,n3, A[2], B[2], C[2], D[2], P[2], Q[2] );

   //   double F0 = 0.0;
   //   fgamma0( 0, T, &F0, ftable, ftable_ld );
   //   printf("F @ %d %d %d (%d %d | %d %d)[ %d %d %d %d ] + | %d %d %d | = %lg * %lg * %lg * %lg \n", blockIdx.x, threadIdx.x, Of, idx_A, idx_B, idx_C, idx_D, ipa,ipb,ipc,ipd, n1, n2, n3, Zn, Kab, Kcd, F0 );

   //   printf(" shifting A %lf %lf %lf and B %lf %lf %lf by %lf %lf %lf \n", 
   //       Ao[0], Ao[1], Ao[2], Bo[0], Bo[1], Bo[2], ABs[0], ABs[1], ABs[2] );
   //   printf(" shifting A %lf %lf %lf and B %lf %lf %lf by %lf %lf %lf \n", 
   //       Ao[0], Ao[1], Ao[2], Bo[0], Bo[1], Bo[2], ABs[0], ABs[1], ABs[2] );
   //   printf(" shifting P %lf %lf %lf and Q %lf %lf %lf by %lf %lf %lf \n", 
   //       P [0], P [1], P [2], Q [0], Q [1], Q [2], PQs[0], PQs[1], PQs[2] );

   //   printf( " p: %d | P: [ %d ]  %lf %lf %lf \n", p, 0, P[0], P[1], P[2] );
   //   printf( " p: %d | Q: [ %d ]  %lf %lf %lf \n", p, 0, Q[0], Q[1], Q[2] );

      // TODO it may be good to split the calculation of T,R,PA,WP,QC,WQ,Kfac
      // and the calculation of Fm to separate kernels to limit reg pressure
      switch ( potential_type ){
         case COULOMB :
            fgamma0( L, T, &Fm[Of], ftable, ftable_ld );
         break;
         case TRUNCATED :
            double R = R_cut * sqrt(rho) ;
            bool use_gamma = t_c_g0_n( &Fm[Of], R, T, L, C0, ld_C0 );
            if (use_gamma) { fgamma0( L, T, &Fm[Of], ftable, ftable_ld ); }
         break;
      } // end switch potential_type

   //   for( unsigned int m=0; m < L+1; m++ ){
   //      double tmp = Fm[Of+m]*Kfactor;
   //      double F0 = 0.0;
   //      double R = R_cut * sqrt(rho) ;
   //      printf ( " Fm[%d @ %d](T=%lg,R=%lg) = %4.12lg = %4.12lg * %4.12lg || F00 = %4.12lg \n", p, m, T, R, tmp, Fm[Of+m], Kfactor, F0 );
   //   }

      // Don't forget to scale by Zn, Ka and Kb
      for( unsigned int m=0; m < L+1; m++ ){ Fm[Of+m] *= Kfactor; }

      if ( L > 0 ){
         double inv_2zab = inv_zab * 0.5;
         double inv_2zcd = inv_zcd * 0.5;
         double inv_2z = inv_z * 0.5;
         double rho_zab = rho*inv_zab;
         double rho_zcd = rho*inv_zcd;
         Fm[Of+L+ 1] = P[0]-A[0];
         Fm[Of+L+ 2] = P[1]-A[1];
         Fm[Of+L+ 3] = P[2]-A[2];
         Fm[Of+L+ 4] = W[0]-P[0];
         Fm[Of+L+ 5] = W[1]-P[1];
         Fm[Of+L+ 6] = W[2]-P[2];
         Fm[Of+L+ 7] = Q[0]-C[0];
         Fm[Of+L+ 8] = Q[1]-C[1];
         Fm[Of+L+ 9] = Q[2]-C[2];
         Fm[Of+L+10] = W[0]-Q[0];
         Fm[Of+L+11] = W[1]-Q[1];
         Fm[Of+L+12] = W[2]-Q[2];
         Fm[Of+L+13] = inv_2zab;
         Fm[Of+L+14] = - inv_2zab * rho_zab;
         Fm[Of+L+15] = inv_2zcd;
         Fm[Of+L+16] = - inv_2zcd * rho_zcd;
         Fm[Of+L+17] = inv_2z;
      }
   }
//   if (p == 0 ){
//      printf(" Fm2  FVH: " );
//      for ( int ii = 0 ; ii < FVH_SIZE ; ii++ ){
//         printf( " %u " , FVH[i*FVH_SIZE+ii] );
//      } printf("\n"); ; 
//   }


}

void compute_Fm_batched_low(
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const double* const __restrict__ ftable, const int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0, int potential_type ){
//#pragma omp target map(tofrom:FVH, PMI, data, cell, NFm, periodic, Fm)
//{
//#pragma omp parallel for
//   for( int p = 0 ; p < NFm ; p++ ){
//      compute_Fm_batched_single( p, FVH, OF,PMX,data,Fm,NFm,L,periodic,cell,neighs,ftable,ftable_ld,R_cut,C0,ld_C0,potential_type );
//   }
}

__global__ void compute_Fm_batched_low_gpu(
      unsigned int* __restrict__ FVH,
      unsigned int* __restrict__ OF,
      unsigned int* __restrict__ PMX,
      double* __restrict__ data,
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ cell,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0, int potential_type, const int Ng  ){
   for( int p = threadIdx.x + blockIdx.x*blockDim.x ; p < NFm ; p += blockDim.x*gridDim.x ){
      compute_Fm_batched_single( p, FVH, OF,PMX,data,Fm,NFm,L,periodic,cell,neighs,ftable,ftable_ld,R_cut,C0,ld_C0,potential_type,Ng );
   }
}

void compute_Fm_batched(
      const std::vector<unsigned int>& FVH,const std::vector<unsigned int>& OF, const std::vector<unsigned int>& PMX,
      const std::vector<double>& data, std::vector<double>& Fm, int NFm, int L, bool periodic, double* cell, double* neighs,
      const double* const __restrict__ ftable, const int ftable_ld,
       const double R_cut, const double * const __restrict__ C0, const int ld_C0, int potential_type  ){

   compute_Fm_batched_low(
      FVH.data(), OF.data(), PMX.data(),
      data.data(), Fm.data(), NFm, L, periodic, cell, neighs,
      ftable, ftable_ld,
      R_cut,C0,ld_C0, potential_type  );
} 

