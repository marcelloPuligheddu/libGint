#include <cmath>
#include <omp.h>
#include <vector>
#include "define.h"
#include "compute_Fm.h"
#include "util.h"
#include "fgamma.h"
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

__device__ __host__ void compute_Fm_batched_single( int p, 
      const unsigned int* const __restrict__ Fm_input_list,
      const unsigned int* const __restrict__ Pm_input_list,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ ftable, const int ftable_ld ){

   unsigned int i    = Pm_input_list[p*PM_SIZE+PMA_OFFSET_IF  ];
   unsigned int ipzn = Pm_input_list[p*PM_SIZE+PMA_OFFSET_IPZN];
   unsigned int ipa,ipb,ipc,ipd;
   int n1,n2,n3;
   decode_ipabcd_n123( ipzn, &ipa, &ipb, &ipc, &ipd, &n1, &n2, &n3 );
   unsigned int Lmax   = Fm_input_list[i*FM_SIZE+FMA_OFFSET_LMAX];
   unsigned int idx_A  = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_A];
   unsigned int idx_B  = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_B];
   unsigned int idx_C  = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_C];
   unsigned int idx_D  = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_D];
   unsigned int idx_za = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_ZA] + ipa;
   unsigned int idx_zb = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_ZB] + ipb;
   unsigned int idx_zc = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_ZC] + ipc;
   unsigned int idx_zd = Fm_input_list[i*FM_SIZE+FMA_OFFSET_IDX_ZD] + ipd;
   unsigned int Of0    = Fm_input_list[i*FM_SIZE+FMA_OFFSET_OF0];
   unsigned int p0     = Fm_input_list[i*FM_SIZE+FMA_OFFSET_P0];

   const double* A = &data[idx_A];
   const double* B = &data[idx_B];
   const double* C = &data[idx_C];
   const double* D = &data[idx_D];
   double za = data[idx_za];
   double zb = data[idx_zb];
   double zc = data[idx_zc];
   double zd = data[idx_zd];

   int F_size = Lmax + 1;
   if (Lmax > 0 ) { F_size += 4*3+5; }

   int Of = Of0 + (p-p0) * F_size;

   double zab = za+zb;
   double inv_zab = 1. / zab;
   double P[3];
   compute_weighted_distance( P, A,B,za,zb,zab );
   double zcd = zc+zd;
   double inv_zcd = 1. / zcd;
   double Q[3];
   compute_weighted_distance( Q, C,D,zc,zd,zcd );
   
   if (periodic){
      Q[0] += n1 * cell[0*3+0] + n2 * cell[1*3+0] + n3 * cell[2*3+0];
      Q[1] += n1 * cell[0*3+1] + n2 * cell[1*3+1] + n3 * cell[2*3+1];
      Q[2] += n1 * cell[0*3+2] + n2 * cell[1*3+2] + n3 * cell[2*3+2];
   }
   double z = zab + zcd;
   double inv_z = 1./z;
   double W[3];
   compute_weighted_distance( W, P,Q,zab,zcd,z );
   
   double rho = zab*zcd*inv_z;
   double PQ[3] = { P[0]-Q[0], P[1]-Q[1], P[2]-Q[2] };
   double T = rho * (PQ[0]*PQ[0] + PQ[1]*PQ[1] + PQ[2]*PQ[2]);
   double Kab,Kcd;
   Kab = compute_K(za,zb,A,B);
   Kcd = compute_K(zc,zd,C,D);
   double Zn = 1./sqrt(z)/16./M_PI/M_PI;
   double Kfactor = Zn * Kab * Kcd;

//      compute_Fm(&Fm[Of],Lmax,T, ftable,ftable_ld );
   fgamma0( Lmax, T, &Fm[Of], ftable, ftable_ld );

   for( unsigned int m=0; m < Lmax+1; m++ ){
//         cout << " Fm[" << m << "](" << T << ") = " << Fm[Of+m] << " * " << Kfactor << endl;
      Fm[Of+m] *= Kfactor;
   }

   if ( Lmax > 0 ){
      double inv_2zab = inv_zab * 0.5;
      double inv_2zcd = inv_zcd * 0.5;
      double inv_2z = inv_z * 0.5;
      double rho_zab = rho*inv_zab;
      double rho_zcd = rho*inv_zcd;
      Fm[Of+Lmax+ 1] = P[0]-A[0];
      Fm[Of+Lmax+ 2] = P[1]-A[1];
      Fm[Of+Lmax+ 3] = P[2]-A[2];
      Fm[Of+Lmax+ 4] = W[0]-P[0];
      Fm[Of+Lmax+ 5] = W[1]-P[1];
      Fm[Of+Lmax+ 6] = W[2]-P[2];
      Fm[Of+Lmax+ 7] = Q[0]-C[0];
      Fm[Of+Lmax+ 8] = Q[1]-C[1];
      Fm[Of+Lmax+ 9] = Q[2]-C[2];
      Fm[Of+Lmax+10] = W[0]-Q[0];
      Fm[Of+Lmax+11] = W[1]-Q[1];
      Fm[Of+Lmax+12] = W[2]-Q[2];
      Fm[Of+Lmax+13] = inv_2zab;
      Fm[Of+Lmax+14] = - inv_2zab * rho_zab;
      Fm[Of+Lmax+15] = inv_2zcd;
      Fm[Of+Lmax+16] = - inv_2zcd * rho_zcd;
      Fm[Of+Lmax+17] = inv_2z;
   }
}

void compute_Fm_batched_low(
      const unsigned int* const __restrict__ Fm_input_list,
      const unsigned int* const __restrict__ Pm_input_list,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ ftable, const int ftable_ld ){
//#pragma omp target map(tofrom:Fm_input_list, Pm_input_list, data, cell, NFm, periodic, Fm)
//{
//#pragma omp parallel for
   for( int p = 0 ; p < NFm ; p++ ){
      compute_Fm_batched_single( p, Fm_input_list, Pm_input_list,data,Fm,NFm,periodic,cell,ftable,ftable_ld );
   }
}

__global__ void compute_Fm_batched_low_gpu(
      unsigned int* __restrict__ Fm_input_list,
      unsigned int* __restrict__ Pm_input_list,
      double* __restrict__ data,
      double* __restrict__ Fm,
      int NFm, bool periodic,
      double* __restrict__ cell,
      double* __restrict__ ftable, int ftable_ld ){

   for( int p = threadIdx.x + blockIdx.x*blockDim.x ; p < NFm ; p += blockDim.x*gridDim.x ){
      compute_Fm_batched_single( p, Fm_input_list, Pm_input_list,data,Fm,NFm,periodic,cell,ftable,ftable_ld );
   }
}

void compute_Fm_batched(
      const std::vector<unsigned int>& Fm_input_list, const std::vector<unsigned int>& Pm_input_list,
      const std::vector<double>& data, std::vector<double>& Fm, int NFm, bool periodic, double* cell,
      const double* const __restrict__ ftable, const int ftable_ld ){
   compute_Fm_batched_low(
      Fm_input_list.data(), Pm_input_list.data(),
      data.data(), Fm.data(), NFm, periodic, cell,
      ftable, ftable_ld );
} 

