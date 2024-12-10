#include <iostream>
#include <cassert>
#include "compute_VRR.h"
#include "util.h"
#include "define.h"

using std::cout;
using std::endl;

template < int M >
__device__ void execute_VRR1( // int AL, int CL, int m, 
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ amc0mp,
      const double PA[3], const double WP[3], const int my_vrr_rank=0, const int vrr_team_size=1 ){

   for( int imm=my_vrr_rank; imm < 3*M; imm+=vrr_team_size ){

      int mm = imm / 3 ;
      int i  = imm % 3 ;

      int d  = i ;
      int im = 0 ; 

      int idx_000 = i  + 3*mm;
      int idx_m00 = im + 1*mm;

      a0c0m0[idx_000] = PA[d] * amc0m0[idx_m00] + WP[d] * amc0mp[idx_m00];
   }
}


__device__ void execute_VRR1_gpu( // int AL, int CL, int m, 
      const int m, 
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ amc0mp,
      const double PA[3], const double WP[3], const int my_vrr_rank=0, const int vrr_team_size=1 ){

   switch ( m ){
      case  1: execute_VRR1< 1>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      case  2: execute_VRR1< 2>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      case  3: execute_VRR1< 3>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      case  4: execute_VRR1< 4>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      case  5: execute_VRR1< 5>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      case  6: execute_VRR1< 6>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      case  7: execute_VRR1< 7>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      case  8: execute_VRR1< 8>(a0c0m0, amc0m0,amc0mp,PA,WP,my_vrr_rank,vrr_team_size); break;
      default : assert(false);
   }
}




template < int AL >
__device__ void execute_VRR2(
      const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ amc0mp,
      const double* __restrict__ awc0m0,
      const double* __restrict__ awc0mp,
      const double PA[3], const double WP[3], const double inv_2zab, const double min_rho_zab2,
      const int my_vrr_rank=0, const int vrr_team_size=1 ){

   const int NcoA   = (AL+1)*(AL+2)/2;
   const int NcoAxx = (AL+0)*(AL-1)/2;
   const int NcoAx  = (AL+1)*(AL+0)/2;
   const int NcoAyy = (AL+3)*(AL+0)/2 - 1;
   const int NcoAy  = (AL+1)*(AL+2)/2 - 2 ;
   const int NcoAm  = (AL+1)*(AL+0)/2;
   const int NcoAw  = (AL-1)*(AL+0)/2;


   for ( int imm = my_vrr_rank; imm < m*NcoA; imm+=vrr_team_size ){

      int i  = imm % NcoA ;
      int mm = imm / NcoA ;

      int ex = lx_dev(i,AL);
      int ey = ly_dev(i,AL);
      int ez = AL - ex - ey ;
      int d, im, iw, e2 ;
      if ( i < NcoAxx ){
         d  = 0;
         im = i;
         iw = i;
         e2 = ex - 1;
      } else if ( i < NcoAx ) {
         d  = 0;
         im = i;
         iw = 0;
         e2 = 0;
      } else if ( i < NcoAyy ){
         d  = 1;
         im = i - AL;
         iw = i - 2 * AL + 1;
         e2 = ey - 1;
      } else if ( i == NcoAy ){
         d  = 1;
         im = i - AL;
         iw = 0;
         e2 = 0;
      } else {
         d  = 2;
         im = i - AL - 1;
         iw = i - 2 * AL - 1;
         e2 = ez - 1;
      }
      int idx_000 = imm ; // i  + NcoA *mm ;
      int idx_m00 = im + NcoAm*mm ;
      int idx_w00 = iw + NcoAw*mm ;

      a0c0m0[ idx_000 ] = PA[d] * amc0m0[idx_m00] + 
                          WP[d] * amc0mp[idx_m00] + 
                          e2*( inv_2zab * awc0m0[idx_w00] + min_rho_zab2 * awc0mp[idx_w00] );
   }
}


__device__ void execute_VRR2_gpu(
      const int AL, const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ amc0mp,
      const double* __restrict__ awc0m0,
      const double* __restrict__ awc0mp,
      const double PA[3], const double WP[3], const double inv_2zab, const double min_rho_zab2,
      const int my_vrr_rank=0, const int vrr_team_size=1 ){


   switch ( AL ){
      case  1: execute_VRR2< 1>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  2: execute_VRR2< 2>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  3: execute_VRR2< 3>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  4: execute_VRR2< 4>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  5: execute_VRR2< 5>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  6: execute_VRR2< 6>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  7: execute_VRR2< 7>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  8: execute_VRR2< 8>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case  9: execute_VRR2< 9>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case 10: execute_VRR2<10>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case 11: execute_VRR2<11>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      case 12: execute_VRR2<12>(m,a0c0m0, amc0m0,amc0mp,awc0m0,awc0mp,PA,WP,inv_2zab,min_rho_zab2,my_vrr_rank,vrr_team_size); break;
      default : assert(false);
   }
}

// more general case, for al>0 and cl>1
template< int AL, int CL >
__device__ void execute_VRR5(
      const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ a0cmmp,
      const double* __restrict__ a0cwm0,
      const double* __restrict__ a0cwmp,
      const double* __restrict__ amcmmp,
      const double QC[3], const double WQ[3],
      const double inv_2zcd, const double min_rho_zcd2, const double inv_2z,
      const int my_vrr_rank=0, const int vrr_team_size=1 ){
   //
   constexpr int NA = ((AL+1)*(AL+2))/2;
   constexpr int MA = ((AL+0)*(AL+1))/2;
   constexpr int NC = ((CL+1)*(CL+2))/2;
   constexpr int MC = ((CL+0)*(CL+1))/2;
   constexpr int WC = ((CL-1)*(CL+0))/2;
   constexpr int YC = (CL+3)*CL/2-1;

   // k is the faster variable, followed by i, then m
   for ( int mik=my_vrr_rank; mik < NA*NC; mik+=vrr_team_size ){
      int ex, ey, ez, fx, fy, fz;
      int i, k, d, km, kw, f2, im, e2 ;
      int idx_000, idx_0m0, idx_0w0, idx_mmp;
      double i_0m0, i_0mp, i_0w0, i_0wp, i_mmp;

      k  =  mik % NC;
      i  = (mik / NC)%NA;

      ex = lx_dev(i,AL);
      ez = lz_dev(i,AL);
      ey = AL - ex - ez;
      fx = lx_dev(k,CL);
      fz = lz_dev(k,CL);
      fy = CL - fx - fz;

      // if the x moment of k is more than zero, than apply the vrr along x
      if (k < WC){ 
         d  = 0;
         km = k;
         kw = k;
         f2 = fx-1;
         im = i;
         e2 = ex;
      // x[....]
      } else if (k < MC){
         d  = 0;
         km = k;
         kw = 0; // not used since f2 = fx-1 = 0 
         f2 = fx-1; // == 0
         im = i;
         e2 = ex;
      // yy[...]
      } else if (k < YC){
         d  = 1;
         km = k - CL; // k - (fy+fz) = k - (CL-fx) = k - CL
         kw = k - 2 * CL + 1;
         f2 = fy-1;
         im = i - (ey+ez); // we need the general version since ex may not be zero
         e2 = ey;
      // y[...]
      } else if (k == NC - 2){
          d  = 1;
          km = k - CL;
          kw = k - 2 * CL + 1;
          f2 = fy-1;
          im = i - (ey+ez);
          e2 = ey;
      // zzzzzzz
      } else {
          d  = 2;
          km = k - CL - 1;
          kw = k - 2 * CL - 1;
          f2 = fz-1;
          im = i - (ey+ez) - 1;
          e2 = ez;
      }

      idx_000 = mik;
      idx_0m0 = km+i *MC;
      idx_0w0 = kw+i *WC;
      idx_mmp = km+im*MC;

      for ( int mm = 0 ; mm < m ; mm ++ ){
         idx_000 += NA*NC*mm;
         idx_0m0 += NA*MC*mm;
         idx_0w0 += NA*WC*mm;
         idx_mmp += MA*MC*mm;

         i_0m0 = a0cmm0[ idx_0m0 ];
         i_0mp = a0cmm0[ idx_0m0+NA*MC ];
         i_0w0 = a0cwm0[ idx_0w0 ];
         i_0wp = a0cwm0[ idx_0w0+NA*WC ];
         i_mmp = amcmmp[ idx_mmp ];

         a0c0m0[ idx_000 ] = QC[d] * i_0m0 + WQ[d] * i_0mp + f2*( inv_2zcd * i_0w0 + min_rho_zcd2 * i_0wp ) + e2*inv_2z*i_mmp;
      }
   }
}

__device__ void execute_VRR5_gpu(
      const int AL, const int CL, const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ a0cmmp,
      const double* __restrict__ a0cwm0,
      const double* __restrict__ a0cwmp,
      const double* __restrict__ amcmmp,
      const double QC[3], const double WQ[3],
      const double inv_2zcd, const double min_rho_zcd2, const double inv_2z,
      const int my_vrr_rank=0, const int vrr_team_size=1 ){

   const int AC = AL * 100 + CL;
   switch ( AC ){

      case 102 : execute_VRR5<1,2>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 103 : execute_VRR5<1,3>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 104 : execute_VRR5<1,4>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;

      case 202 : execute_VRR5<2,2>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 203 : execute_VRR5<2,3>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 204 : execute_VRR5<2,4>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;

      case 302 : execute_VRR5<3,2>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 303 : execute_VRR5<3,3>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 304 : execute_VRR5<3,4>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;

      case 402 : execute_VRR5<4,2>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 403 : execute_VRR5<4,3>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;
      case 404 : execute_VRR5<4,4>(m,a0c0m0,a0cmm0,a0cmmp,a0cwm0,a0cwmp,amcmmp,QC,WQ,inv_2zcd,min_rho_zcd2,inv_2z,my_vrr_rank,vrr_team_size); break;

      default : assert(false);

   }
}
 
// case for a>0, c=1
__device__ void execute_VRR6_gpu( 
      const int AL, const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ a0cmmp,
      const double* __restrict__ amcmmp,
      const double QC[3], const double WQ[3], const double inv_2z,
      const int my_vrr_rank=0, const int vrr_team_size=1 ){

   const int NcoA = NLco_dev(AL);
   const int NcoAm= NLco_dev(AL-1);
   const int NcoC = 3 ; // NLco_dev(CL);
   // k is the faster variable, followed by i, then m
   for ( int mik=my_vrr_rank; mik < NcoA*NcoC*m; mik+=vrr_team_size ){

      int ex, ey, ez ;
      int i, k, mm, d, im, e2 ;
      int idx_000, idx_0m0, idx_mmp;
      double i_0m0, i_0mp, i_mmp;

      k  = mik % NcoC;
      i  = (mik/NcoC)%NcoA;
      mm = mik/NcoC/NcoA;

      ex = lx_dev(i,AL);
      ey = ly_dev(i,AL);
      ez = AL - ex - ey ; // lz(i,AL);

      if (k == 0 ){
         d  = 0;
         im = i;
         e2 = ex;
      } else if (k == 1){
         d  = 1;
         im = i - (ey+ez);
         e2 = ey;
      } else {
          d  = 2;
          im = i - (ey+ez) - 1;
          e2 = ez;
      }
      idx_000 = mik ; // k +i *NcoC + NcoA *NcoC *(mm)
      idx_0m0 = i  + NcoA *mm;
      idx_mmp = im + NcoAm*mm;
      i_0m0 = a0cmm0[ idx_0m0 ];
      i_0mp = a0cmmp[ idx_0m0 ];
      i_mmp = amcmmp[ idx_mmp ];

      a0c0m0[ idx_000 ] = QC[d] * i_0m0 + WQ[d] * i_0mp + e2*inv_2z*i_mmp;
   }
}

__global__ void compute_VRR_batched_gpu_low(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){
   

   int F_size = L+1;
   if (L > 0){ F_size += 4*3+5; }

   const int MT = 16;
   const int SS = 1024;
   __shared__ double shared_scratch[SS];

   int vrr_team_size = 16;
   int num_vrr_teams = 4;

   assert( num_vrr_teams * vrr_team_size == blockDim.x);
   assert( num_vrr_teams <= MT );

   int my_vrr_team = threadIdx.x / vrr_team_size;
   int my_vrr_rank = threadIdx.x % vrr_team_size;

   const int TS = SS / num_vrr_teams;
   double * scratch = shared_scratch + my_vrr_team * TS;

//   if ( blockIdx.x == 0 and threadIdx.x == 0 ) { printf(" vrr %d (%d.%d) -> %d \n", L, vrr_team_size, num_vrr_teams,TS); }

   for( int block=blockIdx.x; block < Ncells*Ng ; block += gridDim.x ){

      unsigned int p = block / Ng; 
      int n3 = block % Ng;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];

      for ( unsigned i = my_vrr_team; i < n_prm ;  i += num_vrr_teams ){

         // Screening on the (ab.n1|cd.n2@n3) fondamental integrals
         bool found = false;
         unsigned int Of = 0;
         while ( not found and i < n_prm ){
            Of = ((Ov+i) * Ng + n3 ) * F_size;
            // copy Fm[0] ( the ssss(0) integral ) to AC for later screening in SFT
            double* pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            pr_mem[0] = Fm[Of];
            // Immediate screening
            if (Fm[Of] > 0. ){ found = true ; }
            else { i += num_vrr_teams; }
         }
         if ( not found or i >= n_prm ){ break; }

         // Copy the ssss(m) integrals         
         double* pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
         if ( L+1 < TS ){ 
            for( int il = my_vrr_rank; il < L+1; il += vrr_team_size ){
               scratch[il] = Fm[Of+il]; 
            }
         }
         for( int il = my_vrr_rank; il < L+1; il += vrr_team_size ){
            pr_mem[il] = Fm[Of+il];
         }


//         __syncthreads();

         __shared__ double sPA[3*MT];
         __shared__ double sWP[3*MT];
         __shared__ double sQC[3*MT];
         __shared__ double sWQ[3*MT];
         __shared__ double s_inv_2zab[MT];
         __shared__ double s_min_rho_zab2[MT];
         __shared__ double s_inv_2zcd[MT];
         __shared__ double s_min_rho_zcd2[MT];
         __shared__ double s_inv_2z[MT];

         if (L > 0 ){

            sPA[my_vrr_team*3+0] = Fm[Of+L+1 ]; sPA[my_vrr_team*3+1] = Fm[Of+L+2 ]; sPA[my_vrr_team*3+2] = Fm[Of+L+3 ];
            sWP[my_vrr_team*3+0] = Fm[Of+L+4 ]; sWP[my_vrr_team*3+1] = Fm[Of+L+5 ]; sWP[my_vrr_team*3+2] = Fm[Of+L+6 ];
            sQC[my_vrr_team*3+0] = Fm[Of+L+7 ]; sQC[my_vrr_team*3+1] = Fm[Of+L+8 ]; sQC[my_vrr_team*3+2] = Fm[Of+L+9 ];
            sWQ[my_vrr_team*3+0] = Fm[Of+L+10]; sWQ[my_vrr_team*3+1] = Fm[Of+L+11]; sWQ[my_vrr_team*3+2] = Fm[Of+L+12];
            s_inv_2zab[my_vrr_team] = Fm[Of+L+13];
            s_min_rho_zab2[my_vrr_team]  = Fm[Of+L+14]; // - rho/zab**2
            s_inv_2zcd[my_vrr_team] = Fm[Of+L+15];
            s_min_rho_zcd2[my_vrr_team]  = Fm[Of+L+16]; // - rho/zcd**2
            s_inv_2z[my_vrr_team] = Fm[Of+L+17];
         }

//          __syncthreads();

         const double * const PA = &sPA[my_vrr_team*3];
         const double * const WP = &sWP[my_vrr_team*3];
         const double * const QC = &sQC[my_vrr_team*3];
         const double * const WQ = &sWQ[my_vrr_team*3];

         const double inv_2zab = s_inv_2zab[my_vrr_team];
         const double min_rho_zab2 = s_min_rho_zab2[my_vrr_team];
         const double inv_2zcd = s_inv_2zcd[my_vrr_team];
         const double min_rho_zcd2 = s_min_rho_zcd2[my_vrr_team];
         const double inv_2z = s_inv_2z[my_vrr_team];

         for ( int op=0; op < numVC; op++ ){

            const int t  = plan[ op*OP_SIZE + T__OFFSET ];
            const int la = plan[ op*OP_SIZE + LA_OFFSET ];
            const int lc = plan[ op*OP_SIZE + LC_OFFSET ];
            const int max_m = plan[ op*OP_SIZE + M__OFFSET ];
            const int min_m = plan[ op*OP_SIZE + H__OFFSET ];
            const int m = max_m - min_m + 1 ;

            const int  off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
            const int  off_m2 = plan[ op*OP_SIZE + M2_OFFSET ];
            const int  off_m3 = plan[ op*OP_SIZE + M3_OFFSET ];
            const int  off_m4 = plan[ op*OP_SIZE + M4_OFFSET ];
            const int  off_m5 = plan[ op*OP_SIZE + M5_OFFSET ];
            const int  off_m6 = plan[ op*OP_SIZE + M6_OFFSET ];

            const int  lim_m1 = plan[ op*OP_SIZE + L1_OFFSET ];
            const int  lim_m2 = plan[ op*OP_SIZE + L2_OFFSET ];
            const int  lim_m3 = plan[ op*OP_SIZE + L3_OFFSET ];
            const int  lim_m4 = plan[ op*OP_SIZE + L4_OFFSET ];
            const int  lim_m5 = plan[ op*OP_SIZE + L5_OFFSET ];
            const int  lim_m6 = plan[ op*OP_SIZE + L6_OFFSET ];

            double *m1, *m2, *m3;
            double *m4, *m5, *m6;
          
            if ( lim_m1 < TS ){ m1 = &scratch[off_m1]; } else { m1 = &pr_mem[off_m1]; }
            if ( lim_m2 < TS ){ m2 = &scratch[off_m2]; } else { m2 = &pr_mem[off_m2]; }
            if ( lim_m3 < TS ){ m3 = &scratch[off_m3]; } else { m3 = &pr_mem[off_m3]; }
            if ( lim_m4 < TS ){ m4 = &scratch[off_m4]; } else { m4 = &pr_mem[off_m4]; }
            if ( lim_m5 < TS ){ m5 = &scratch[off_m5]; } else { m5 = &pr_mem[off_m5]; }
            if ( lim_m6 < TS ){ m6 = &scratch[off_m6]; } else { m6 = &pr_mem[off_m6]; }
           
//            if ( blockIdx.x == 0 and threadIdx.x == 0 ) { printf(" vrr # %d at l: %d %d off: %d %d %d %d %d %d lim: %d %d %d %d %d %d \n", 
//                  op, la,lc, off_m1,off_m2,off_m3,off_m4,off_m5,off_m6, lim_m1,lim_m2,lim_m3,lim_m4,lim_m5,lim_m6); }
//            continue;

            // TODO is it worth it to unify the VRRs types as a single, maybe individually less efficient, call ?
            if ( t == VRR1 ){ 
               execute_VRR1_gpu( m, m1, m2, m3, PA, WP, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR2 ){
               execute_VRR2_gpu( la, m, m1, m2, m3, m4, m5, PA, WP, inv_2zab, min_rho_zab2, my_vrr_rank, vrr_team_size);
            } else if ( t == VRR3 ){ 
               execute_VRR1_gpu( m, m1, m2, m3, QC, WQ, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR4 ){ 
               execute_VRR2_gpu( lc, m, m1, m2, m3, m4, m5, QC, WQ, inv_2zcd, min_rho_zcd2, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR5 ){ 
               execute_VRR5_gpu( la, lc, m, m1, m2, m3, m4, m5, m6, 
                  QC, WQ, inv_2zcd, min_rho_zcd2, inv_2z, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR6 ){ 
               execute_VRR6_gpu( la, m, m1, m2, m3, m4, QC, WQ, inv_2z, my_vrr_rank, vrr_team_size);
            } else if ( t == SYTM and vrr_team_size > CUDA_WARPSIZE ){
//               __syncthreads();
            }
         }

         for ( int op=0; op < numVC; op++ ){

            const int t  = plan[ op*OP_SIZE + T__OFFSET ];
            const int la = plan[ op*OP_SIZE + LA_OFFSET ];
            const int lc = plan[ op*OP_SIZE + LC_OFFSET ];
            const int min_m = plan[ op*OP_SIZE + H__OFFSET ];
            const int  off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
            const int  lim_m1 = plan[ op*OP_SIZE + L1_OFFSET ];
          
            // Copy (a,c,0) to pr // TODO: plan to copy only if needed later by sft for eco and hrr
            if ( t <= VRR6 and min_m == 0 and lim_m1 < TS ){
               int NA = ((la+1)*(la+2))/2;
               int NC = ((lc+1)*(lc+2))/2;
//               if ( blockIdx.x == 0 and threadIdx.x == 0 ) { printf("Force writing %d element at l: %d %d from %d to %d \n", NA*NC, la,lc, off_m1, lim_m1 ); }
               for( int mik=my_vrr_rank; mik < NA*NC; mik += vrr_team_size ){
                  pr_mem[off_m1 + mik ] = scratch[off_m1 + mik];
               }
            }
         } // end of loop over op

//         __syncthreads();

      } // end of loop over prm
   } // end of loop over blocks
}




