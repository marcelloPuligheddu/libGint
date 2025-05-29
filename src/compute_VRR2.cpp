#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include<cstdint>
#include <iostream>
#include "compute_VRR.h"
#include "util.h"
#include "define.h"

using std::cout;
using std::endl;

#define FM_N_VEC 4
#define FM_N_SCA 5
__device__ __host__ constexpr int Fsize_but_constexpr( int L ){ return L + 1 + FM_N_VEC * 3 + FM_N_SCA; }
#undef FM_N_VEC
#undef FM_N_SCA



__host__ __device__ constexpr int SA ( int L ) {
   int tmp[] = {0,1,4,10,20,35,56, 84,120, 165};
   return tmp[L];
}

__host__ __device__ constexpr int NLco_dev_but_constexpr( int L ){ return (L+1)*(L+2) / 2; }

constexpr int MAXL = 4*2;

__constant__ int _d[ SA(MAXL+1) ] = {
0, 
0, 1, 2, 
0, 0, 0, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
};

__constant__ int _idx_m[ SA(MAXL+1) ] = {
0, 
0, 0, 0, 
0, 1, 2, 1, 2, 2, 
0, 1, 2, 3, 4, 5, 3, 4, 5, 5, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 9, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 14, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 15, 16, 17, 18, 19, 20, 20, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 21, 22, 23, 24, 25, 26, 27, 27, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 28, 29, 30, 31, 32, 33, 34, 35, 35, 
};

__constant__ int _idx_w[ SA(MAXL+1) ] = {
0, 
0, 0, 0, 
0, 1, 2, 0, 1, 0, 
0, 1, 2, 3, 4, 5, 1, 2, 3, 2, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 5, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 6, 7, 8, 9, 10, 9, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10, 11, 12, 13, 14, 15, 14, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 15, 16, 17, 18, 19, 20, 21, 20, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 21, 22, 23, 24, 25, 26, 27, 28, 27, 
};

__constant__ double _f2[ SA(MAXL+1) ] = {
0, 
0, 0, 0, 
1, 0, 0, 1, 0, 1, 
2, 1, 1, 0, 0, 0, 2, 1, 0, 2, 
3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 2, 1, 0, 3, 
4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 3, 2, 1, 0, 4, 
5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 5, 
6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4, 3, 2, 1, 0, 6, 
7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0, 7, 
};

__constant__ int _idx_off[ 3*(MAXL+1) ] = {
     0,  0,  0,
     3,  6,  9,
    12, 18, 24,
    30, 40, 50,
    60, 75, 90,
   105,126,147,
   168,196,224,
   252,288,324,
   360,405,450
};


__constant__ uint8_t _idx_r[ 3*SA(MAXL+1) ] = {
0, 
0, 
0, // 3
0, 1, 2, 
0, 0, 1, 
0, 0, 0, // 12
0, 1, 2, 3, 4, 5, 
0, 0, 1, 1, 2, 3, 
0, 0, 0, 0, 1, 2, // 30
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, // 60
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, // 105
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, // 168
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, // 84*3 = 252
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, // 120*3 = 360
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 33, 34, 35,  // 165*3 = 495
};

__constant__ double _e2[ 3*SA(MAXL+1) ] = {
0, 
0, 
0, 
1, 0, 0, 
0, 1, 0, 
0, 0, 1, 
2, 1, 1, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 
3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 
4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 
5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 
6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 
7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 
8, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8,   
};


template < int VTS, int AL, int CL, int m >
__device__ void execute_VRR1_gpu( // int AL, int CL, int m, 
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double PA_WP[6] ){

   int my_vrr_rank = threadIdx.x % VTS ;
   for( int imm=my_vrr_rank; imm < 3*m; imm+=VTS ){
      int mm = imm / 3 ;
      int i  = imm % 3 ;
      int d  = i ;
      int im = 0 ; 
      int idx_000 = i  + 3*mm;
      int idx_m00 = im + 1*mm;
      int idx_m0p = im + 1*(1+mm);
      a0c0m0[idx_000] = PA_WP[  d] * amc0m0[idx_m00] + 
                        PA_WP[d+3] * amc0m0[idx_m0p];

//      printf(" (1) i mm d %d %d %d | PA/QC * M00 %lg %d | WP/WQ * M0P %lg %d | \n", i,mm,d,PA_WP[d],idx_m00, PA_WP[d+3],idx_m0p);

   }
}

template < int VTS, int AL, int CL, int m >
__device__ void execute_VRR2_gpu(
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ awc0m0,
      const double PA_WP[3], const double inv_2zab[2] ){
 
   int my_vrr_rank = threadIdx.x % VTS ; 
   const int NcoA   = ((AL+1)*(AL+2))/2;
   const int NcoAm  = ((AL+0)*(AL+1))/2;
   const int NcoAw  = ((AL-1)*(AL+0))/2;

   for ( int imm = my_vrr_rank; imm < m*NcoA; imm+=VTS ){

      int i  = imm % NcoA ;
      int mm = imm / NcoA ;
      int d  = _d[SA(AL)+i];
      int im = _idx_m[SA(AL)+i];
      int iw = _idx_w[SA(AL)+i];
      double e2 = _f2[SA(AL)+i];

      int idx_000 = imm ;
      int idx_m00 = im + NcoAm*mm ;
      int idx_w00 = iw + NcoAw*mm ;
      int idx_m0p = idx_m00 + NcoAm;
      int idx_w0p = idx_w00 + NcoAw;

      a0c0m0[idx_000] = PA_WP[d  ] * amc0m0[idx_m00] + 
                        PA_WP[d+3] * amc0m0[idx_m0p] + 
                        e2*( inv_2zab[0] * awc0m0[idx_w00] + inv_2zab[1] * awc0m0[idx_w0p] );

//      printf(" (2) i m d im iw %d %d %d %d %d | PA/QC * M00 %lg %d | WP/WQ * M0P %lg %d | e2 * z * W00 + e2 * z * W0P %lg %lg %d %lg %d \n", 
//                 i,mm,d,im,iw,               PA_WP[d],idx_m00, PA_WP[d+3],idx_m0p,         e2,inv_2zab[0],idx_w00,inv_2zab[1],idx_w0p );
   }
}


// more general case, for al>0 and cl>1
template < int VTS, int AL, int CL, int m >
__device__ void execute_VRR5_gpu(
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ a0cwm0,
      const double* __restrict__ amcmmp,
      const double QC_WQ[6],
      const double inv_2zcd[3] ){

   int my_vrr_rank = threadIdx.x % VTS ;

   constexpr int NcoA  = NLco_dev_but_constexpr(AL);
   constexpr int NcoAm = NLco_dev_but_constexpr(AL-1);
   constexpr int NcoC  = NLco_dev_but_constexpr(CL);
   constexpr int NcoCm = NLco_dev_but_constexpr(CL-1);
   constexpr int NcoCw = NLco_dev_but_constexpr(CL-2);

   // k is the faster variable, followed by i
   for ( int ik=my_vrr_rank; ik < NcoA*NcoC; ik+=VTS ){

      double f2, e2;
      int i, k, d, km, kw, im, o;
      int idx_000, idx_0m0, idx_0w0, idx_mmp, idx_0mp, idx_0wp;
      double i_0m0, i_0mp, i_0w0, i_0wp, i_mmp;

      k  = ik % NcoC;
      i  = ik / NcoC;
      constexpr int t = SA(CL);
      d  = _d[t+k];
      km = _idx_m[t+k];
      kw = _idx_w[t+k];
      f2 = _f2[t+k];
      o  = _idx_off[3*AL+d];
      im = _idx_r[o+i];
      e2 = _e2[o+i];

      for ( int mm = 0; mm < m; mm++ ){
         idx_000 = k +i *NcoC  + NcoA *NcoC *mm;
         idx_0m0 = km+i *NcoCm + NcoA *NcoCm*mm;
         idx_0mp = km+i *NcoCm + NcoA *NcoCm*(1+mm);
         idx_0w0 = kw+i *NcoCw + NcoA *NcoCw*mm;
         idx_0wp = kw+i *NcoCw + NcoA *NcoCw*(1+mm);
         idx_mmp = km+im*NcoCm + NcoAm*NcoCm*(mm);

         i_0m0 = a0cmm0[ idx_0m0 ];
         i_0mp = a0cmm0[ idx_0mp ];
         i_0w0 = a0cwm0[ idx_0w0 ];
         i_0wp = a0cwm0[ idx_0wp ];
         i_mmp = amcmmp[ idx_mmp ];

         a0c0m0[ idx_000 ] = QC_WQ[d] * i_0m0 + QC_WQ[d+3] * i_0mp + f2*( inv_2zcd[0] * i_0w0 + inv_2zcd[1] * i_0wp ) + e2*inv_2zcd[2]*i_mmp;

//         printf(" (5) %d %d | i k m d km kw f2 o im e2 %d %d %d %d %d %d %lg %d %d %lg \n", AL,CL, i,k,mm,d,km,kw,f2,o,im,e2 );
 
      }
   }
}

template< int VTS, int  AL, int CL, int m >
__device__ void execute_VRR6_gpu( 
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ amcmmp,
      const double QC_WQ[6], const double inv_2z ){

   int my_vrr_rank = threadIdx.x % VTS ;
      
   const int NcoA = NLco_dev_but_constexpr(AL);
   const int NcoAm= NLco_dev_but_constexpr(AL-1);
   const int NcoC = NLco_dev_but_constexpr(CL);
   const int NcoCm= NLco_dev_but_constexpr(CL-1);
   // k is the faster variable, followed by i, then m

   for ( int ik=my_vrr_rank; ik < NcoA*NcoC; ik+=VTS ){

      int i, k, km, d, im, o;
      double e2 ;
      int idx_000, idx_0m0, idx_0mp, idx_mmp;
      double i_0m0, i_0mp, i_mmp;

      k  = ik % NcoC;
      i  = ik / NcoC;

      d = k;
      o  = _idx_off[3*AL+d];
      im = _idx_r[o+i];
      e2 = _e2[o+i];
      km = 0;

      for ( int mm = 0; mm < m; mm++ ){
         idx_000 = k +i *NcoC  + NcoA *NcoC *(  mm);
         idx_0m0 = km+i *NcoCm + NcoA *NcoCm*(  mm);
         idx_0mp = km+i *NcoCm + NcoA *NcoCm*(1+mm);
         idx_mmp = km+im*NcoCm + NcoAm*NcoCm*(mm);

         i_0m0 = a0cmm0[ idx_0m0 ];
         i_0mp = a0cmm0[ idx_0mp ];
         i_mmp = amcmmp[ idx_mmp ];

         a0c0m0[ idx_000 ] = QC_WQ[d] * i_0m0 + QC_WQ[d+3] * i_0mp + e2*inv_2z*i_mmp;

//         printf(" (6) %d %d | i k m d km o im e2 %d %d %d %d %d %d %d %lg \n", AL,CL, i,k,mm,d,km,o,im,e2 );
      }
   }
}





template< int VTS >
__device__ void ssss(double * ss0 , double para[4*3+5] ){
}

template< int VTS >
__device__ void sssp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double*  sp0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
}

template< int VTS >
__device__ void sssd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 3 ;
   double*  sd0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sssf(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssps(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double*  sp0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
}

template< int VTS >
__device__ void sspp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 3 ;
   double*  sd0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sspd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sspf(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 5 ;
   double*  sd0  = ss0 + 17 ;
   double*  sf0  = ss0 + 35 ;
   double*  sg0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssds(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 3 ;
   double*  sd0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssdp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssdd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 5 ;
   double*  sd0  = ss0 + 17 ;
   double*  sf0  = ss0 + 35 ;
   double*  sg0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssdf(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 6 ;
   double*  sd0  = ss0 + 21 ;
   double*  sf0  = ss0 + 45 ;
   double*  sg0  = ss0 + 75 ;
   double*  sh0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssfs(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssfp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 5 ;
   double*  sd0  = ss0 + 17 ;
   double*  sf0  = ss0 + 35 ;
   double*  sg0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssfd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 6 ;
   double*  sd0  = ss0 + 21 ;
   double*  sf0  = ss0 + 45 ;
   double*  sg0  = ss0 + 75 ;
   double*  sh0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssff(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 7 ;
   double*  sd0  = ss0 + 25 ;
   double*  sf0  = ss0 + 55 ;
   double*  sg0  = ss0 + 95 ;
   double*  sh0  = ss0 + 140 ;
   double*  si0  = ss0 + 182 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double*  ps0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( ps0 , ss0 , PA_WP);
}

template< int VTS >
__device__ void spsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  sp0  = ss0 + 9 ;
   double*  pp0  = ss0 + 12 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void spsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp0  = ss0 + 13 ;
   double*  sp1  = ss0 + 16 ;
   double*  sd0  = ss0 + 19 ;
   double*  pp0  = ss0 + 25 ;
   double*  pd0  = ss0 + 43 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  pp0  = ss0 + 26 ;
   double*  sd0  = ss0 + 53 ;
   double*  sd1  = ss0 + 59 ;
   double*  sf0  = ss0 + 65 ;
   double*  pd0  = ss0 + 75 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  sp0  = ss0 + 9 ;
   double*  pp0  = ss0 + 12 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void sppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp0  = ss0 + 13 ;
   double*  sp1  = ss0 + 16 ;
   double*  pp0  = ss0 + 19 ;
   double*  sd0  = ss0 + 37 ;
   double*  pd0  = ss0 + 43 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  sd0  = ss0 + 26 ;
   double*  sd1  = ss0 + 32 ;
   double*  pp0  = ss0 + 38 ;
   double*  pd0  = ss0 + 65 ;
   double*  sf0  = ss0 + 101 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp0  = ss0 + 21 ;
   double*  sp1  = ss0 + 24 ;
   double*  pp0  = ss0 + 33 ;
   double*  sd0  = ss0 + 69 ;
   double*  sd1  = ss0 + 75 ;
   double*  sf0  = ss0 + 87 ;
   double*  sf1  = ss0 + 97 ;
   double*  pd0  = ss0 + 107 ;
   double*  sg0  = ss0 + 161 ;
   double*  pf0  = ss0 + 176 ;
   double*  pg0  = ss0 + 236 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp0  = ss0 + 13 ;
   double*  sp1  = ss0 + 16 ;
   double*  sd0  = ss0 + 19 ;
   double*  pp0  = ss0 + 25 ;
   double*  pd0  = ss0 + 43 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  sd0  = ss0 + 26 ;
   double*  sd1  = ss0 + 32 ;
   double*  pp0  = ss0 + 38 ;
   double*  pd0  = ss0 + 65 ;
   double*  sf0  = ss0 + 101 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp0  = ss0 + 21 ;
   double*  sp1  = ss0 + 24 ;
   double*  pp0  = ss0 + 33 ;
   double*  sd0  = ss0 + 69 ;
   double*  sd1  = ss0 + 75 ;
   double*  sf0  = ss0 + 87 ;
   double*  sf1  = ss0 + 97 ;
   double*  pd0  = ss0 + 107 ;
   double*  pf0  = ss0 + 161 ;
   double*  sg0  = ss0 + 221 ;
   double*  pg0  = ss0 + 236 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp0  = ss0 + 25 ;
   double*  sp1  = ss0 + 28 ;
   double*  pp0  = ss0 + 40 ;
   double*  sd0  = ss0 + 85 ;
   double*  sd1  = ss0 + 91 ;
   double*  pd0  = ss0 + 109 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  sg0  = ss0 + 211 ;
   double*  sg1  = ss0 + 226 ;
   double*  pf0  = ss0 + 241 ;
   double*  sh0  = ss0 + 331 ;
   double*  pg0  = ss0 + 352 ;
   double*  ph0  = ss0 + 442 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  sd0  = ss0 + 26 ;
   double*  sd1  = ss0 + 32 ;
   double*  pp0  = ss0 + 38 ;
   double*  pd0  = ss0 + 65 ;
   double*  sf0  = ss0 + 101 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp0  = ss0 + 21 ;
   double*  sp1  = ss0 + 24 ;
   double*  sd0  = ss0 + 33 ;
   double*  sd1  = ss0 + 39 ;
   double*  pp0  = ss0 + 51 ;
   double*  pd0  = ss0 + 87 ;
   double*  sf0  = ss0 + 141 ;
   double*  sf1  = ss0 + 151 ;
   double*  sg0  = ss0 + 161 ;
   double*  pf0  = ss0 + 176 ;
   double*  pg0  = ss0 + 236 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp0  = ss0 + 25 ;
   double*  sp1  = ss0 + 28 ;
   double*  pp0  = ss0 + 40 ;
   double*  sd0  = ss0 + 85 ;
   double*  sd1  = ss0 + 91 ;
   double*  pd0  = ss0 + 109 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  sg0  = ss0 + 211 ;
   double*  sg1  = ss0 + 226 ;
   double*  pf0  = ss0 + 241 ;
   double*  sh0  = ss0 + 331 ;
   double*  pg0  = ss0 + 352 ;
   double*  ph0  = ss0 + 442 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  sp0  = ss0 + 29 ;
   double*  sp1  = ss0 + 32 ;
   double*  sd0  = ss0 + 47 ;
   double*  sd1  = ss0 + 53 ;
   double*  pp0  = ss0 + 77 ;
   double*  pd0  = ss0 + 131 ;
   double*  sf0  = ss0 + 221 ;
   double*  sf1  = ss0 + 231 ;
   double*  sg0  = ss0 + 261 ;
   double*  sg1  = ss0 + 276 ;
   double*  pf0  = ss0 + 306 ;
   double*  sh0  = ss0 + 426 ;
   double*  sh1  = ss0 + 447 ;
   double*  pg0  = ss0 + 468 ;
   double*  ph0  = ss0 + 603 ;
   double*  si0  = ss0 + 729 ;
   double*  pi0  = ss0 + 757 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 3 ;
   double*  ds0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( ds0 , ps0 , ss0 , PA_WP, zab );
}

template< int VTS >
__device__ void sdsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  sp0  = ss0 + 25 ;
   double*  pp0  = ss0 + 28 ;
   double*  dp0  = ss0 + 37 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void sdsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp0  = ss0 + 35 ;
   double*  sp1  = ss0 + 38 ;
   double*  sd0  = ss0 + 41 ;
   double*  pp0  = ss0 + 47 ;
   double*  pp1  = ss0 + 56 ;
   double*  pd0  = ss0 + 65 ;
   double*  dp0  = ss0 + 83 ;
   double*  dd0  = ss0 + 119 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  sd0  = ss0 + 54 ;
   double*  sd1  = ss0 + 60 ;
   double*  pp0  = ss0 + 66 ;
   double*  pp1  = ss0 + 75 ;
   double*  sf0  = ss0 + 93 ;
   double*  pd0  = ss0 + 103 ;
   double*  pd1  = ss0 + 121 ;
   double*  dp0  = ss0 + 139 ;
   double*  pf0  = ss0 + 193 ;
   double*  dd0  = ss0 + 223 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  sp0  = ss0 + 25 ;
   double*  pp0  = ss0 + 28 ;
   double*  dp0  = ss0 + 37 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void sdpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp0  = ss0 + 35 ;
   double*  sp1  = ss0 + 38 ;
   double*  sd0  = ss0 + 41 ;
   double*  pp0  = ss0 + 47 ;
   double*  pp1  = ss0 + 56 ;
   double*  pd0  = ss0 + 65 ;
   double*  dp0  = ss0 + 83 ;
   double*  dd0  = ss0 + 119 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  pp0  = ss0 + 54 ;
   double*  pp1  = ss0 + 63 ;
   double*  sd0  = ss0 + 81 ;
   double*  sd1  = ss0 + 87 ;
   double*  sf0  = ss0 + 93 ;
   double*  dp0  = ss0 + 103 ;
   double*  pd0  = ss0 + 157 ;
   double*  pd1  = ss0 + 175 ;
   double*  dd0  = ss0 + 193 ;
   double*  pf0  = ss0 + 265 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp0  = ss0 + 55 ;
   double*  sp1  = ss0 + 58 ;
   double*  pp0  = ss0 + 67 ;
   double*  pp1  = ss0 + 76 ;
   double*  sd0  = ss0 + 103 ;
   double*  sd1  = ss0 + 109 ;
   double*  dp0  = ss0 + 121 ;
   double*  pd0  = ss0 + 193 ;
   double*  pd1  = ss0 + 211 ;
   double*  sf0  = ss0 + 247 ;
   double*  sf1  = ss0 + 257 ;
   double*  dd0  = ss0 + 267 ;
   double*  pf0  = ss0 + 375 ;
   double*  pf1  = ss0 + 405 ;
   double*  sg0  = ss0 + 435 ;
   double*  df0  = ss0 + 450 ;
   double*  pg0  = ss0 + 570 ;
   double*  dg0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp0  = ss0 + 35 ;
   double*  sp1  = ss0 + 38 ;
   double*  sd0  = ss0 + 41 ;
   double*  pp0  = ss0 + 47 ;
   double*  pp1  = ss0 + 56 ;
   double*  pd0  = ss0 + 65 ;
   double*  dp0  = ss0 + 83 ;
   double*  dd0  = ss0 + 119 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  pp0  = ss0 + 54 ;
   double*  pp1  = ss0 + 63 ;
   double*  sd0  = ss0 + 81 ;
   double*  sd1  = ss0 + 87 ;
   double*  sf0  = ss0 + 93 ;
   double*  dp0  = ss0 + 103 ;
   double*  pd0  = ss0 + 157 ;
   double*  pd1  = ss0 + 175 ;
   double*  dd0  = ss0 + 193 ;
   double*  pf0  = ss0 + 265 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp0  = ss0 + 55 ;
   double*  sp1  = ss0 + 58 ;
   double*  sd0  = ss0 + 67 ;
   double*  sd1  = ss0 + 73 ;
   double*  pp0  = ss0 + 85 ;
   double*  pp1  = ss0 + 94 ;
   double*  sf0  = ss0 + 121 ;
   double*  sf1  = ss0 + 131 ;
   double*  pd0  = ss0 + 141 ;
   double*  pd1  = ss0 + 159 ;
   double*  dp0  = ss0 + 195 ;
   double*  sg0  = ss0 + 267 ;
   double*  pf0  = ss0 + 282 ;
   double*  pf1  = ss0 + 312 ;
   double*  dd0  = ss0 + 342 ;
   double*  pg0  = ss0 + 450 ;
   double*  df0  = ss0 + 495 ;
   double*  dg0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp0  = ss0 + 65 ;
   double*  sp1  = ss0 + 68 ;
   double*  pp0  = ss0 + 80 ;
   double*  pp1  = ss0 + 89 ;
   double*  sd0  = ss0 + 125 ;
   double*  sd1  = ss0 + 131 ;
   double*  dp0  = ss0 + 149 ;
   double*  pd0  = ss0 + 239 ;
   double*  pd1  = ss0 + 257 ;
   double*  sf0  = ss0 + 311 ;
   double*  sf1  = ss0 + 321 ;
   double*  sg0  = ss0 + 341 ;
   double*  sg1  = ss0 + 356 ;
   double*  dd0  = ss0 + 371 ;
   double*  pf0  = ss0 + 515 ;
   double*  pf1  = ss0 + 545 ;
   double*  sh0  = ss0 + 605 ;
   double*  df0  = ss0 + 626 ;
   double*  pg0  = ss0 + 806 ;
   double*  pg1  = ss0 + 851 ;
   double*  dg0  = ss0 + 896 ;
   double*  ph0  = ss0 + 1076 ;
   double*  dh0  = ss0 + 1139 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  pp0  = ss0 + 54 ;
   double*  pp1  = ss0 + 63 ;
   double*  sd0  = ss0 + 81 ;
   double*  sd1  = ss0 + 87 ;
   double*  pd0  = ss0 + 93 ;
   double*  pd1  = ss0 + 111 ;
   double*  dp0  = ss0 + 129 ;
   double*  sf0  = ss0 + 183 ;
   double*  dd0  = ss0 + 193 ;
   double*  pf0  = ss0 + 265 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp0  = ss0 + 55 ;
   double*  sp1  = ss0 + 58 ;
   double*  sd0  = ss0 + 67 ;
   double*  sd1  = ss0 + 73 ;
   double*  pp0  = ss0 + 85 ;
   double*  pp1  = ss0 + 94 ;
   double*  sf0  = ss0 + 121 ;
   double*  sf1  = ss0 + 131 ;
   double*  pd0  = ss0 + 141 ;
   double*  pd1  = ss0 + 159 ;
   double*  dp0  = ss0 + 195 ;
   double*  sg0  = ss0 + 267 ;
   double*  pf0  = ss0 + 282 ;
   double*  pf1  = ss0 + 312 ;
   double*  dd0  = ss0 + 342 ;
   double*  pg0  = ss0 + 450 ;
   double*  df0  = ss0 + 495 ;
   double*  dg0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp0  = ss0 + 65 ;
   double*  sp1  = ss0 + 68 ;
   double*  pp0  = ss0 + 80 ;
   double*  pp1  = ss0 + 89 ;
   double*  sd0  = ss0 + 125 ;
   double*  sd1  = ss0 + 131 ;
   double*  dp0  = ss0 + 149 ;
   double*  pd0  = ss0 + 239 ;
   double*  pd1  = ss0 + 257 ;
   double*  sf0  = ss0 + 311 ;
   double*  sf1  = ss0 + 321 ;
   double*  sg0  = ss0 + 341 ;
   double*  sg1  = ss0 + 356 ;
   double*  dd0  = ss0 + 371 ;
   double*  pf0  = ss0 + 515 ;
   double*  pf1  = ss0 + 545 ;
   double*  sh0  = ss0 + 605 ;
   double*  df0  = ss0 + 626 ;
   double*  pg0  = ss0 + 806 ;
   double*  pg1  = ss0 + 851 ;
   double*  dg0  = ss0 + 896 ;
   double*  ph0  = ss0 + 1076 ;
   double*  dh0  = ss0 + 1139 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  sd0  = ss0 + 93 ;
   double*  sd1  = ss0 + 99 ;
   double*  pp0  = ss0 + 123 ;
   double*  pp1  = ss0 + 132 ;
   double*  sf0  = ss0 + 177 ;
   double*  sf1  = ss0 + 187 ;
   double*  dp0  = ss0 + 217 ;
   double*  pd0  = ss0 + 325 ;
   double*  pd1  = ss0 + 343 ;
   double*  sg0  = ss0 + 415 ;
   double*  sg1  = ss0 + 430 ;
   double*  dd0  = ss0 + 460 ;
   double*  pf0  = ss0 + 640 ;
   double*  pf1  = ss0 + 670 ;
   double*  pg0  = ss0 + 760 ;
   double*  pg1  = ss0 + 805 ;
   double*  sh0  = ss0 + 895 ;
   double*  sh1  = ss0 + 916 ;
   double*  df0  = ss0 + 937 ;
   double*  si0  = ss0 + 1177 ;
   double*  ph0  = ss0 + 1205 ;
   double*  ph1  = ss0 + 1268 ;
   double*  dg0  = ss0 + 1331 ;
   double*  pi0  = ss0 + 1601 ;
   double*  dh0  = ss0 + 1685 ;
   double*  di0  = ss0 + 1937 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void sfsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  sp0  = ss0 + 55 ;
   double*  pp0  = ss0 + 58 ;
   double*  dp0  = ss0 + 67 ;
   double*  fp0  = ss0 + 85 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void sfsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  pp0  = ss0 + 81 ;
   double*  pp1  = ss0 + 90 ;
   double*  sd0  = ss0 + 99 ;
   double*  pd0  = ss0 + 105 ;
   double*  dp0  = ss0 + 123 ;
   double*  dp1  = ss0 + 141 ;
   double*  fp0  = ss0 + 159 ;
   double*  dd0  = ss0 + 219 ;
   double*  fd0  = ss0 + 255 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  sf0  = ss0 + 143 ;
   double*  dp0  = ss0 + 153 ;
   double*  dp1  = ss0 + 171 ;
   double*  pd0  = ss0 + 207 ;
   double*  pd1  = ss0 + 225 ;
   double*  fp0  = ss0 + 243 ;
   double*  dd0  = ss0 + 333 ;
   double*  dd1  = ss0 + 369 ;
   double*  pf0  = ss0 + 405 ;
   double*  fd0  = ss0 + 435 ;
   double*  df0  = ss0 + 555 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  sp0  = ss0 + 55 ;
   double*  pp0  = ss0 + 58 ;
   double*  dp0  = ss0 + 67 ;
   double*  fp0  = ss0 + 85 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void sfpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  sd0  = ss0 + 81 ;
   double*  pp0  = ss0 + 87 ;
   double*  pp1  = ss0 + 96 ;
   double*  pd0  = ss0 + 105 ;
   double*  dp0  = ss0 + 123 ;
   double*  dp1  = ss0 + 141 ;
   double*  fp0  = ss0 + 159 ;
   double*  dd0  = ss0 + 219 ;
   double*  fd0  = ss0 + 255 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  pd0  = ss0 + 143 ;
   double*  pd1  = ss0 + 161 ;
   double*  sf0  = ss0 + 179 ;
   double*  dp0  = ss0 + 189 ;
   double*  dp1  = ss0 + 207 ;
   double*  pf0  = ss0 + 243 ;
   double*  dd0  = ss0 + 273 ;
   double*  dd1  = ss0 + 309 ;
   double*  fp0  = ss0 + 345 ;
   double*  df0  = ss0 + 435 ;
   double*  fd0  = ss0 + 495 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp0  = ss0 + 115 ;
   double*  sp1  = ss0 + 118 ;
   double*  pp0  = ss0 + 127 ;
   double*  pp1  = ss0 + 136 ;
   double*  sd0  = ss0 + 163 ;
   double*  sd1  = ss0 + 169 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  dp0  = ss0 + 201 ;
   double*  dp1  = ss0 + 219 ;
   double*  pd0  = ss0 + 273 ;
   double*  pd1  = ss0 + 291 ;
   double*  sg0  = ss0 + 327 ;
   double*  fp0  = ss0 + 342 ;
   double*  dd0  = ss0 + 462 ;
   double*  dd1  = ss0 + 498 ;
   double*  pf0  = ss0 + 570 ;
   double*  pf1  = ss0 + 600 ;
   double*  pg0  = ss0 + 630 ;
   double*  fd0  = ss0 + 675 ;
   double*  df0  = ss0 + 855 ;
   double*  df1  = ss0 + 915 ;
   double*  ff0  = ss0 + 975 ;
   double*  dg0  = ss0 + 1175 ;
   double*  fg0  = ss0 + 1265 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  sd0  = ss0 + 81 ;
   double*  pp0  = ss0 + 87 ;
   double*  pp1  = ss0 + 96 ;
   double*  dp0  = ss0 + 105 ;
   double*  dp1  = ss0 + 123 ;
   double*  pd0  = ss0 + 141 ;
   double*  fp0  = ss0 + 159 ;
   double*  dd0  = ss0 + 219 ;
   double*  fd0  = ss0 + 255 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  pd0  = ss0 + 143 ;
   double*  pd1  = ss0 + 161 ;
   double*  dp0  = ss0 + 179 ;
   double*  dp1  = ss0 + 197 ;
   double*  sf0  = ss0 + 233 ;
   double*  pf0  = ss0 + 243 ;
   double*  dd0  = ss0 + 273 ;
   double*  dd1  = ss0 + 309 ;
   double*  fp0  = ss0 + 345 ;
   double*  df0  = ss0 + 435 ;
   double*  fd0  = ss0 + 495 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp0  = ss0 + 115 ;
   double*  sp1  = ss0 + 118 ;
   double*  pp0  = ss0 + 127 ;
   double*  pp1  = ss0 + 136 ;
   double*  sd0  = ss0 + 163 ;
   double*  sd1  = ss0 + 169 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  dp0  = ss0 + 201 ;
   double*  dp1  = ss0 + 219 ;
   double*  pd0  = ss0 + 273 ;
   double*  pd1  = ss0 + 291 ;
   double*  sg0  = ss0 + 327 ;
   double*  fp0  = ss0 + 342 ;
   double*  dd0  = ss0 + 462 ;
   double*  dd1  = ss0 + 498 ;
   double*  pf0  = ss0 + 570 ;
   double*  pf1  = ss0 + 600 ;
   double*  pg0  = ss0 + 630 ;
   double*  fd0  = ss0 + 675 ;
   double*  df0  = ss0 + 855 ;
   double*  df1  = ss0 + 915 ;
   double*  ff0  = ss0 + 975 ;
   double*  dg0  = ss0 + 1175 ;
   double*  fg0  = ss0 + 1265 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp0  = ss0 + 135 ;
   double*  sp1  = ss0 + 138 ;
   double*  sd0  = ss0 + 150 ;
   double*  sd1  = ss0 + 156 ;
   double*  pp0  = ss0 + 174 ;
   double*  pp1  = ss0 + 183 ;
   double*  sf0  = ss0 + 219 ;
   double*  sf1  = ss0 + 229 ;
   double*  dp0  = ss0 + 249 ;
   double*  dp1  = ss0 + 267 ;
   double*  pd0  = ss0 + 339 ;
   double*  pd1  = ss0 + 357 ;
   double*  sg0  = ss0 + 411 ;
   double*  sg1  = ss0 + 426 ;
   double*  pf0  = ss0 + 441 ;
   double*  pf1  = ss0 + 471 ;
   double*  fp0  = ss0 + 531 ;
   double*  dd0  = ss0 + 681 ;
   double*  dd1  = ss0 + 717 ;
   double*  sh0  = ss0 + 825 ;
   double*  pg0  = ss0 + 846 ;
   double*  pg1  = ss0 + 891 ;
   double*  df0  = ss0 + 936 ;
   double*  df1  = ss0 + 996 ;
   double*  fd0  = ss0 + 1116 ;
   double*  ph0  = ss0 + 1356 ;
   double*  dg0  = ss0 + 1419 ;
   double*  dg1  = ss0 + 1509 ;
   double*  ff0  = ss0 + 1599 ;
   double*  dh0  = ss0 + 1899 ;
   double*  fg0  = ss0 + 2025 ;
   double*  fh0  = ss0 + 2325 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  pd0  = ss0 + 143 ;
   double*  pd1  = ss0 + 161 ;
   double*  sf0  = ss0 + 179 ;
   double*  dp0  = ss0 + 189 ;
   double*  dp1  = ss0 + 207 ;
   double*  pf0  = ss0 + 243 ;
   double*  dd0  = ss0 + 273 ;
   double*  dd1  = ss0 + 309 ;
   double*  fp0  = ss0 + 345 ;
   double*  df0  = ss0 + 435 ;
   double*  fd0  = ss0 + 495 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp0  = ss0 + 115 ;
   double*  sp1  = ss0 + 118 ;
   double*  pp0  = ss0 + 127 ;
   double*  pp1  = ss0 + 136 ;
   double*  sd0  = ss0 + 163 ;
   double*  sd1  = ss0 + 169 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  dp0  = ss0 + 201 ;
   double*  dp1  = ss0 + 219 ;
   double*  pd0  = ss0 + 273 ;
   double*  pd1  = ss0 + 291 ;
   double*  sg0  = ss0 + 327 ;
   double*  dd0  = ss0 + 342 ;
   double*  dd1  = ss0 + 378 ;
   double*  fp0  = ss0 + 450 ;
   double*  pf0  = ss0 + 570 ;
   double*  pf1  = ss0 + 600 ;
   double*  pg0  = ss0 + 630 ;
   double*  fd0  = ss0 + 675 ;
   double*  df0  = ss0 + 855 ;
   double*  df1  = ss0 + 915 ;
   double*  ff0  = ss0 + 975 ;
   double*  dg0  = ss0 + 1175 ;
   double*  fg0  = ss0 + 1265 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp0  = ss0 + 135 ;
   double*  sp1  = ss0 + 138 ;
   double*  sd0  = ss0 + 150 ;
   double*  sd1  = ss0 + 156 ;
   double*  pp0  = ss0 + 174 ;
   double*  pp1  = ss0 + 183 ;
   double*  sf0  = ss0 + 219 ;
   double*  sf1  = ss0 + 229 ;
   double*  pd0  = ss0 + 249 ;
   double*  pd1  = ss0 + 267 ;
   double*  dp0  = ss0 + 321 ;
   double*  dp1  = ss0 + 339 ;
   double*  sg0  = ss0 + 411 ;
   double*  sg1  = ss0 + 426 ;
   double*  fp0  = ss0 + 441 ;
   double*  dd0  = ss0 + 591 ;
   double*  dd1  = ss0 + 627 ;
   double*  pf0  = ss0 + 735 ;
   double*  pf1  = ss0 + 765 ;
   double*  sh0  = ss0 + 825 ;
   double*  pg0  = ss0 + 846 ;
   double*  pg1  = ss0 + 891 ;
   double*  df0  = ss0 + 936 ;
   double*  df1  = ss0 + 996 ;
   double*  fd0  = ss0 + 1116 ;
   double*  ph0  = ss0 + 1356 ;
   double*  dg0  = ss0 + 1419 ;
   double*  dg1  = ss0 + 1509 ;
   double*  ff0  = ss0 + 1599 ;
   double*  dh0  = ss0 + 1899 ;
   double*  fg0  = ss0 + 2025 ;
   double*  fh0  = ss0 + 2325 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp0  = ss0 + 155 ;
   double*  sp1  = ss0 + 158 ;
   double*  pp0  = ss0 + 173 ;
   double*  pp1  = ss0 + 182 ;
   double*  sd0  = ss0 + 227 ;
   double*  sd1  = ss0 + 233 ;
   double*  dp0  = ss0 + 257 ;
   double*  dp1  = ss0 + 275 ;
   double*  sf0  = ss0 + 365 ;
   double*  sf1  = ss0 + 375 ;
   double*  pd0  = ss0 + 405 ;
   double*  pd1  = ss0 + 423 ;
   double*  pf0  = ss0 + 495 ;
   double*  pf1  = ss0 + 525 ;
   double*  sg0  = ss0 + 615 ;
   double*  sg1  = ss0 + 630 ;
   double*  dd0  = ss0 + 660 ;
   double*  dd1  = ss0 + 696 ;
   double*  fp0  = ss0 + 840 ;
   double*  df0  = ss0 + 1020 ;
   double*  df1  = ss0 + 1080 ;
   double*  pg0  = ss0 + 1260 ;
   double*  pg1  = ss0 + 1305 ;
   double*  fd0  = ss0 + 1395 ;
   double*  sh0  = ss0 + 1695 ;
   double*  sh1  = ss0 + 1716 ;
   double*  si0  = ss0 + 1737 ;
   double*  ff0  = ss0 + 1765 ;
   double*  ph0  = ss0 + 2165 ;
   double*  ph1  = ss0 + 2228 ;
   double*  dg0  = ss0 + 2291 ;
   double*  dg1  = ss0 + 2381 ;
   double*  fg0  = ss0 + 2561 ;
   double*  dh0  = ss0 + 3011 ;
   double*  dh1  = ss0 + 3137 ;
   double*  pi0  = ss0 + 3263 ;
   double*  fh0  = ss0 + 3347 ;
   double*  di0  = ss0 + 3767 ;
   double*  fi0  = ss0 + 3935 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double*  ps0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( ps0 , ss0 , PA_WP);
}

template< int VTS >
__device__ void pssp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  pp0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void pssd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp1  = ss0 + 13 ;
   double*  pp0  = ss0 + 16 ;
   double*  pd0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pssf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  pp0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void pspp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp1  = ss0 + 13 ;
   double*  pp0  = ss0 + 16 ;
   double*  pd0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pspd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pspf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp1  = ss0 + 21 ;
   double*  sd1  = ss0 + 30 ;
   double*  pp0  = ss0 + 42 ;
   double*  pd0  = ss0 + 78 ;
   double*  sf1  = ss0 + 132 ;
   double*  pf0  = ss0 + 142 ;
   double*  pg0  = ss0 + 202 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp1  = ss0 + 13 ;
   double*  pp0  = ss0 + 16 ;
   double*  pd0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp1  = ss0 + 21 ;
   double*  sd1  = ss0 + 30 ;
   double*  pp0  = ss0 + 42 ;
   double*  pd0  = ss0 + 78 ;
   double*  sf1  = ss0 + 132 ;
   double*  pf0  = ss0 + 142 ;
   double*  pg0  = ss0 + 202 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp1  = ss0 + 25 ;
   double*  pp0  = ss0 + 37 ;
   double*  sd1  = ss0 + 82 ;
   double*  pd0  = ss0 + 100 ;
   double*  sf1  = ss0 + 172 ;
   double*  sg1  = ss0 + 192 ;
   double*  pf0  = ss0 + 207 ;
   double*  pg0  = ss0 + 297 ;
   double*  ph0  = ss0 + 387 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp1  = ss0 + 21 ;
   double*  sd1  = ss0 + 30 ;
   double*  pp0  = ss0 + 42 ;
   double*  pd0  = ss0 + 78 ;
   double*  sf1  = ss0 + 132 ;
   double*  pf0  = ss0 + 142 ;
   double*  pg0  = ss0 + 202 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp1  = ss0 + 25 ;
   double*  pp0  = ss0 + 37 ;
   double*  sd1  = ss0 + 82 ;
   double*  pd0  = ss0 + 100 ;
   double*  sf1  = ss0 + 172 ;
   double*  sg1  = ss0 + 192 ;
   double*  pf0  = ss0 + 207 ;
   double*  pg0  = ss0 + 297 ;
   double*  ph0  = ss0 + 387 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  sp1  = ss0 + 29 ;
   double*  pp0  = ss0 + 44 ;
   double*  sd1  = ss0 + 98 ;
   double*  sf1  = ss0 + 122 ;
   double*  pd0  = ss0 + 152 ;
   double*  pf0  = ss0 + 242 ;
   double*  sg1  = ss0 + 362 ;
   double*  pg0  = ss0 + 392 ;
   double*  sh1  = ss0 + 527 ;
   double*  ph0  = ss0 + 548 ;
   double*  pi0  = ss0 + 674 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 3 ;
   double*  ds0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( ds0 , ps0 , ss0 , PA_WP, zab );
}

template< int VTS >
__device__ void ppsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  pp0  = ss0 + 25 ;
   double*  dp0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void ppsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp1  = ss0 + 35 ;
   double*  pp0  = ss0 + 38 ;
   double*  pp1  = ss0 + 47 ;
   double*  pd0  = ss0 + 56 ;
   double*  dp0  = ss0 + 74 ;
   double*  dd0  = ss0 + 110 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  sd1  = ss0 + 51 ;
   double*  pp0  = ss0 + 57 ;
   double*  pp1  = ss0 + 66 ;
   double*  pd0  = ss0 + 84 ;
   double*  pd1  = ss0 + 102 ;
   double*  dp0  = ss0 + 120 ;
   double*  pf0  = ss0 + 174 ;
   double*  dd0  = ss0 + 204 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  pp0  = ss0 + 25 ;
   double*  dp0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void pppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp1  = ss0 + 35 ;
   double*  pp0  = ss0 + 38 ;
   double*  pp1  = ss0 + 47 ;
   double*  pd0  = ss0 + 56 ;
   double*  dp0  = ss0 + 74 ;
   double*  dd0  = ss0 + 110 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  pp0  = ss0 + 51 ;
   double*  pp1  = ss0 + 60 ;
   double*  sd1  = ss0 + 78 ;
   double*  dp0  = ss0 + 84 ;
   double*  pd0  = ss0 + 138 ;
   double*  pd1  = ss0 + 156 ;
   double*  dd0  = ss0 + 174 ;
   double*  pf0  = ss0 + 246 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp1  = ss0 + 55 ;
   double*  sd1  = ss0 + 64 ;
   double*  pp0  = ss0 + 76 ;
   double*  pp1  = ss0 + 85 ;
   double*  sf1  = ss0 + 112 ;
   double*  pd0  = ss0 + 122 ;
   double*  pd1  = ss0 + 140 ;
   double*  dp0  = ss0 + 176 ;
   double*  pf0  = ss0 + 248 ;
   double*  pf1  = ss0 + 278 ;
   double*  dd0  = ss0 + 308 ;
   double*  pg0  = ss0 + 416 ;
   double*  df0  = ss0 + 461 ;
   double*  dg0  = ss0 + 581 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp1  = ss0 + 35 ;
   double*  pp0  = ss0 + 38 ;
   double*  pp1  = ss0 + 47 ;
   double*  pd0  = ss0 + 56 ;
   double*  dp0  = ss0 + 74 ;
   double*  dd0  = ss0 + 110 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  pp0  = ss0 + 51 ;
   double*  pp1  = ss0 + 60 ;
   double*  sd1  = ss0 + 78 ;
   double*  pd0  = ss0 + 84 ;
   double*  pd1  = ss0 + 102 ;
   double*  dp0  = ss0 + 120 ;
   double*  pf0  = ss0 + 174 ;
   double*  dd0  = ss0 + 204 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp1  = ss0 + 55 ;
   double*  sd1  = ss0 + 64 ;
   double*  pp0  = ss0 + 76 ;
   double*  pp1  = ss0 + 85 ;
   double*  sf1  = ss0 + 112 ;
   double*  pd0  = ss0 + 122 ;
   double*  pd1  = ss0 + 140 ;
   double*  dp0  = ss0 + 176 ;
   double*  pf0  = ss0 + 248 ;
   double*  pf1  = ss0 + 278 ;
   double*  dd0  = ss0 + 308 ;
   double*  pg0  = ss0 + 416 ;
   double*  df0  = ss0 + 461 ;
   double*  dg0  = ss0 + 581 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp1  = ss0 + 65 ;
   double*  pp0  = ss0 + 77 ;
   double*  pp1  = ss0 + 86 ;
   double*  sd1  = ss0 + 122 ;
   double*  dp0  = ss0 + 140 ;
   double*  pd0  = ss0 + 230 ;
   double*  pd1  = ss0 + 248 ;
   double*  sf1  = ss0 + 302 ;
   double*  sg1  = ss0 + 322 ;
   double*  dd0  = ss0 + 337 ;
   double*  pf0  = ss0 + 481 ;
   double*  pf1  = ss0 + 511 ;
   double*  df0  = ss0 + 571 ;
   double*  pg0  = ss0 + 751 ;
   double*  pg1  = ss0 + 796 ;
   double*  dg0  = ss0 + 841 ;
   double*  ph0  = ss0 + 1021 ;
   double*  dh0  = ss0 + 1084 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  pp0  = ss0 + 51 ;
   double*  pp1  = ss0 + 60 ;
   double*  sd1  = ss0 + 78 ;
   double*  dp0  = ss0 + 84 ;
   double*  pd0  = ss0 + 138 ;
   double*  pd1  = ss0 + 156 ;
   double*  pf0  = ss0 + 174 ;
   double*  dd0  = ss0 + 204 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp1  = ss0 + 55 ;
   double*  sd1  = ss0 + 64 ;
   double*  pp0  = ss0 + 76 ;
   double*  pp1  = ss0 + 85 ;
   double*  sf1  = ss0 + 112 ;
   double*  pd0  = ss0 + 122 ;
   double*  pd1  = ss0 + 140 ;
   double*  dp0  = ss0 + 176 ;
   double*  pf0  = ss0 + 248 ;
   double*  pf1  = ss0 + 278 ;
   double*  dd0  = ss0 + 308 ;
   double*  pg0  = ss0 + 416 ;
   double*  df0  = ss0 + 461 ;
   double*  dg0  = ss0 + 581 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp1  = ss0 + 65 ;
   double*  pp0  = ss0 + 77 ;
   double*  pp1  = ss0 + 86 ;
   double*  sd1  = ss0 + 122 ;
   double*  dp0  = ss0 + 140 ;
   double*  pd0  = ss0 + 230 ;
   double*  pd1  = ss0 + 248 ;
   double*  sf1  = ss0 + 302 ;
   double*  dd0  = ss0 + 322 ;
   double*  pf0  = ss0 + 466 ;
   double*  pf1  = ss0 + 496 ;
   double*  sg1  = ss0 + 556 ;
   double*  df0  = ss0 + 571 ;
   double*  pg0  = ss0 + 751 ;
   double*  pg1  = ss0 + 796 ;
   double*  dg0  = ss0 + 841 ;
   double*  ph0  = ss0 + 1021 ;
   double*  dh0  = ss0 + 1084 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  sp1  = ss0 + 75 ;
   double*  sd1  = ss0 + 90 ;
   double*  pp0  = ss0 + 114 ;
   double*  pp1  = ss0 + 123 ;
   double*  sf1  = ss0 + 168 ;
   double*  dp0  = ss0 + 198 ;
   double*  pd0  = ss0 + 306 ;
   double*  pd1  = ss0 + 324 ;
   double*  sg1  = ss0 + 396 ;
   double*  dd0  = ss0 + 426 ;
   double*  pf0  = ss0 + 606 ;
   double*  pf1  = ss0 + 636 ;
   double*  pg0  = ss0 + 726 ;
   double*  pg1  = ss0 + 771 ;
   double*  sh1  = ss0 + 861 ;
   double*  df0  = ss0 + 882 ;
   double*  ph0  = ss0 + 1122 ;
   double*  ph1  = ss0 + 1185 ;
   double*  dg0  = ss0 + 1248 ;
   double*  pi0  = ss0 + 1518 ;
   double*  dh0  = ss0 + 1602 ;
   double*  di0  = ss0 + 1854 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void pdsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  pp0  = ss0 + 55 ;
   double*  dp0  = ss0 + 64 ;
   double*  fp0  = ss0 + 82 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void pdsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp1  = ss0 + 75 ;
   double*  pp0  = ss0 + 78 ;
   double*  pp1  = ss0 + 87 ;
   double*  pd0  = ss0 + 96 ;
   double*  dp0  = ss0 + 114 ;
   double*  dp1  = ss0 + 132 ;
   double*  fp0  = ss0 + 150 ;
   double*  dd0  = ss0 + 210 ;
   double*  fd0  = ss0 + 246 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  pp0  = ss0 + 55 ;
   double*  dp0  = ss0 + 64 ;
   double*  fp0  = ss0 + 82 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void pdpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp1  = ss0 + 75 ;
   double*  pp0  = ss0 + 78 ;
   double*  pp1  = ss0 + 87 ;
   double*  pd0  = ss0 + 96 ;
   double*  dp0  = ss0 + 114 ;
   double*  dp1  = ss0 + 132 ;
   double*  dd0  = ss0 + 150 ;
   double*  fp0  = ss0 + 186 ;
   double*  fd0  = ss0 + 246 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp1  = ss0 + 115 ;
   double*  pp0  = ss0 + 124 ;
   double*  pp1  = ss0 + 133 ;
   double*  sd1  = ss0 + 160 ;
   double*  sf1  = ss0 + 172 ;
   double*  dp0  = ss0 + 182 ;
   double*  dp1  = ss0 + 200 ;
   double*  pd0  = ss0 + 254 ;
   double*  pd1  = ss0 + 272 ;
   double*  fp0  = ss0 + 308 ;
   double*  dd0  = ss0 + 428 ;
   double*  dd1  = ss0 + 464 ;
   double*  pf0  = ss0 + 536 ;
   double*  pf1  = ss0 + 566 ;
   double*  pg0  = ss0 + 596 ;
   double*  fd0  = ss0 + 641 ;
   double*  df0  = ss0 + 821 ;
   double*  df1  = ss0 + 881 ;
   double*  ff0  = ss0 + 941 ;
   double*  dg0  = ss0 + 1141 ;
   double*  fg0  = ss0 + 1231 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp1  = ss0 + 75 ;
   double*  pp0  = ss0 + 78 ;
   double*  pp1  = ss0 + 87 ;
   double*  pd0  = ss0 + 96 ;
   double*  dp0  = ss0 + 114 ;
   double*  dp1  = ss0 + 132 ;
   double*  dd0  = ss0 + 150 ;
   double*  fp0  = ss0 + 186 ;
   double*  fd0  = ss0 + 246 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp1  = ss0 + 115 ;
   double*  pp0  = ss0 + 124 ;
   double*  pp1  = ss0 + 133 ;
   double*  sd1  = ss0 + 160 ;
   double*  sf1  = ss0 + 172 ;
   double*  dp0  = ss0 + 182 ;
   double*  dp1  = ss0 + 200 ;
   double*  pd0  = ss0 + 254 ;
   double*  pd1  = ss0 + 272 ;
   double*  fp0  = ss0 + 308 ;
   double*  dd0  = ss0 + 428 ;
   double*  dd1  = ss0 + 464 ;
   double*  pf0  = ss0 + 536 ;
   double*  pf1  = ss0 + 566 ;
   double*  pg0  = ss0 + 596 ;
   double*  fd0  = ss0 + 641 ;
   double*  df0  = ss0 + 821 ;
   double*  df1  = ss0 + 881 ;
   double*  ff0  = ss0 + 941 ;
   double*  dg0  = ss0 + 1141 ;
   double*  fg0  = ss0 + 1231 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp1  = ss0 + 135 ;
   double*  sd1  = ss0 + 147 ;
   double*  pp0  = ss0 + 165 ;
   double*  pp1  = ss0 + 174 ;
   double*  sf1  = ss0 + 210 ;
   double*  dp0  = ss0 + 230 ;
   double*  dp1  = ss0 + 248 ;
   double*  pd0  = ss0 + 320 ;
   double*  pd1  = ss0 + 338 ;
   double*  sg1  = ss0 + 392 ;
   double*  fp0  = ss0 + 407 ;
   double*  dd0  = ss0 + 557 ;
   double*  dd1  = ss0 + 593 ;
   double*  pf0  = ss0 + 701 ;
   double*  pf1  = ss0 + 731 ;
   double*  pg0  = ss0 + 791 ;
   double*  pg1  = ss0 + 836 ;
   double*  df0  = ss0 + 881 ;
   double*  df1  = ss0 + 941 ;
   double*  fd0  = ss0 + 1061 ;
   double*  ph0  = ss0 + 1301 ;
   double*  dg0  = ss0 + 1364 ;
   double*  dg1  = ss0 + 1454 ;
   double*  ff0  = ss0 + 1544 ;
   double*  dh0  = ss0 + 1844 ;
   double*  fg0  = ss0 + 1970 ;
   double*  fh0  = ss0 + 2270 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp1  = ss0 + 115 ;
   double*  pp0  = ss0 + 124 ;
   double*  pp1  = ss0 + 133 ;
   double*  sd1  = ss0 + 160 ;
   double*  sf1  = ss0 + 172 ;
   double*  dp0  = ss0 + 182 ;
   double*  dp1  = ss0 + 200 ;
   double*  pd0  = ss0 + 254 ;
   double*  pd1  = ss0 + 272 ;
   double*  dd0  = ss0 + 308 ;
   double*  dd1  = ss0 + 344 ;
   double*  fp0  = ss0 + 416 ;
   double*  pf0  = ss0 + 536 ;
   double*  pf1  = ss0 + 566 ;
   double*  pg0  = ss0 + 596 ;
   double*  fd0  = ss0 + 641 ;
   double*  df0  = ss0 + 821 ;
   double*  df1  = ss0 + 881 ;
   double*  ff0  = ss0 + 941 ;
   double*  dg0  = ss0 + 1141 ;
   double*  fg0  = ss0 + 1231 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp1  = ss0 + 135 ;
   double*  pp0  = ss0 + 147 ;
   double*  pp1  = ss0 + 156 ;
   double*  sd1  = ss0 + 192 ;
   double*  sf1  = ss0 + 210 ;
   double*  pd0  = ss0 + 230 ;
   double*  pd1  = ss0 + 248 ;
   double*  dp0  = ss0 + 302 ;
   double*  dp1  = ss0 + 320 ;
   double*  sg1  = ss0 + 392 ;
   double*  fp0  = ss0 + 407 ;
   double*  dd0  = ss0 + 557 ;
   double*  dd1  = ss0 + 593 ;
   double*  pf0  = ss0 + 701 ;
   double*  pf1  = ss0 + 731 ;
   double*  pg0  = ss0 + 791 ;
   double*  pg1  = ss0 + 836 ;
   double*  df0  = ss0 + 881 ;
   double*  df1  = ss0 + 941 ;
   double*  fd0  = ss0 + 1061 ;
   double*  ph0  = ss0 + 1301 ;
   double*  dg0  = ss0 + 1364 ;
   double*  dg1  = ss0 + 1454 ;
   double*  ff0  = ss0 + 1544 ;
   double*  dh0  = ss0 + 1844 ;
   double*  fg0  = ss0 + 1970 ;
   double*  fh0  = ss0 + 2270 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp1  = ss0 + 155 ;
   double*  sd1  = ss0 + 170 ;
   double*  pp0  = ss0 + 194 ;
   double*  pp1  = ss0 + 203 ;
   double*  sf1  = ss0 + 248 ;
   double*  dp0  = ss0 + 278 ;
   double*  dp1  = ss0 + 296 ;
   double*  pd0  = ss0 + 386 ;
   double*  pd1  = ss0 + 404 ;
   double*  fp0  = ss0 + 476 ;
   double*  sg1  = ss0 + 656 ;
   double*  dd0  = ss0 + 686 ;
   double*  dd1  = ss0 + 722 ;
   double*  pf0  = ss0 + 866 ;
   double*  pf1  = ss0 + 896 ;
   double*  sh1  = ss0 + 986 ;
   double*  pg0  = ss0 + 1007 ;
   double*  pg1  = ss0 + 1052 ;
   double*  fd0  = ss0 + 1142 ;
   double*  df0  = ss0 + 1442 ;
   double*  df1  = ss0 + 1502 ;
   double*  dg0  = ss0 + 1682 ;
   double*  dg1  = ss0 + 1772 ;
   double*  ph0  = ss0 + 1952 ;
   double*  ph1  = ss0 + 2015 ;
   double*  ff0  = ss0 + 2078 ;
   double*  pi0  = ss0 + 2478 ;
   double*  dh0  = ss0 + 2562 ;
   double*  dh1  = ss0 + 2688 ;
   double*  fg0  = ss0 + 2814 ;
   double*  di0  = ss0 + 3264 ;
   double*  fh0  = ss0 + 3432 ;
   double*  fi0  = ss0 + 3852 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  fs0  = ss0 + 35 ;
   double*  gs0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( gs0 , fs0 , ds0 , PA_WP, zab );
}

template< int VTS >
__device__ void pfsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  pp0  = ss0 + 105 ;
   double*  dp0  = ss0 + 114 ;
   double*  fp0  = ss0 + 132 ;
   double*  gp0  = ss0 + 162 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void pfsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  sp1  = ss0 + 140 ;
   double*  pp0  = ss0 + 143 ;
   double*  pp1  = ss0 + 152 ;
   double*  pd0  = ss0 + 161 ;
   double*  dp0  = ss0 + 179 ;
   double*  dp1  = ss0 + 197 ;
   double*  dd0  = ss0 + 215 ;
   double*  fp0  = ss0 + 251 ;
   double*  fp1  = ss0 + 281 ;
   double*  fd0  = ss0 + 311 ;
   double*  gp0  = ss0 + 371 ;
   double*  gd0  = ss0 + 461 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  sd1  = ss0 + 181 ;
   double*  pp0  = ss0 + 187 ;
   double*  pp1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  pd0  = ss0 + 268 ;
   double*  pd1  = ss0 + 286 ;
   double*  pf0  = ss0 + 304 ;
   double*  fp0  = ss0 + 334 ;
   double*  fp1  = ss0 + 364 ;
   double*  dd0  = ss0 + 424 ;
   double*  dd1  = ss0 + 460 ;
   double*  df0  = ss0 + 496 ;
   double*  gp0  = ss0 + 556 ;
   double*  fd0  = ss0 + 691 ;
   double*  fd1  = ss0 + 751 ;
   double*  gd0  = ss0 + 811 ;
   double*  ff0  = ss0 + 991 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  pp0  = ss0 + 105 ;
   double*  dp0  = ss0 + 114 ;
   double*  fp0  = ss0 + 132 ;
   double*  gp0  = ss0 + 162 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void pfpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  sp1  = ss0 + 140 ;
   double*  pp0  = ss0 + 143 ;
   double*  pp1  = ss0 + 152 ;
   double*  pd0  = ss0 + 161 ;
   double*  dp0  = ss0 + 179 ;
   double*  dp1  = ss0 + 197 ;
   double*  dd0  = ss0 + 215 ;
   double*  fp0  = ss0 + 251 ;
   double*  fp1  = ss0 + 281 ;
   double*  gp0  = ss0 + 311 ;
   double*  fd0  = ss0 + 401 ;
   double*  gd0  = ss0 + 461 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  sd1  = ss0 + 181 ;
   double*  pp0  = ss0 + 187 ;
   double*  pp1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  pd0  = ss0 + 268 ;
   double*  pd1  = ss0 + 286 ;
   double*  pf0  = ss0 + 304 ;
   double*  fp0  = ss0 + 334 ;
   double*  fp1  = ss0 + 364 ;
   double*  dd0  = ss0 + 424 ;
   double*  dd1  = ss0 + 460 ;
   double*  gp0  = ss0 + 496 ;
   double*  fd0  = ss0 + 631 ;
   double*  fd1  = ss0 + 691 ;
   double*  df0  = ss0 + 751 ;
   double*  gd0  = ss0 + 811 ;
   double*  ff0  = ss0 + 991 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp1  = ss0 + 210 ;
   double*  sd1  = ss0 + 219 ;
   double*  pp0  = ss0 + 231 ;
   double*  pp1  = ss0 + 240 ;
   double*  sf1  = ss0 + 267 ;
   double*  dp0  = ss0 + 277 ;
   double*  dp1  = ss0 + 295 ;
   double*  pd0  = ss0 + 349 ;
   double*  pd1  = ss0 + 367 ;
   double*  pf0  = ss0 + 403 ;
   double*  pf1  = ss0 + 433 ;
   double*  fp0  = ss0 + 463 ;
   double*  fp1  = ss0 + 493 ;
   double*  dd0  = ss0 + 583 ;
   double*  dd1  = ss0 + 619 ;
   double*  pg0  = ss0 + 691 ;
   double*  df0  = ss0 + 736 ;
   double*  df1  = ss0 + 796 ;
   double*  gp0  = ss0 + 856 ;
   double*  fd0  = ss0 + 1036 ;
   double*  fd1  = ss0 + 1096 ;
   double*  dg0  = ss0 + 1216 ;
   double*  gd0  = ss0 + 1306 ;
   double*  ff0  = ss0 + 1576 ;
   double*  ff1  = ss0 + 1676 ;
   double*  gf0  = ss0 + 1776 ;
   double*  fg0  = ss0 + 2076 ;
   double*  gg0  = ss0 + 2226 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  sp1  = ss0 + 140 ;
   double*  pp0  = ss0 + 143 ;
   double*  pp1  = ss0 + 152 ;
   double*  dp0  = ss0 + 161 ;
   double*  dp1  = ss0 + 179 ;
   double*  pd0  = ss0 + 197 ;
   double*  dd0  = ss0 + 215 ;
   double*  fp0  = ss0 + 251 ;
   double*  fp1  = ss0 + 281 ;
   double*  gp0  = ss0 + 311 ;
   double*  fd0  = ss0 + 401 ;
   double*  gd0  = ss0 + 461 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  sd1  = ss0 + 181 ;
   double*  pp0  = ss0 + 187 ;
   double*  pp1  = ss0 + 196 ;
   double*  pd0  = ss0 + 214 ;
   double*  pd1  = ss0 + 232 ;
   double*  dp0  = ss0 + 250 ;
   double*  dp1  = ss0 + 268 ;
   double*  pf0  = ss0 + 304 ;
   double*  dd0  = ss0 + 334 ;
   double*  dd1  = ss0 + 370 ;
   double*  fp0  = ss0 + 406 ;
   double*  fp1  = ss0 + 436 ;
   double*  df0  = ss0 + 496 ;
   double*  fd0  = ss0 + 556 ;
   double*  fd1  = ss0 + 616 ;
   double*  gp0  = ss0 + 676 ;
   double*  ff0  = ss0 + 811 ;
   double*  gd0  = ss0 + 911 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp1  = ss0 + 210 ;
   double*  sd1  = ss0 + 219 ;
   double*  pp0  = ss0 + 231 ;
   double*  pp1  = ss0 + 240 ;
   double*  sf1  = ss0 + 267 ;
   double*  pd0  = ss0 + 277 ;
   double*  pd1  = ss0 + 295 ;
   double*  dp0  = ss0 + 331 ;
   double*  dp1  = ss0 + 349 ;
   double*  pf0  = ss0 + 403 ;
   double*  pf1  = ss0 + 433 ;
   double*  fp0  = ss0 + 463 ;
   double*  fp1  = ss0 + 493 ;
   double*  dd0  = ss0 + 583 ;
   double*  dd1  = ss0 + 619 ;
   double*  pg0  = ss0 + 691 ;
   double*  gp0  = ss0 + 736 ;
   double*  fd0  = ss0 + 916 ;
   double*  fd1  = ss0 + 976 ;
   double*  df0  = ss0 + 1096 ;
   double*  df1  = ss0 + 1156 ;
   double*  dg0  = ss0 + 1216 ;
   double*  gd0  = ss0 + 1306 ;
   double*  ff0  = ss0 + 1576 ;
   double*  ff1  = ss0 + 1676 ;
   double*  gf0  = ss0 + 1776 ;
   double*  fg0  = ss0 + 2076 ;
   double*  gg0  = ss0 + 2226 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp1  = ss0 + 245 ;
   double*  pp0  = ss0 + 257 ;
   double*  pp1  = ss0 + 266 ;
   double*  sd1  = ss0 + 302 ;
   double*  dp0  = ss0 + 320 ;
   double*  dp1  = ss0 + 338 ;
   double*  pd0  = ss0 + 410 ;
   double*  pd1  = ss0 + 428 ;
   double*  sf1  = ss0 + 482 ;
   double*  pf0  = ss0 + 502 ;
   double*  pf1  = ss0 + 532 ;
   double*  sg1  = ss0 + 592 ;
   double*  dd0  = ss0 + 607 ;
   double*  dd1  = ss0 + 643 ;
   double*  fp0  = ss0 + 751 ;
   double*  fp1  = ss0 + 781 ;
   double*  df0  = ss0 + 901 ;
   double*  df1  = ss0 + 961 ;
   double*  fd0  = ss0 + 1081 ;
   double*  fd1  = ss0 + 1141 ;
   double*  gp0  = ss0 + 1321 ;
   double*  pg0  = ss0 + 1546 ;
   double*  pg1  = ss0 + 1591 ;
   double*  gd0  = ss0 + 1636 ;
   double*  dg0  = ss0 + 1996 ;
   double*  dg1  = ss0 + 2086 ;
   double*  ff0  = ss0 + 2176 ;
   double*  ff1  = ss0 + 2276 ;
   double*  ph0  = ss0 + 2476 ;
   double*  gf0  = ss0 + 2539 ;
   double*  fg0  = ss0 + 2989 ;
   double*  fg1  = ss0 + 3139 ;
   double*  dh0  = ss0 + 3289 ;
   double*  gg0  = ss0 + 3415 ;
   double*  fh0  = ss0 + 3865 ;
   double*  gh0  = ss0 + 4075 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  pp0  = ss0 + 181 ;
   double*  pp1  = ss0 + 190 ;
   double*  sd1  = ss0 + 208 ;
   double*  pd0  = ss0 + 214 ;
   double*  pd1  = ss0 + 232 ;
   double*  dp0  = ss0 + 250 ;
   double*  dp1  = ss0 + 268 ;
   double*  dd0  = ss0 + 304 ;
   double*  dd1  = ss0 + 340 ;
   double*  fp0  = ss0 + 376 ;
   double*  fp1  = ss0 + 406 ;
   double*  pf0  = ss0 + 466 ;
   double*  df0  = ss0 + 496 ;
   double*  fd0  = ss0 + 556 ;
   double*  fd1  = ss0 + 616 ;
   double*  gp0  = ss0 + 676 ;
   double*  ff0  = ss0 + 811 ;
   double*  gd0  = ss0 + 911 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp1  = ss0 + 210 ;
   double*  sd1  = ss0 + 219 ;
   double*  pp0  = ss0 + 231 ;
   double*  pp1  = ss0 + 240 ;
   double*  sf1  = ss0 + 267 ;
   double*  pd0  = ss0 + 277 ;
   double*  pd1  = ss0 + 295 ;
   double*  dp0  = ss0 + 331 ;
   double*  dp1  = ss0 + 349 ;
   double*  pf0  = ss0 + 403 ;
   double*  pf1  = ss0 + 433 ;
   double*  fp0  = ss0 + 463 ;
   double*  fp1  = ss0 + 493 ;
   double*  dd0  = ss0 + 583 ;
   double*  dd1  = ss0 + 619 ;
   double*  pg0  = ss0 + 691 ;
   double*  fd0  = ss0 + 736 ;
   double*  fd1  = ss0 + 796 ;
   double*  gp0  = ss0 + 916 ;
   double*  df0  = ss0 + 1096 ;
   double*  df1  = ss0 + 1156 ;
   double*  dg0  = ss0 + 1216 ;
   double*  gd0  = ss0 + 1306 ;
   double*  ff0  = ss0 + 1576 ;
   double*  ff1  = ss0 + 1676 ;
   double*  gf0  = ss0 + 1776 ;
   double*  fg0  = ss0 + 2076 ;
   double*  gg0  = ss0 + 2226 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp1  = ss0 + 245 ;
   double*  sd1  = ss0 + 257 ;
   double*  pp0  = ss0 + 275 ;
   double*  pp1  = ss0 + 284 ;
   double*  sf1  = ss0 + 320 ;
   double*  dp0  = ss0 + 340 ;
   double*  dp1  = ss0 + 358 ;
   double*  pd0  = ss0 + 430 ;
   double*  pd1  = ss0 + 448 ;
   double*  fp0  = ss0 + 502 ;
   double*  fp1  = ss0 + 532 ;
   double*  sg1  = ss0 + 652 ;
   double*  dd0  = ss0 + 667 ;
   double*  dd1  = ss0 + 703 ;
   double*  pf0  = ss0 + 811 ;
   double*  pf1  = ss0 + 841 ;
   double*  pg0  = ss0 + 901 ;
   double*  pg1  = ss0 + 946 ;
   double*  fd0  = ss0 + 991 ;
   double*  fd1  = ss0 + 1051 ;
   double*  df0  = ss0 + 1231 ;
   double*  df1  = ss0 + 1291 ;
   double*  gp0  = ss0 + 1411 ;
   double*  ph0  = ss0 + 1636 ;
   double*  dg0  = ss0 + 1699 ;
   double*  dg1  = ss0 + 1789 ;
   double*  ff0  = ss0 + 1879 ;
   double*  ff1  = ss0 + 1979 ;
   double*  gd0  = ss0 + 2179 ;
   double*  dh0  = ss0 + 2539 ;
   double*  fg0  = ss0 + 2665 ;
   double*  fg1  = ss0 + 2815 ;
   double*  gf0  = ss0 + 2965 ;
   double*  fh0  = ss0 + 3415 ;
   double*  gg0  = ss0 + 3625 ;
   double*  gh0  = ss0 + 4075 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  sp1  = ss0 + 280 ;
   double*  sd1  = ss0 + 295 ;
   double*  pp0  = ss0 + 319 ;
   double*  pp1  = ss0 + 328 ;
   double*  dp0  = ss0 + 373 ;
   double*  dp1  = ss0 + 391 ;
   double*  pd0  = ss0 + 481 ;
   double*  pd1  = ss0 + 499 ;
   double*  sf1  = ss0 + 571 ;
   double*  sg1  = ss0 + 601 ;
   double*  pf0  = ss0 + 631 ;
   double*  pf1  = ss0 + 661 ;
   double*  fp0  = ss0 + 751 ;
   double*  fp1  = ss0 + 781 ;
   double*  dd0  = ss0 + 931 ;
   double*  dd1  = ss0 + 967 ;
   double*  sh1  = ss0 + 1111 ;
   double*  df0  = ss0 + 1132 ;
   double*  df1  = ss0 + 1192 ;
   double*  fd0  = ss0 + 1372 ;
   double*  fd1  = ss0 + 1432 ;
   double*  pg0  = ss0 + 1672 ;
   double*  pg1  = ss0 + 1717 ;
   double*  gp0  = ss0 + 1807 ;
   double*  dg0  = ss0 + 2077 ;
   double*  dg1  = ss0 + 2167 ;
   double*  ff0  = ss0 + 2347 ;
   double*  ff1  = ss0 + 2447 ;
   double*  ph0  = ss0 + 2747 ;
   double*  ph1  = ss0 + 2810 ;
   double*  gd0  = ss0 + 2873 ;
   double*  pi0  = ss0 + 3323 ;
   double*  gf0  = ss0 + 3407 ;
   double*  dh0  = ss0 + 4007 ;
   double*  dh1  = ss0 + 4133 ;
   double*  fg0  = ss0 + 4259 ;
   double*  fg1  = ss0 + 4409 ;
   double*  gg0  = ss0 + 4709 ;
   double*  fh0  = ss0 + 5384 ;
   double*  fh1  = ss0 + 5594 ;
   double*  di0  = ss0 + 5804 ;
   double*  gh0  = ss0 + 5972 ;
   double*  fi0  = ss0 + 6602 ;
   double*  gi0  = ss0 + 6882 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 3 ;
   double*  ds0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( ds0 , ps0 , ss0 , PA_WP, zab );
}

template< int VTS >
__device__ void dssp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  dp0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void dssd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  pp1  = ss0 + 35 ;
   double*  dp0  = ss0 + 44 ;
   double*  dd0  = ss0 + 80 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dssf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  dp0  = ss0 + 66 ;
   double*  pd1  = ss0 + 120 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  dp0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void dspp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  pp1  = ss0 + 35 ;
   double*  dp0  = ss0 + 44 ;
   double*  dd0  = ss0 + 80 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dspd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  pd1  = ss0 + 66 ;
   double*  dp0  = ss0 + 84 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dspf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp2  = ss0 + 55 ;
   double*  sd2  = ss0 + 61 ;
   double*  pp1  = ss0 + 67 ;
   double*  pd1  = ss0 + 94 ;
   double*  dp0  = ss0 + 130 ;
   double*  pf1  = ss0 + 202 ;
   double*  dd0  = ss0 + 232 ;
   double*  df0  = ss0 + 340 ;
   double*  dg0  = ss0 + 460 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  pp1  = ss0 + 35 ;
   double*  dp0  = ss0 + 44 ;
   double*  dd0  = ss0 + 80 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  pd1  = ss0 + 66 ;
   double*  dp0  = ss0 + 84 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp2  = ss0 + 55 ;
   double*  sd2  = ss0 + 61 ;
   double*  pp1  = ss0 + 67 ;
   double*  pd1  = ss0 + 94 ;
   double*  dp0  = ss0 + 130 ;
   double*  pf1  = ss0 + 202 ;
   double*  dd0  = ss0 + 232 ;
   double*  df0  = ss0 + 340 ;
   double*  dg0  = ss0 + 460 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp2  = ss0 + 65 ;
   double*  sd2  = ss0 + 74 ;
   double*  pp1  = ss0 + 86 ;
   double*  sf2  = ss0 + 122 ;
   double*  pd1  = ss0 + 132 ;
   double*  dp0  = ss0 + 186 ;
   double*  pf1  = ss0 + 276 ;
   double*  dd0  = ss0 + 336 ;
   double*  pg1  = ss0 + 480 ;
   double*  df0  = ss0 + 525 ;
   double*  dg0  = ss0 + 705 ;
   double*  dh0  = ss0 + 885 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  pd1  = ss0 + 66 ;
   double*  dp0  = ss0 + 84 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp2  = ss0 + 55 ;
   double*  sd2  = ss0 + 61 ;
   double*  pp1  = ss0 + 67 ;
   double*  pd1  = ss0 + 94 ;
   double*  dp0  = ss0 + 130 ;
   double*  pf1  = ss0 + 202 ;
   double*  dd0  = ss0 + 232 ;
   double*  df0  = ss0 + 340 ;
   double*  dg0  = ss0 + 460 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp2  = ss0 + 65 ;
   double*  sd2  = ss0 + 74 ;
   double*  pp1  = ss0 + 86 ;
   double*  sf2  = ss0 + 122 ;
   double*  pd1  = ss0 + 132 ;
   double*  dp0  = ss0 + 186 ;
   double*  pf1  = ss0 + 276 ;
   double*  dd0  = ss0 + 336 ;
   double*  pg1  = ss0 + 480 ;
   double*  df0  = ss0 + 525 ;
   double*  dg0  = ss0 + 705 ;
   double*  dh0  = ss0 + 885 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  sp2  = ss0 + 75 ;
   double*  sd2  = ss0 + 87 ;
   double*  pp1  = ss0 + 105 ;
   double*  sf2  = ss0 + 150 ;
   double*  dp0  = ss0 + 170 ;
   double*  pd1  = ss0 + 278 ;
   double*  sg2  = ss0 + 350 ;
   double*  dd0  = ss0 + 365 ;
   double*  pf1  = ss0 + 545 ;
   double*  pg1  = ss0 + 635 ;
   double*  df0  = ss0 + 725 ;
   double*  ph1  = ss0 + 965 ;
   double*  dg0  = ss0 + 1028 ;
   double*  dh0  = ss0 + 1298 ;
   double*  di0  = ss0 + 1550 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void dpsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  dp0  = ss0 + 55 ;
   double*  fp0  = ss0 + 73 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void dpsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  pp1  = ss0 + 75 ;
   double*  dp0  = ss0 + 84 ;
   double*  dp1  = ss0 + 102 ;
   double*  dd0  = ss0 + 120 ;
   double*  fp0  = ss0 + 156 ;
   double*  fd0  = ss0 + 216 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  dp0  = ss0 + 55 ;
   double*  fp0  = ss0 + 73 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void dppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  pp1  = ss0 + 75 ;
   double*  dp0  = ss0 + 84 ;
   double*  dp1  = ss0 + 102 ;
   double*  dd0  = ss0 + 120 ;
   double*  fp0  = ss0 + 156 ;
   double*  fd0  = ss0 + 216 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp2  = ss0 + 115 ;
   double*  pp1  = ss0 + 121 ;
   double*  sd2  = ss0 + 148 ;
   double*  dp0  = ss0 + 154 ;
   double*  dp1  = ss0 + 172 ;
   double*  pd1  = ss0 + 226 ;
   double*  fp0  = ss0 + 262 ;
   double*  dd0  = ss0 + 382 ;
   double*  dd1  = ss0 + 418 ;
   double*  pf1  = ss0 + 490 ;
   double*  fd0  = ss0 + 520 ;
   double*  df0  = ss0 + 700 ;
   double*  df1  = ss0 + 760 ;
   double*  ff0  = ss0 + 820 ;
   double*  dg0  = ss0 + 1020 ;
   double*  fg0  = ss0 + 1110 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  pp1  = ss0 + 75 ;
   double*  dp0  = ss0 + 84 ;
   double*  dp1  = ss0 + 102 ;
   double*  dd0  = ss0 + 120 ;
   double*  fp0  = ss0 + 156 ;
   double*  fd0  = ss0 + 216 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp2  = ss0 + 115 ;
   double*  sd2  = ss0 + 121 ;
   double*  pp1  = ss0 + 127 ;
   double*  pd1  = ss0 + 154 ;
   double*  dp0  = ss0 + 190 ;
   double*  dp1  = ss0 + 208 ;
   double*  pf1  = ss0 + 262 ;
   double*  dd0  = ss0 + 292 ;
   double*  dd1  = ss0 + 328 ;
   double*  fp0  = ss0 + 400 ;
   double*  df0  = ss0 + 520 ;
   double*  df1  = ss0 + 580 ;
   double*  fd0  = ss0 + 640 ;
   double*  dg0  = ss0 + 820 ;
   double*  ff0  = ss0 + 910 ;
   double*  fg0  = ss0 + 1110 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp2  = ss0 + 135 ;
   double*  pp1  = ss0 + 144 ;
   double*  sd2  = ss0 + 180 ;
   double*  sf2  = ss0 + 192 ;
   double*  dp0  = ss0 + 202 ;
   double*  dp1  = ss0 + 220 ;
   double*  pd1  = ss0 + 292 ;
   double*  fp0  = ss0 + 346 ;
   double*  dd0  = ss0 + 496 ;
   double*  dd1  = ss0 + 532 ;
   double*  pf1  = ss0 + 640 ;
   double*  pg1  = ss0 + 700 ;
   double*  df0  = ss0 + 745 ;
   double*  df1  = ss0 + 805 ;
   double*  fd0  = ss0 + 925 ;
   double*  dg0  = ss0 + 1165 ;
   double*  dg1  = ss0 + 1255 ;
   double*  ff0  = ss0 + 1345 ;
   double*  dh0  = ss0 + 1645 ;
   double*  fg0  = ss0 + 1771 ;
   double*  fh0  = ss0 + 2071 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp2  = ss0 + 115 ;
   double*  sd2  = ss0 + 121 ;
   double*  pp1  = ss0 + 127 ;
   double*  pd1  = ss0 + 154 ;
   double*  dp0  = ss0 + 190 ;
   double*  dp1  = ss0 + 208 ;
   double*  pf1  = ss0 + 262 ;
   double*  fp0  = ss0 + 292 ;
   double*  dd0  = ss0 + 412 ;
   double*  dd1  = ss0 + 448 ;
   double*  df0  = ss0 + 520 ;
   double*  df1  = ss0 + 580 ;
   double*  fd0  = ss0 + 640 ;
   double*  dg0  = ss0 + 820 ;
   double*  ff0  = ss0 + 910 ;
   double*  fg0  = ss0 + 1110 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp2  = ss0 + 135 ;
   double*  pp1  = ss0 + 144 ;
   double*  sd2  = ss0 + 180 ;
   double*  sf2  = ss0 + 192 ;
   double*  pd1  = ss0 + 202 ;
   double*  dp0  = ss0 + 256 ;
   double*  dp1  = ss0 + 274 ;
   double*  fp0  = ss0 + 346 ;
   double*  dd0  = ss0 + 496 ;
   double*  dd1  = ss0 + 532 ;
   double*  pf1  = ss0 + 640 ;
   double*  pg1  = ss0 + 700 ;
   double*  df0  = ss0 + 745 ;
   double*  df1  = ss0 + 805 ;
   double*  fd0  = ss0 + 925 ;
   double*  dg0  = ss0 + 1165 ;
   double*  dg1  = ss0 + 1255 ;
   double*  ff0  = ss0 + 1345 ;
   double*  dh0  = ss0 + 1645 ;
   double*  fg0  = ss0 + 1771 ;
   double*  fh0  = ss0 + 2071 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp2  = ss0 + 155 ;
   double*  sd2  = ss0 + 167 ;
   double*  pp1  = ss0 + 185 ;
   double*  sf2  = ss0 + 230 ;
   double*  dp0  = ss0 + 250 ;
   double*  dp1  = ss0 + 268 ;
   double*  pd1  = ss0 + 358 ;
   double*  fp0  = ss0 + 430 ;
   double*  sg2  = ss0 + 610 ;
   double*  dd0  = ss0 + 625 ;
   double*  dd1  = ss0 + 661 ;
   double*  pf1  = ss0 + 805 ;
   double*  pg1  = ss0 + 895 ;
   double*  fd0  = ss0 + 985 ;
   double*  df0  = ss0 + 1285 ;
   double*  df1  = ss0 + 1345 ;
   double*  dg0  = ss0 + 1525 ;
   double*  dg1  = ss0 + 1615 ;
   double*  ph1  = ss0 + 1795 ;
   double*  ff0  = ss0 + 1858 ;
   double*  dh0  = ss0 + 2258 ;
   double*  dh1  = ss0 + 2384 ;
   double*  fg0  = ss0 + 2510 ;
   double*  di0  = ss0 + 2960 ;
   double*  fh0  = ss0 + 3128 ;
   double*  fi0  = ss0 + 3548 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  fs0  = ss0 + 35 ;
   double*  gs0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( gs0 , fs0 , ds0 , PA_WP, zab );
}

template< int VTS >
__device__ void ddsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  dp0  = ss0 + 105 ;
   double*  fp0  = ss0 + 123 ;
   double*  gp0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ddsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  pp1  = ss0 + 140 ;
   double*  dp0  = ss0 + 149 ;
   double*  dp1  = ss0 + 167 ;
   double*  dd0  = ss0 + 185 ;
   double*  fp0  = ss0 + 221 ;
   double*  fp1  = ss0 + 251 ;
   double*  fd0  = ss0 + 281 ;
   double*  gp0  = ss0 + 341 ;
   double*  gd0  = ss0 + 431 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  dp0  = ss0 + 196 ;
   double*  dp1  = ss0 + 214 ;
   double*  pd1  = ss0 + 250 ;
   double*  fp0  = ss0 + 268 ;
   double*  fp1  = ss0 + 298 ;
   double*  dd0  = ss0 + 358 ;
   double*  dd1  = ss0 + 394 ;
   double*  gp0  = ss0 + 430 ;
   double*  fd0  = ss0 + 565 ;
   double*  fd1  = ss0 + 625 ;
   double*  df0  = ss0 + 685 ;
   double*  gd0  = ss0 + 745 ;
   double*  ff0  = ss0 + 925 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  dp0  = ss0 + 105 ;
   double*  fp0  = ss0 + 123 ;
   double*  gp0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ddpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  pp1  = ss0 + 140 ;
   double*  dp0  = ss0 + 149 ;
   double*  dp1  = ss0 + 167 ;
   double*  dd0  = ss0 + 185 ;
   double*  fp0  = ss0 + 221 ;
   double*  fp1  = ss0 + 251 ;
   double*  gp0  = ss0 + 281 ;
   double*  fd0  = ss0 + 371 ;
   double*  gd0  = ss0 + 431 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  pd1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  dd0  = ss0 + 268 ;
   double*  dd1  = ss0 + 304 ;
   double*  fp0  = ss0 + 340 ;
   double*  fp1  = ss0 + 370 ;
   double*  df0  = ss0 + 430 ;
   double*  fd0  = ss0 + 490 ;
   double*  fd1  = ss0 + 550 ;
   double*  gp0  = ss0 + 610 ;
   double*  ff0  = ss0 + 745 ;
   double*  gd0  = ss0 + 845 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp2  = ss0 + 210 ;
   double*  sd2  = ss0 + 216 ;
   double*  pp1  = ss0 + 222 ;
   double*  dp0  = ss0 + 249 ;
   double*  dp1  = ss0 + 267 ;
   double*  pd1  = ss0 + 321 ;
   double*  pf1  = ss0 + 357 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 507 ;
   double*  dd1  = ss0 + 543 ;
   double*  df0  = ss0 + 615 ;
   double*  df1  = ss0 + 675 ;
   double*  gp0  = ss0 + 735 ;
   double*  fd0  = ss0 + 915 ;
   double*  fd1  = ss0 + 975 ;
   double*  dg0  = ss0 + 1095 ;
   double*  gd0  = ss0 + 1185 ;
   double*  ff0  = ss0 + 1455 ;
   double*  ff1  = ss0 + 1555 ;
   double*  gf0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 1955 ;
   double*  gg0  = ss0 + 2105 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  pp1  = ss0 + 140 ;
   double*  dp0  = ss0 + 149 ;
   double*  dp1  = ss0 + 167 ;
   double*  fp0  = ss0 + 185 ;
   double*  fp1  = ss0 + 215 ;
   double*  dd0  = ss0 + 245 ;
   double*  gp0  = ss0 + 281 ;
   double*  fd0  = ss0 + 371 ;
   double*  gd0  = ss0 + 431 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  pd1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  dd0  = ss0 + 268 ;
   double*  dd1  = ss0 + 304 ;
   double*  fp0  = ss0 + 340 ;
   double*  fp1  = ss0 + 370 ;
   double*  df0  = ss0 + 430 ;
   double*  fd0  = ss0 + 490 ;
   double*  fd1  = ss0 + 550 ;
   double*  gp0  = ss0 + 610 ;
   double*  ff0  = ss0 + 745 ;
   double*  gd0  = ss0 + 845 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp2  = ss0 + 210 ;
   double*  sd2  = ss0 + 216 ;
   double*  pp1  = ss0 + 222 ;
   double*  pd1  = ss0 + 249 ;
   double*  dp0  = ss0 + 285 ;
   double*  dp1  = ss0 + 303 ;
   double*  pf1  = ss0 + 357 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 507 ;
   double*  dd1  = ss0 + 543 ;
   double*  gp0  = ss0 + 615 ;
   double*  fd0  = ss0 + 795 ;
   double*  fd1  = ss0 + 855 ;
   double*  df0  = ss0 + 975 ;
   double*  df1  = ss0 + 1035 ;
   double*  dg0  = ss0 + 1095 ;
   double*  gd0  = ss0 + 1185 ;
   double*  ff0  = ss0 + 1455 ;
   double*  ff1  = ss0 + 1555 ;
   double*  gf0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 1955 ;
   double*  gg0  = ss0 + 2105 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp2  = ss0 + 245 ;
   double*  sd2  = ss0 + 254 ;
   double*  pp1  = ss0 + 266 ;
   double*  sf2  = ss0 + 302 ;
   double*  dp0  = ss0 + 312 ;
   double*  dp1  = ss0 + 330 ;
   double*  pd1  = ss0 + 402 ;
   double*  fp0  = ss0 + 456 ;
   double*  fp1  = ss0 + 486 ;
   double*  dd0  = ss0 + 606 ;
   double*  dd1  = ss0 + 642 ;
   double*  pf1  = ss0 + 750 ;
   double*  pg1  = ss0 + 810 ;
   double*  fd0  = ss0 + 855 ;
   double*  fd1  = ss0 + 915 ;
   double*  df0  = ss0 + 1095 ;
   double*  df1  = ss0 + 1155 ;
   double*  gp0  = ss0 + 1275 ;
   double*  dg0  = ss0 + 1500 ;
   double*  dg1  = ss0 + 1590 ;
   double*  ff0  = ss0 + 1680 ;
   double*  ff1  = ss0 + 1780 ;
   double*  gd0  = ss0 + 1980 ;
   double*  dh0  = ss0 + 2340 ;
   double*  fg0  = ss0 + 2466 ;
   double*  fg1  = ss0 + 2616 ;
   double*  gf0  = ss0 + 2766 ;
   double*  fh0  = ss0 + 3216 ;
   double*  gg0  = ss0 + 3426 ;
   double*  gh0  = ss0 + 3876 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  pd1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  dd0  = ss0 + 268 ;
   double*  dd1  = ss0 + 304 ;
   double*  fp0  = ss0 + 340 ;
   double*  fp1  = ss0 + 370 ;
   double*  df0  = ss0 + 430 ;
   double*  fd0  = ss0 + 490 ;
   double*  fd1  = ss0 + 550 ;
   double*  gp0  = ss0 + 610 ;
   double*  ff0  = ss0 + 745 ;
   double*  gd0  = ss0 + 845 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp2  = ss0 + 210 ;
   double*  sd2  = ss0 + 216 ;
   double*  pp1  = ss0 + 222 ;
   double*  dp0  = ss0 + 249 ;
   double*  dp1  = ss0 + 267 ;
   double*  pd1  = ss0 + 321 ;
   double*  pf1  = ss0 + 357 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 507 ;
   double*  dd1  = ss0 + 543 ;
   double*  fd0  = ss0 + 615 ;
   double*  fd1  = ss0 + 675 ;
   double*  gp0  = ss0 + 795 ;
   double*  df0  = ss0 + 975 ;
   double*  df1  = ss0 + 1035 ;
   double*  dg0  = ss0 + 1095 ;
   double*  gd0  = ss0 + 1185 ;
   double*  ff0  = ss0 + 1455 ;
   double*  ff1  = ss0 + 1555 ;
   double*  gf0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 1955 ;
   double*  gg0  = ss0 + 2105 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp2  = ss0 + 245 ;
   double*  sd2  = ss0 + 254 ;
   double*  pp1  = ss0 + 266 ;
   double*  sf2  = ss0 + 302 ;
   double*  dp0  = ss0 + 312 ;
   double*  dp1  = ss0 + 330 ;
   double*  pd1  = ss0 + 402 ;
   double*  fp0  = ss0 + 456 ;
   double*  fp1  = ss0 + 486 ;
   double*  dd0  = ss0 + 606 ;
   double*  dd1  = ss0 + 642 ;
   double*  pf1  = ss0 + 750 ;
   double*  pg1  = ss0 + 810 ;
   double*  fd0  = ss0 + 855 ;
   double*  fd1  = ss0 + 915 ;
   double*  df0  = ss0 + 1095 ;
   double*  df1  = ss0 + 1155 ;
   double*  gp0  = ss0 + 1275 ;
   double*  dg0  = ss0 + 1500 ;
   double*  dg1  = ss0 + 1590 ;
   double*  ff0  = ss0 + 1680 ;
   double*  ff1  = ss0 + 1780 ;
   double*  gd0  = ss0 + 1980 ;
   double*  dh0  = ss0 + 2340 ;
   double*  fg0  = ss0 + 2466 ;
   double*  fg1  = ss0 + 2616 ;
   double*  gf0  = ss0 + 2766 ;
   double*  fh0  = ss0 + 3216 ;
   double*  gg0  = ss0 + 3426 ;
   double*  gh0  = ss0 + 3876 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  sp2  = ss0 + 280 ;
   double*  sd2  = ss0 + 292 ;
   double*  pp1  = ss0 + 310 ;
   double*  dp0  = ss0 + 355 ;
   double*  dp1  = ss0 + 373 ;
   double*  sf2  = ss0 + 463 ;
   double*  pd1  = ss0 + 483 ;
   double*  sg2  = ss0 + 555 ;
   double*  pf1  = ss0 + 570 ;
   double*  fp0  = ss0 + 660 ;
   double*  fp1  = ss0 + 690 ;
   double*  dd0  = ss0 + 840 ;
   double*  dd1  = ss0 + 876 ;
   double*  df0  = ss0 + 1020 ;
   double*  df1  = ss0 + 1080 ;
   double*  fd0  = ss0 + 1260 ;
   double*  fd1  = ss0 + 1320 ;
   double*  pg1  = ss0 + 1560 ;
   double*  gp0  = ss0 + 1650 ;
   double*  dg0  = ss0 + 1920 ;
   double*  dg1  = ss0 + 2010 ;
   double*  ff0  = ss0 + 2190 ;
   double*  ff1  = ss0 + 2290 ;
   double*  gd0  = ss0 + 2590 ;
   double*  ph1  = ss0 + 3040 ;
   double*  gf0  = ss0 + 3103 ;
   double*  dh0  = ss0 + 3703 ;
   double*  dh1  = ss0 + 3829 ;
   double*  fg0  = ss0 + 3955 ;
   double*  fg1  = ss0 + 4105 ;
   double*  gg0  = ss0 + 4405 ;
   double*  fh0  = ss0 + 5080 ;
   double*  fh1  = ss0 + 5290 ;
   double*  di0  = ss0 + 5500 ;
   double*  gh0  = ss0 + 5668 ;
   double*  fi0  = ss0 + 6298 ;
   double*  gi0  = ss0 + 6578 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  fs0  = ss0 + 45 ;
   double*  gs0  = ss0 + 75 ;
   double*  hs0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( hs0 , gs0 , fs0 , PA_WP, zab );
}

template< int VTS >
__device__ void dfsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  dp0  = ss0 + 182 ;
   double*  fp0  = ss0 + 200 ;
   double*  gp0  = ss0 + 230 ;
   double*  hp0  = ss0 + 275 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void dfsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  pp1  = ss0 + 238 ;
   double*  dp0  = ss0 + 247 ;
   double*  dp1  = ss0 + 265 ;
   double*  dd0  = ss0 + 283 ;
   double*  fp0  = ss0 + 319 ;
   double*  fp1  = ss0 + 349 ;
   double*  fd0  = ss0 + 379 ;
   double*  gp0  = ss0 + 439 ;
   double*  gp1  = ss0 + 484 ;
   double*  gd0  = ss0 + 529 ;
   double*  hp0  = ss0 + 619 ;
   double*  hd0  = ss0 + 745 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  pd1  = ss0 + 315 ;
   double*  dp0  = ss0 + 333 ;
   double*  dp1  = ss0 + 351 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  ff0  = ss0 + 864 ;
   double*  hp0  = ss0 + 964 ;
   double*  gd0  = ss0 + 1153 ;
   double*  gd1  = ss0 + 1243 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  dp0  = ss0 + 182 ;
   double*  fp0  = ss0 + 200 ;
   double*  gp0  = ss0 + 230 ;
   double*  hp0  = ss0 + 275 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void dfpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  pp1  = ss0 + 238 ;
   double*  dp0  = ss0 + 247 ;
   double*  dp1  = ss0 + 265 ;
   double*  dd0  = ss0 + 283 ;
   double*  fp0  = ss0 + 319 ;
   double*  fp1  = ss0 + 349 ;
   double*  fd0  = ss0 + 379 ;
   double*  gp0  = ss0 + 439 ;
   double*  gp1  = ss0 + 484 ;
   double*  gd0  = ss0 + 529 ;
   double*  hp0  = ss0 + 619 ;
   double*  hd0  = ss0 + 745 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  pd1  = ss0 + 315 ;
   double*  dp0  = ss0 + 333 ;
   double*  dp1  = ss0 + 351 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  hp0  = ss0 + 864 ;
   double*  ff0  = ss0 + 1053 ;
   double*  gd0  = ss0 + 1153 ;
   double*  gd1  = ss0 + 1243 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp2  = ss0 + 350 ;
   double*  sd2  = ss0 + 356 ;
   double*  pp1  = ss0 + 362 ;
   double*  pd1  = ss0 + 389 ;
   double*  dp0  = ss0 + 425 ;
   double*  dp1  = ss0 + 443 ;
   double*  pf1  = ss0 + 497 ;
   double*  fp0  = ss0 + 527 ;
   double*  fp1  = ss0 + 557 ;
   double*  dd0  = ss0 + 647 ;
   double*  dd1  = ss0 + 683 ;
   double*  df0  = ss0 + 755 ;
   double*  df1  = ss0 + 815 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  gp0  = ss0 + 1055 ;
   double*  gp1  = ss0 + 1100 ;
   double*  dg0  = ss0 + 1235 ;
   double*  ff0  = ss0 + 1325 ;
   double*  ff1  = ss0 + 1425 ;
   double*  hp0  = ss0 + 1525 ;
   double*  gd0  = ss0 + 1777 ;
   double*  gd1  = ss0 + 1867 ;
   double*  fg0  = ss0 + 2047 ;
   double*  hd0  = ss0 + 2197 ;
   double*  gf0  = ss0 + 2575 ;
   double*  gf1  = ss0 + 2725 ;
   double*  gg0  = ss0 + 2875 ;
   double*  hf0  = ss0 + 3100 ;
   double*  hg0  = ss0 + 3520 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  pp1  = ss0 + 238 ;
   double*  dp0  = ss0 + 247 ;
   double*  dp1  = ss0 + 265 ;
   double*  fp0  = ss0 + 283 ;
   double*  fp1  = ss0 + 313 ;
   double*  dd0  = ss0 + 343 ;
   double*  fd0  = ss0 + 379 ;
   double*  gp0  = ss0 + 439 ;
   double*  gp1  = ss0 + 484 ;
   double*  gd0  = ss0 + 529 ;
   double*  hp0  = ss0 + 619 ;
   double*  hd0  = ss0 + 745 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  dp0  = ss0 + 315 ;
   double*  dp1  = ss0 + 333 ;
   double*  pd1  = ss0 + 369 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  hp0  = ss0 + 864 ;
   double*  gd0  = ss0 + 1053 ;
   double*  gd1  = ss0 + 1143 ;
   double*  ff0  = ss0 + 1233 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp2  = ss0 + 350 ;
   double*  sd2  = ss0 + 356 ;
   double*  pp1  = ss0 + 362 ;
   double*  pd1  = ss0 + 389 ;
   double*  dp0  = ss0 + 425 ;
   double*  dp1  = ss0 + 443 ;
   double*  pf1  = ss0 + 497 ;
   double*  dd0  = ss0 + 527 ;
   double*  dd1  = ss0 + 563 ;
   double*  fp0  = ss0 + 635 ;
   double*  fp1  = ss0 + 665 ;
   double*  df0  = ss0 + 755 ;
   double*  df1  = ss0 + 815 ;
   double*  gp0  = ss0 + 875 ;
   double*  gp1  = ss0 + 920 ;
   double*  fd0  = ss0 + 1055 ;
   double*  fd1  = ss0 + 1115 ;
   double*  dg0  = ss0 + 1235 ;
   double*  ff0  = ss0 + 1325 ;
   double*  ff1  = ss0 + 1425 ;
   double*  hp0  = ss0 + 1525 ;
   double*  gd0  = ss0 + 1777 ;
   double*  gd1  = ss0 + 1867 ;
   double*  fg0  = ss0 + 2047 ;
   double*  hd0  = ss0 + 2197 ;
   double*  gf0  = ss0 + 2575 ;
   double*  gf1  = ss0 + 2725 ;
   double*  gg0  = ss0 + 2875 ;
   double*  hf0  = ss0 + 3100 ;
   double*  hg0  = ss0 + 3520 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp2  = ss0 + 406 ;
   double*  pp1  = ss0 + 415 ;
   double*  sd2  = ss0 + 451 ;
   double*  sf2  = ss0 + 463 ;
   double*  pd1  = ss0 + 473 ;
   double*  dp0  = ss0 + 527 ;
   double*  dp1  = ss0 + 545 ;
   double*  pf1  = ss0 + 617 ;
   double*  fp0  = ss0 + 677 ;
   double*  fp1  = ss0 + 707 ;
   double*  dd0  = ss0 + 827 ;
   double*  dd1  = ss0 + 863 ;
   double*  pg1  = ss0 + 971 ;
   double*  df0  = ss0 + 1016 ;
   double*  df1  = ss0 + 1076 ;
   double*  fd0  = ss0 + 1196 ;
   double*  fd1  = ss0 + 1256 ;
   double*  gp0  = ss0 + 1436 ;
   double*  gp1  = ss0 + 1481 ;
   double*  gd0  = ss0 + 1661 ;
   double*  gd1  = ss0 + 1751 ;
   double*  ff0  = ss0 + 2021 ;
   double*  ff1  = ss0 + 2121 ;
   double*  hp0  = ss0 + 2321 ;
   double*  dg0  = ss0 + 2636 ;
   double*  dg1  = ss0 + 2726 ;
   double*  hd0  = ss0 + 2816 ;
   double*  gf0  = ss0 + 3320 ;
   double*  gf1  = ss0 + 3470 ;
   double*  dh0  = ss0 + 3770 ;
   double*  fg0  = ss0 + 3896 ;
   double*  fg1  = ss0 + 4046 ;
   double*  hf0  = ss0 + 4196 ;
   double*  gg0  = ss0 + 4826 ;
   double*  gg1  = ss0 + 5051 ;
   double*  fh0  = ss0 + 5276 ;
   double*  hg0  = ss0 + 5486 ;
   double*  gh0  = ss0 + 6116 ;
   double*  hh0  = ss0 + 6431 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  dp0  = ss0 + 315 ;
   double*  dp1  = ss0 + 333 ;
   double*  pd1  = ss0 + 369 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  hp0  = ss0 + 864 ;
   double*  gd0  = ss0 + 1053 ;
   double*  gd1  = ss0 + 1143 ;
   double*  ff0  = ss0 + 1233 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp2  = ss0 + 350 ;
   double*  sd2  = ss0 + 356 ;
   double*  pp1  = ss0 + 362 ;
   double*  dp0  = ss0 + 389 ;
   double*  dp1  = ss0 + 407 ;
   double*  pd1  = ss0 + 461 ;
   double*  pf1  = ss0 + 497 ;
   double*  dd0  = ss0 + 527 ;
   double*  dd1  = ss0 + 563 ;
   double*  fp0  = ss0 + 635 ;
   double*  fp1  = ss0 + 665 ;
   double*  df0  = ss0 + 755 ;
   double*  df1  = ss0 + 815 ;
   double*  gp0  = ss0 + 875 ;
   double*  gp1  = ss0 + 920 ;
   double*  fd0  = ss0 + 1055 ;
   double*  fd1  = ss0 + 1115 ;
   double*  dg0  = ss0 + 1235 ;
   double*  ff0  = ss0 + 1325 ;
   double*  ff1  = ss0 + 1425 ;
   double*  gd0  = ss0 + 1525 ;
   double*  gd1  = ss0 + 1615 ;
   double*  hp0  = ss0 + 1795 ;
   double*  fg0  = ss0 + 2047 ;
   double*  hd0  = ss0 + 2197 ;
   double*  gf0  = ss0 + 2575 ;
   double*  gf1  = ss0 + 2725 ;
   double*  gg0  = ss0 + 2875 ;
   double*  hf0  = ss0 + 3100 ;
   double*  hg0  = ss0 + 3520 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp2  = ss0 + 406 ;
   double*  pp1  = ss0 + 415 ;
   double*  sd2  = ss0 + 451 ;
   double*  sf2  = ss0 + 463 ;
   double*  dp0  = ss0 + 473 ;
   double*  dp1  = ss0 + 491 ;
   double*  pd1  = ss0 + 563 ;
   double*  pf1  = ss0 + 617 ;
   double*  fp0  = ss0 + 677 ;
   double*  fp1  = ss0 + 707 ;
   double*  dd0  = ss0 + 827 ;
   double*  dd1  = ss0 + 863 ;
   double*  pg1  = ss0 + 971 ;
   double*  df0  = ss0 + 1016 ;
   double*  df1  = ss0 + 1076 ;
   double*  fd0  = ss0 + 1196 ;
   double*  fd1  = ss0 + 1256 ;
   double*  gp0  = ss0 + 1436 ;
   double*  gp1  = ss0 + 1481 ;
   double*  gd0  = ss0 + 1661 ;
   double*  gd1  = ss0 + 1751 ;
   double*  ff0  = ss0 + 2021 ;
   double*  ff1  = ss0 + 2121 ;
   double*  dg0  = ss0 + 2321 ;
   double*  dg1  = ss0 + 2411 ;
   double*  hp0  = ss0 + 2501 ;
   double*  hd0  = ss0 + 2816 ;
   double*  gf0  = ss0 + 3320 ;
   double*  gf1  = ss0 + 3470 ;
   double*  fg0  = ss0 + 3770 ;
   double*  fg1  = ss0 + 3920 ;
   double*  dh0  = ss0 + 4070 ;
   double*  hf0  = ss0 + 4196 ;
   double*  gg0  = ss0 + 4826 ;
   double*  gg1  = ss0 + 5051 ;
   double*  fh0  = ss0 + 5276 ;
   double*  hg0  = ss0 + 5486 ;
   double*  gh0  = ss0 + 6116 ;
   double*  hh0  = ss0 + 6431 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps1  = ss0 + 15 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  sp2  = ss0 + 462 ;
   double*  sd2  = ss0 + 474 ;
   double*  pp1  = ss0 + 492 ;
   double*  sf2  = ss0 + 537 ;
   double*  pd1  = ss0 + 557 ;
   double*  dp0  = ss0 + 629 ;
   double*  dp1  = ss0 + 647 ;
   double*  sg2  = ss0 + 737 ;
   double*  pf1  = ss0 + 752 ;
   double*  dd0  = ss0 + 842 ;
   double*  dd1  = ss0 + 878 ;
   double*  fp0  = ss0 + 1022 ;
   double*  fp1  = ss0 + 1052 ;
   double*  pg1  = ss0 + 1202 ;
   double*  df0  = ss0 + 1292 ;
   double*  df1  = ss0 + 1352 ;
   double*  gp0  = ss0 + 1532 ;
   double*  gp1  = ss0 + 1577 ;
   double*  fd0  = ss0 + 1802 ;
   double*  fd1  = ss0 + 1862 ;
   double*  ph1  = ss0 + 2102 ;
   double*  dg0  = ss0 + 2165 ;
   double*  dg1  = ss0 + 2255 ;
   double*  gd0  = ss0 + 2435 ;
   double*  gd1  = ss0 + 2525 ;
   double*  ff0  = ss0 + 2885 ;
   double*  ff1  = ss0 + 2985 ;
   double*  hp0  = ss0 + 3285 ;
   double*  dh0  = ss0 + 3663 ;
   double*  dh1  = ss0 + 3789 ;
   double*  gf0  = ss0 + 3915 ;
   double*  gf1  = ss0 + 4065 ;
   double*  hd0  = ss0 + 4515 ;
   double*  fg0  = ss0 + 5145 ;
   double*  fg1  = ss0 + 5295 ;
   double*  di0  = ss0 + 5595 ;
   double*  fh0  = ss0 + 5763 ;
   double*  fh1  = ss0 + 5973 ;
   double*  hf0  = ss0 + 6183 ;
   double*  gg0  = ss0 + 7023 ;
   double*  gg1  = ss0 + 7248 ;
   double*  fi0  = ss0 + 7698 ;
   double*  hg0  = ss0 + 7978 ;
   double*  gh0  = ss0 + 8923 ;
   double*  gh1  = ss0 + 9238 ;
   double*  hh0  = ss0 + 9553 ;
   double*  gi0  = ss0 + 10435 ;
   double*  hi0  = ss0 + 10855 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 6 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 5 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 4 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 3 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 2 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 6, 1 >( hi0 , hh0 , hg0 , gh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void fssp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  fp0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void fssd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps2  = ss0 + 12 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  dp1  = ss0 + 75 ;
   double*  fp0  = ss0 + 93 ;
   double*  fd0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fssf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  dd1  = ss0 + 140 ;
   double*  fp0  = ss0 + 176 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  fp0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void fspp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps2  = ss0 + 12 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  dp1  = ss0 + 75 ;
   double*  fp0  = ss0 + 93 ;
   double*  fd0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fspd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  fp0  = ss0 + 140 ;
   double*  dd1  = ss0 + 230 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fspf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp3  = ss0 + 115 ;
   double*  pp2  = ss0 + 118 ;
   double*  pd2  = ss0 + 136 ;
   double*  dp1  = ss0 + 154 ;
   double*  dd1  = ss0 + 208 ;
   double*  fp0  = ss0 + 280 ;
   double*  df1  = ss0 + 400 ;
   double*  fd0  = ss0 + 460 ;
   double*  ff0  = ss0 + 640 ;
   double*  fg0  = ss0 + 840 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps2  = ss0 + 12 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  dp1  = ss0 + 75 ;
   double*  fp0  = ss0 + 93 ;
   double*  fd0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  fp0  = ss0 + 140 ;
   double*  dd1  = ss0 + 230 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp3  = ss0 + 115 ;
   double*  pp2  = ss0 + 118 ;
   double*  pd2  = ss0 + 136 ;
   double*  dp1  = ss0 + 154 ;
   double*  dd1  = ss0 + 208 ;
   double*  fp0  = ss0 + 280 ;
   double*  df1  = ss0 + 400 ;
   double*  fd0  = ss0 + 460 ;
   double*  ff0  = ss0 + 640 ;
   double*  fg0  = ss0 + 840 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp3  = ss0 + 135 ;
   double*  pp2  = ss0 + 141 ;
   double*  sd3  = ss0 + 168 ;
   double*  dp1  = ss0 + 174 ;
   double*  pd2  = ss0 + 246 ;
   double*  fp0  = ss0 + 282 ;
   double*  dd1  = ss0 + 432 ;
   double*  pf2  = ss0 + 540 ;
   double*  df1  = ss0 + 570 ;
   double*  fd0  = ss0 + 690 ;
   double*  dg1  = ss0 + 930 ;
   double*  ff0  = ss0 + 1020 ;
   double*  fg0  = ss0 + 1320 ;
   double*  fh0  = ss0 + 1620 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  fp0  = ss0 + 140 ;
   double*  dd1  = ss0 + 230 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp3  = ss0 + 115 ;
   double*  pp2  = ss0 + 118 ;
   double*  pd2  = ss0 + 136 ;
   double*  dp1  = ss0 + 154 ;
   double*  dd1  = ss0 + 208 ;
   double*  fp0  = ss0 + 280 ;
   double*  df1  = ss0 + 400 ;
   double*  fd0  = ss0 + 460 ;
   double*  ff0  = ss0 + 640 ;
   double*  fg0  = ss0 + 840 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp3  = ss0 + 135 ;
   double*  sd3  = ss0 + 141 ;
   double*  pp2  = ss0 + 147 ;
   double*  pd2  = ss0 + 174 ;
   double*  dp1  = ss0 + 210 ;
   double*  pf2  = ss0 + 282 ;
   double*  dd1  = ss0 + 312 ;
   double*  fp0  = ss0 + 420 ;
   double*  fd0  = ss0 + 570 ;
   double*  df1  = ss0 + 810 ;
   double*  ff0  = ss0 + 930 ;
   double*  dg1  = ss0 + 1230 ;
   double*  fg0  = ss0 + 1320 ;
   double*  fh0  = ss0 + 1620 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp3  = ss0 + 155 ;
   double*  sd3  = ss0 + 164 ;
   double*  pp2  = ss0 + 176 ;
   double*  sf3  = ss0 + 212 ;
   double*  dp1  = ss0 + 222 ;
   double*  pd2  = ss0 + 312 ;
   double*  fp0  = ss0 + 366 ;
   double*  dd1  = ss0 + 546 ;
   double*  pf2  = ss0 + 690 ;
   double*  pg2  = ss0 + 750 ;
   double*  fd0  = ss0 + 795 ;
   double*  df1  = ss0 + 1095 ;
   double*  dg1  = ss0 + 1275 ;
   double*  ff0  = ss0 + 1455 ;
   double*  dh1  = ss0 + 1855 ;
   double*  fg0  = ss0 + 1981 ;
   double*  fh0  = ss0 + 2431 ;
   double*  fi0  = ss0 + 2851 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  fs0  = ss0 + 35 ;
   double*  gs0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( gs0 , fs0 , ds0 , PA_WP, zab );
}

template< int VTS >
__device__ void fpsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  fp0  = ss0 + 105 ;
   double*  gp0  = ss0 + 135 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fpsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  dp1  = ss0 + 140 ;
   double*  fp0  = ss0 + 158 ;
   double*  fp1  = ss0 + 188 ;
   double*  gp0  = ss0 + 218 ;
   double*  fd0  = ss0 + 308 ;
   double*  gd0  = ss0 + 368 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  fp0  = ss0 + 105 ;
   double*  gp0  = ss0 + 135 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  dp1  = ss0 + 140 ;
   double*  fp0  = ss0 + 158 ;
   double*  fp1  = ss0 + 188 ;
   double*  fd0  = ss0 + 218 ;
   double*  gp0  = ss0 + 278 ;
   double*  gd0  = ss0 + 368 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp3  = ss0 + 210 ;
   double*  pp2  = ss0 + 213 ;
   double*  dp1  = ss0 + 231 ;
   double*  pd2  = ss0 + 285 ;
   double*  fp0  = ss0 + 303 ;
   double*  fp1  = ss0 + 333 ;
   double*  dd1  = ss0 + 423 ;
   double*  df1  = ss0 + 495 ;
   double*  gp0  = ss0 + 555 ;
   double*  fd0  = ss0 + 735 ;
   double*  fd1  = ss0 + 795 ;
   double*  gd0  = ss0 + 915 ;
   double*  ff0  = ss0 + 1185 ;
   double*  ff1  = ss0 + 1285 ;
   double*  gf0  = ss0 + 1385 ;
   double*  fg0  = ss0 + 1685 ;
   double*  gg0  = ss0 + 1835 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  dp1  = ss0 + 140 ;
   double*  fp0  = ss0 + 158 ;
   double*  fp1  = ss0 + 188 ;
   double*  fd0  = ss0 + 218 ;
   double*  gp0  = ss0 + 278 ;
   double*  gd0  = ss0 + 368 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp3  = ss0 + 210 ;
   double*  pp2  = ss0 + 213 ;
   double*  pd2  = ss0 + 231 ;
   double*  dp1  = ss0 + 249 ;
   double*  fp0  = ss0 + 303 ;
   double*  fp1  = ss0 + 333 ;
   double*  dd1  = ss0 + 423 ;
   double*  gp0  = ss0 + 495 ;
   double*  fd0  = ss0 + 675 ;
   double*  fd1  = ss0 + 735 ;
   double*  df1  = ss0 + 855 ;
   double*  gd0  = ss0 + 915 ;
   double*  ff0  = ss0 + 1185 ;
   double*  ff1  = ss0 + 1285 ;
   double*  gf0  = ss0 + 1385 ;
   double*  fg0  = ss0 + 1685 ;
   double*  gg0  = ss0 + 1835 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp3  = ss0 + 245 ;
   double*  sd3  = ss0 + 251 ;
   double*  pp2  = ss0 + 257 ;
   double*  dp1  = ss0 + 284 ;
   double*  pd2  = ss0 + 356 ;
   double*  fp0  = ss0 + 392 ;
   double*  fp1  = ss0 + 422 ;
   double*  dd1  = ss0 + 542 ;
   double*  pf2  = ss0 + 650 ;
   double*  fd0  = ss0 + 680 ;
   double*  fd1  = ss0 + 740 ;
   double*  df1  = ss0 + 920 ;
   double*  gp0  = ss0 + 1040 ;
   double*  dg1  = ss0 + 1265 ;
   double*  ff0  = ss0 + 1355 ;
   double*  ff1  = ss0 + 1455 ;
   double*  gd0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 2015 ;
   double*  fg1  = ss0 + 2165 ;
   double*  gf0  = ss0 + 2315 ;
   double*  fh0  = ss0 + 2765 ;
   double*  gg0  = ss0 + 2975 ;
   double*  gh0  = ss0 + 3425 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp3  = ss0 + 210 ;
   double*  pp2  = ss0 + 213 ;
   double*  pd2  = ss0 + 231 ;
   double*  dp1  = ss0 + 249 ;
   double*  dd1  = ss0 + 303 ;
   double*  fp0  = ss0 + 375 ;
   double*  fp1  = ss0 + 405 ;
   double*  df1  = ss0 + 495 ;
   double*  gp0  = ss0 + 555 ;
   double*  fd0  = ss0 + 735 ;
   double*  fd1  = ss0 + 795 ;
   double*  ff0  = ss0 + 915 ;
   double*  ff1  = ss0 + 1015 ;
   double*  gd0  = ss0 + 1115 ;
   double*  fg0  = ss0 + 1385 ;
   double*  gf0  = ss0 + 1535 ;
   double*  gg0  = ss0 + 1835 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp3  = ss0 + 245 ;
   double*  sd3  = ss0 + 251 ;
   double*  pp2  = ss0 + 257 ;
   double*  dp1  = ss0 + 284 ;
   double*  pd2  = ss0 + 356 ;
   double*  dd1  = ss0 + 392 ;
   double*  pf2  = ss0 + 500 ;
   double*  fp0  = ss0 + 530 ;
   double*  fp1  = ss0 + 560 ;
   double*  fd0  = ss0 + 680 ;
   double*  fd1  = ss0 + 740 ;
   double*  df1  = ss0 + 920 ;
   double*  gp0  = ss0 + 1040 ;
   double*  dg1  = ss0 + 1265 ;
   double*  ff0  = ss0 + 1355 ;
   double*  ff1  = ss0 + 1455 ;
   double*  gd0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 2015 ;
   double*  fg1  = ss0 + 2165 ;
   double*  gf0  = ss0 + 2315 ;
   double*  fh0  = ss0 + 2765 ;
   double*  gg0  = ss0 + 2975 ;
   double*  gh0  = ss0 + 3425 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  sp3  = ss0 + 280 ;
   double*  pp2  = ss0 + 289 ;
   double*  sd3  = ss0 + 325 ;
   double*  sf3  = ss0 + 337 ;
   double*  pd2  = ss0 + 347 ;
   double*  dp1  = ss0 + 401 ;
   double*  fp0  = ss0 + 491 ;
   double*  fp1  = ss0 + 521 ;
   double*  pf2  = ss0 + 671 ;
   double*  dd1  = ss0 + 731 ;
   double*  gp0  = ss0 + 875 ;
   double*  pg2  = ss0 + 1145 ;
   double*  fd0  = ss0 + 1190 ;
   double*  fd1  = ss0 + 1250 ;
   double*  df1  = ss0 + 1490 ;
   double*  dg1  = ss0 + 1670 ;
   double*  gd0  = ss0 + 1850 ;
   double*  ff0  = ss0 + 2300 ;
   double*  ff1  = ss0 + 2400 ;
   double*  fg0  = ss0 + 2700 ;
   double*  fg1  = ss0 + 2850 ;
   double*  dh1  = ss0 + 3150 ;
   double*  gf0  = ss0 + 3276 ;
   double*  fh0  = ss0 + 3876 ;
   double*  fh1  = ss0 + 4086 ;
   double*  gg0  = ss0 + 4296 ;
   double*  fi0  = ss0 + 4971 ;
   double*  gh0  = ss0 + 5251 ;
   double*  gi0  = ss0 + 5881 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  fs0  = ss0 + 45 ;
   double*  gs0  = ss0 + 75 ;
   double*  hs0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( hs0 , gs0 , fs0 , PA_WP, zab );
}

template< int VTS >
__device__ void fdsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  fp0  = ss0 + 182 ;
   double*  gp0  = ss0 + 212 ;
   double*  hp0  = ss0 + 257 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fdsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  dp1  = ss0 + 238 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 316 ;
   double*  gp0  = ss0 + 376 ;
   double*  gp1  = ss0 + 421 ;
   double*  gd0  = ss0 + 466 ;
   double*  hp0  = ss0 + 556 ;
   double*  hd0  = ss0 + 682 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  fp0  = ss0 + 339 ;
   double*  fp1  = ss0 + 369 ;
   double*  dd1  = ss0 + 429 ;
   double*  gp0  = ss0 + 465 ;
   double*  gp1  = ss0 + 510 ;
   double*  fd0  = ss0 + 600 ;
   double*  fd1  = ss0 + 660 ;
   double*  ff0  = ss0 + 720 ;
   double*  hp0  = ss0 + 820 ;
   double*  gd0  = ss0 + 1009 ;
   double*  gd1  = ss0 + 1099 ;
   double*  hd0  = ss0 + 1189 ;
   double*  gf0  = ss0 + 1441 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  fp0  = ss0 + 182 ;
   double*  gp0  = ss0 + 212 ;
   double*  hp0  = ss0 + 257 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fdpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  dp1  = ss0 + 238 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 316 ;
   double*  gp0  = ss0 + 376 ;
   double*  gp1  = ss0 + 421 ;
   double*  gd0  = ss0 + 466 ;
   double*  hp0  = ss0 + 556 ;
   double*  hd0  = ss0 + 682 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  fp0  = ss0 + 339 ;
   double*  fp1  = ss0 + 369 ;
   double*  dd1  = ss0 + 429 ;
   double*  gp0  = ss0 + 465 ;
   double*  gp1  = ss0 + 510 ;
   double*  fd0  = ss0 + 600 ;
   double*  fd1  = ss0 + 660 ;
   double*  hp0  = ss0 + 720 ;
   double*  ff0  = ss0 + 909 ;
   double*  gd0  = ss0 + 1009 ;
   double*  gd1  = ss0 + 1099 ;
   double*  hd0  = ss0 + 1189 ;
   double*  gf0  = ss0 + 1441 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp3  = ss0 + 350 ;
   double*  pp2  = ss0 + 353 ;
   double*  pd2  = ss0 + 371 ;
   double*  dp1  = ss0 + 389 ;
   double*  fp0  = ss0 + 443 ;
   double*  fp1  = ss0 + 473 ;
   double*  dd1  = ss0 + 563 ;
   double*  df1  = ss0 + 635 ;
   double*  gp0  = ss0 + 695 ;
   double*  gp1  = ss0 + 740 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  ff0  = ss0 + 1055 ;
   double*  ff1  = ss0 + 1155 ;
   double*  hp0  = ss0 + 1255 ;
   double*  gd0  = ss0 + 1507 ;
   double*  gd1  = ss0 + 1597 ;
   double*  fg0  = ss0 + 1777 ;
   double*  hd0  = ss0 + 1927 ;
   double*  gf0  = ss0 + 2305 ;
   double*  gf1  = ss0 + 2455 ;
   double*  gg0  = ss0 + 2605 ;
   double*  hf0  = ss0 + 2830 ;
   double*  hg0  = ss0 + 3250 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  dp1  = ss0 + 238 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 316 ;
   double*  gp0  = ss0 + 376 ;
   double*  gp1  = ss0 + 421 ;
   double*  gd0  = ss0 + 466 ;
   double*  hp0  = ss0 + 556 ;
   double*  hd0  = ss0 + 682 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  dd1  = ss0 + 339 ;
   double*  fp0  = ss0 + 375 ;
   double*  fp1  = ss0 + 405 ;
   double*  fd0  = ss0 + 465 ;
   double*  fd1  = ss0 + 525 ;
   double*  gp0  = ss0 + 585 ;
   double*  gp1  = ss0 + 630 ;
   double*  gd0  = ss0 + 720 ;
   double*  gd1  = ss0 + 810 ;
   double*  ff0  = ss0 + 900 ;
   double*  hp0  = ss0 + 1000 ;
   double*  gf0  = ss0 + 1189 ;
   double*  hd0  = ss0 + 1339 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp3  = ss0 + 350 ;
   double*  pp2  = ss0 + 353 ;
   double*  pd2  = ss0 + 371 ;
   double*  dp1  = ss0 + 389 ;
   double*  dd1  = ss0 + 443 ;
   double*  fp0  = ss0 + 515 ;
   double*  fp1  = ss0 + 545 ;
   double*  df1  = ss0 + 635 ;
   double*  gp0  = ss0 + 695 ;
   double*  gp1  = ss0 + 740 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  ff0  = ss0 + 1055 ;
   double*  ff1  = ss0 + 1155 ;
   double*  hp0  = ss0 + 1255 ;
   double*  gd0  = ss0 + 1507 ;
   double*  gd1  = ss0 + 1597 ;
   double*  fg0  = ss0 + 1777 ;
   double*  hd0  = ss0 + 1927 ;
   double*  gf0  = ss0 + 2305 ;
   double*  gf1  = ss0 + 2455 ;
   double*  gg0  = ss0 + 2605 ;
   double*  hf0  = ss0 + 2830 ;
   double*  hg0  = ss0 + 3250 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp3  = ss0 + 406 ;
   double*  sd3  = ss0 + 412 ;
   double*  pp2  = ss0 + 418 ;
   double*  pd2  = ss0 + 445 ;
   double*  dp1  = ss0 + 481 ;
   double*  pf2  = ss0 + 553 ;
   double*  fp0  = ss0 + 583 ;
   double*  fp1  = ss0 + 613 ;
   double*  dd1  = ss0 + 733 ;
   double*  gp0  = ss0 + 841 ;
   double*  gp1  = ss0 + 886 ;
   double*  fd0  = ss0 + 1066 ;
   double*  fd1  = ss0 + 1126 ;
   double*  df1  = ss0 + 1306 ;
   double*  dg1  = ss0 + 1426 ;
   double*  hp0  = ss0 + 1516 ;
   double*  gd0  = ss0 + 1831 ;
   double*  gd1  = ss0 + 1921 ;
   double*  ff0  = ss0 + 2191 ;
   double*  ff1  = ss0 + 2291 ;
   double*  fg0  = ss0 + 2491 ;
   double*  fg1  = ss0 + 2641 ;
   double*  hd0  = ss0 + 2791 ;
   double*  gf0  = ss0 + 3295 ;
   double*  gf1  = ss0 + 3445 ;
   double*  fh0  = ss0 + 3745 ;
   double*  gg0  = ss0 + 3955 ;
   double*  gg1  = ss0 + 4180 ;
   double*  hf0  = ss0 + 4405 ;
   double*  gh0  = ss0 + 5035 ;
   double*  hg0  = ss0 + 5350 ;
   double*  hh0  = ss0 + 5980 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  dd1  = ss0 + 339 ;
   double*  fp0  = ss0 + 375 ;
   double*  fp1  = ss0 + 405 ;
   double*  fd0  = ss0 + 465 ;
   double*  fd1  = ss0 + 525 ;
   double*  gp0  = ss0 + 585 ;
   double*  gp1  = ss0 + 630 ;
   double*  gd0  = ss0 + 720 ;
   double*  gd1  = ss0 + 810 ;
   double*  ff0  = ss0 + 900 ;
   double*  hp0  = ss0 + 1000 ;
   double*  gf0  = ss0 + 1189 ;
   double*  hd0  = ss0 + 1339 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp3  = ss0 + 350 ;
   double*  pp2  = ss0 + 353 ;
   double*  pd2  = ss0 + 371 ;
   double*  dp1  = ss0 + 389 ;
   double*  dd1  = ss0 + 443 ;
   double*  fp0  = ss0 + 515 ;
   double*  fp1  = ss0 + 545 ;
   double*  df1  = ss0 + 635 ;
   double*  gp0  = ss0 + 695 ;
   double*  gp1  = ss0 + 740 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  ff0  = ss0 + 1055 ;
   double*  ff1  = ss0 + 1155 ;
   double*  gd0  = ss0 + 1255 ;
   double*  gd1  = ss0 + 1345 ;
   double*  hp0  = ss0 + 1525 ;
   double*  fg0  = ss0 + 1777 ;
   double*  hd0  = ss0 + 1927 ;
   double*  gf0  = ss0 + 2305 ;
   double*  gf1  = ss0 + 2455 ;
   double*  gg0  = ss0 + 2605 ;
   double*  hf0  = ss0 + 2830 ;
   double*  hg0  = ss0 + 3250 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp3  = ss0 + 406 ;
   double*  sd3  = ss0 + 412 ;
   double*  pp2  = ss0 + 418 ;
   double*  dp1  = ss0 + 445 ;
   double*  pd2  = ss0 + 517 ;
   double*  pf2  = ss0 + 553 ;
   double*  fp0  = ss0 + 583 ;
   double*  fp1  = ss0 + 613 ;
   double*  dd1  = ss0 + 733 ;
   double*  gp0  = ss0 + 841 ;
   double*  gp1  = ss0 + 886 ;
   double*  fd0  = ss0 + 1066 ;
   double*  fd1  = ss0 + 1126 ;
   double*  df1  = ss0 + 1306 ;
   double*  dg1  = ss0 + 1426 ;
   double*  hp0  = ss0 + 1516 ;
   double*  gd0  = ss0 + 1831 ;
   double*  gd1  = ss0 + 1921 ;
   double*  ff0  = ss0 + 2191 ;
   double*  ff1  = ss0 + 2291 ;
   double*  fg0  = ss0 + 2491 ;
   double*  fg1  = ss0 + 2641 ;
   double*  hd0  = ss0 + 2791 ;
   double*  gf0  = ss0 + 3295 ;
   double*  gf1  = ss0 + 3445 ;
   double*  fh0  = ss0 + 3745 ;
   double*  gg0  = ss0 + 3955 ;
   double*  gg1  = ss0 + 4180 ;
   double*  hf0  = ss0 + 4405 ;
   double*  gh0  = ss0 + 5035 ;
   double*  hg0  = ss0 + 5350 ;
   double*  hh0  = ss0 + 5980 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps2  = ss0 + 18 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  sp3  = ss0 + 462 ;
   double*  sd3  = ss0 + 471 ;
   double*  pp2  = ss0 + 483 ;
   double*  sf3  = ss0 + 519 ;
   double*  pd2  = ss0 + 529 ;
   double*  dp1  = ss0 + 583 ;
   double*  pf2  = ss0 + 673 ;
   double*  dd1  = ss0 + 733 ;
   double*  fp0  = ss0 + 877 ;
   double*  fp1  = ss0 + 907 ;
   double*  pg2  = ss0 + 1057 ;
   double*  df1  = ss0 + 1102 ;
   double*  gp0  = ss0 + 1282 ;
   double*  gp1  = ss0 + 1327 ;
   double*  fd0  = ss0 + 1552 ;
   double*  fd1  = ss0 + 1612 ;
   double*  dg1  = ss0 + 1852 ;
   double*  ff0  = ss0 + 2032 ;
   double*  ff1  = ss0 + 2132 ;
   double*  gd0  = ss0 + 2432 ;
   double*  gd1  = ss0 + 2522 ;
   double*  hp0  = ss0 + 2882 ;
   double*  dh1  = ss0 + 3260 ;
   double*  gf0  = ss0 + 3386 ;
   double*  gf1  = ss0 + 3536 ;
   double*  hd0  = ss0 + 3986 ;
   double*  fg0  = ss0 + 4616 ;
   double*  fg1  = ss0 + 4766 ;
   double*  fh0  = ss0 + 5066 ;
   double*  fh1  = ss0 + 5276 ;
   double*  hf0  = ss0 + 5486 ;
   double*  gg0  = ss0 + 6326 ;
   double*  gg1  = ss0 + 6551 ;
   double*  fi0  = ss0 + 7001 ;
   double*  hg0  = ss0 + 7281 ;
   double*  gh0  = ss0 + 8226 ;
   double*  gh1  = ss0 + 8541 ;
   double*  hh0  = ss0 + 8856 ;
   double*  gi0  = ss0 + 9738 ;
   double*  hi0  = ss0 + 10158 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 6 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 5 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 4 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 3 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 2 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 6, 1 >( hi0 , hh0 , hg0 , gh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 7 ;
   double*  ds0  = ss0 + 25 ;
   double*  fs0  = ss0 + 55 ;
   double*  gs0  = ss0 + 95 ;
   double*  hs0  = ss0 + 140 ;
   double*  is0  = ss0 + 182 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( is0 , hs0 , gs0 , PA_WP, zab );
}

template< int VTS >
__device__ void ffsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  hs1  = ss0 + 196 ;
   double*  is0  = ss0 + 238 ;
   double*  fp0  = ss0 + 294 ;
   double*  gp0  = ss0 + 324 ;
   double*  hp0  = ss0 + 369 ;
   double*  ip0  = ss0 + 432 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 2 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 6, 1, 1 >( ip0 , is0 , hs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ffsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  hs1  = ss0 + 231 ;
   double*  is0  = ss0 + 294 ;
   double*  dp1  = ss0 + 378 ;
   double*  fp0  = ss0 + 396 ;
   double*  fp1  = ss0 + 426 ;
   double*  fd0  = ss0 + 456 ;
   double*  gp0  = ss0 + 516 ;
   double*  gp1  = ss0 + 561 ;
   double*  hp0  = ss0 + 606 ;
   double*  hp1  = ss0 + 669 ;
   double*  gd0  = ss0 + 732 ;
   double*  ip0  = ss0 + 822 ;
   double*  hd0  = ss0 + 990 ;
   double*  id0  = ss0 + 1116 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 3 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 6, 1, 2 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 1 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  fd0  = ss0 + 633 ;
   double*  fd1  = ss0 + 693 ;
   double*  gp0  = ss0 + 753 ;
   double*  gp1  = ss0 + 798 ;
   double*  ff0  = ss0 + 888 ;
   double*  gd0  = ss0 + 988 ;
   double*  gd1  = ss0 + 1078 ;
   double*  hp0  = ss0 + 1168 ;
   double*  hp1  = ss0 + 1231 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  hs1  = ss0 + 196 ;
   double*  is0  = ss0 + 238 ;
   double*  fp0  = ss0 + 294 ;
   double*  gp0  = ss0 + 324 ;
   double*  hp0  = ss0 + 369 ;
   double*  ip0  = ss0 + 432 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 2 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 6, 1, 1 >( ip0 , is0 , hs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ffpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  hs1  = ss0 + 231 ;
   double*  is0  = ss0 + 294 ;
   double*  dp1  = ss0 + 378 ;
   double*  fp0  = ss0 + 396 ;
   double*  fp1  = ss0 + 426 ;
   double*  fd0  = ss0 + 456 ;
   double*  gp0  = ss0 + 516 ;
   double*  gp1  = ss0 + 561 ;
   double*  gd0  = ss0 + 606 ;
   double*  hp0  = ss0 + 696 ;
   double*  hp1  = ss0 + 759 ;
   double*  hd0  = ss0 + 822 ;
   double*  ip0  = ss0 + 948 ;
   double*  id0  = ss0 + 1116 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 3 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 2 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 1 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  gp0  = ss0 + 633 ;
   double*  gp1  = ss0 + 678 ;
   double*  fd0  = ss0 + 768 ;
   double*  fd1  = ss0 + 828 ;
   double*  ff0  = ss0 + 888 ;
   double*  hp0  = ss0 + 988 ;
   double*  hp1  = ss0 + 1051 ;
   double*  gd0  = ss0 + 1177 ;
   double*  gd1  = ss0 + 1267 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  hs1  = ss0 + 301 ;
   double*  is0  = ss0 + 406 ;
   double*  sp3  = ss0 + 546 ;
   double*  pp2  = ss0 + 549 ;
   double*  pd2  = ss0 + 567 ;
   double*  dp1  = ss0 + 585 ;
   double*  dd1  = ss0 + 639 ;
   double*  fp0  = ss0 + 711 ;
   double*  fp1  = ss0 + 741 ;
   double*  df1  = ss0 + 831 ;
   double*  gp0  = ss0 + 891 ;
   double*  gp1  = ss0 + 936 ;
   double*  fd0  = ss0 + 1071 ;
   double*  fd1  = ss0 + 1131 ;
   double*  ff0  = ss0 + 1251 ;
   double*  ff1  = ss0 + 1351 ;
   double*  hp0  = ss0 + 1451 ;
   double*  hp1  = ss0 + 1514 ;
   double*  gd0  = ss0 + 1703 ;
   double*  gd1  = ss0 + 1793 ;
   double*  fg0  = ss0 + 1973 ;
   double*  ip0  = ss0 + 2123 ;
   double*  gf0  = ss0 + 2459 ;
   double*  gf1  = ss0 + 2609 ;
   double*  hd0  = ss0 + 2759 ;
   double*  hd1  = ss0 + 2885 ;
   double*  gg0  = ss0 + 3137 ;
   double*  hf0  = ss0 + 3362 ;
   double*  hf1  = ss0 + 3572 ;
   double*  id0  = ss0 + 3782 ;
   double*  hg0  = ss0 + 4286 ;
   double*  if0  = ss0 + 4601 ;
   double*  ig0  = ss0 + 5161 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 5 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 4 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 3 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 2 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 1 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  hs1  = ss0 + 231 ;
   double*  is0  = ss0 + 294 ;
   double*  dp1  = ss0 + 378 ;
   double*  fp0  = ss0 + 396 ;
   double*  fp1  = ss0 + 426 ;
   double*  fd0  = ss0 + 456 ;
   double*  gp0  = ss0 + 516 ;
   double*  gp1  = ss0 + 561 ;
   double*  gd0  = ss0 + 606 ;
   double*  hp0  = ss0 + 696 ;
   double*  hp1  = ss0 + 759 ;
   double*  hd0  = ss0 + 822 ;
   double*  ip0  = ss0 + 948 ;
   double*  id0  = ss0 + 1116 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 3 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 2 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 1 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  gp0  = ss0 + 633 ;
   double*  gp1  = ss0 + 678 ;
   double*  fd0  = ss0 + 768 ;
   double*  fd1  = ss0 + 828 ;
   double*  ff0  = ss0 + 888 ;
   double*  hp0  = ss0 + 988 ;
   double*  hp1  = ss0 + 1051 ;
   double*  gd0  = ss0 + 1177 ;
   double*  gd1  = ss0 + 1267 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  hs1  = ss0 + 301 ;
   double*  is0  = ss0 + 406 ;
   double*  sp3  = ss0 + 546 ;
   double*  pp2  = ss0 + 549 ;
   double*  pd2  = ss0 + 567 ;
   double*  dp1  = ss0 + 585 ;
   double*  dd1  = ss0 + 639 ;
   double*  fp0  = ss0 + 711 ;
   double*  fp1  = ss0 + 741 ;
   double*  df1  = ss0 + 831 ;
   double*  gp0  = ss0 + 891 ;
   double*  gp1  = ss0 + 936 ;
   double*  fd0  = ss0 + 1071 ;
   double*  fd1  = ss0 + 1131 ;
   double*  ff0  = ss0 + 1251 ;
   double*  ff1  = ss0 + 1351 ;
   double*  hp0  = ss0 + 1451 ;
   double*  hp1  = ss0 + 1514 ;
   double*  gd0  = ss0 + 1703 ;
   double*  gd1  = ss0 + 1793 ;
   double*  fg0  = ss0 + 1973 ;
   double*  ip0  = ss0 + 2123 ;
   double*  gf0  = ss0 + 2459 ;
   double*  gf1  = ss0 + 2609 ;
   double*  hd0  = ss0 + 2759 ;
   double*  hd1  = ss0 + 2885 ;
   double*  gg0  = ss0 + 3137 ;
   double*  hf0  = ss0 + 3362 ;
   double*  hf1  = ss0 + 3572 ;
   double*  id0  = ss0 + 3782 ;
   double*  hg0  = ss0 + 4286 ;
   double*  if0  = ss0 + 4601 ;
   double*  ig0  = ss0 + 5161 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 5 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 4 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 3 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 2 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 1 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps2  = ss0 + 18 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  hs1  = ss0 + 336 ;
   double*  is0  = ss0 + 462 ;
   double*  sp3  = ss0 + 630 ;
   double*  sd3  = ss0 + 636 ;
   double*  pp2  = ss0 + 642 ;
   double*  pd2  = ss0 + 669 ;
   double*  dp1  = ss0 + 705 ;
   double*  pf2  = ss0 + 777 ;
   double*  dd1  = ss0 + 807 ;
   double*  fp0  = ss0 + 915 ;
   double*  fp1  = ss0 + 945 ;
   double*  df1  = ss0 + 1065 ;
   double*  gp0  = ss0 + 1185 ;
   double*  gp1  = ss0 + 1230 ;
   double*  fd0  = ss0 + 1410 ;
   double*  fd1  = ss0 + 1470 ;
   double*  dg1  = ss0 + 1650 ;
   double*  gd0  = ss0 + 1740 ;
   double*  gd1  = ss0 + 1830 ;
   double*  ff0  = ss0 + 2100 ;
   double*  ff1  = ss0 + 2200 ;
   double*  hp0  = ss0 + 2400 ;
   double*  hp1  = ss0 + 2463 ;
   double*  hd0  = ss0 + 2715 ;
   double*  hd1  = ss0 + 2841 ;
   double*  gf0  = ss0 + 3219 ;
   double*  gf1  = ss0 + 3369 ;
   double*  fg0  = ss0 + 3669 ;
   double*  fg1  = ss0 + 3819 ;
   double*  ip0  = ss0 + 3969 ;
   double*  id0  = ss0 + 4389 ;
   double*  gg0  = ss0 + 5061 ;
   double*  gg1  = ss0 + 5286 ;
   double*  hf0  = ss0 + 5511 ;
   double*  hf1  = ss0 + 5721 ;
   double*  fh0  = ss0 + 6141 ;
   double*  if0  = ss0 + 6351 ;
   double*  hg0  = ss0 + 7191 ;
   double*  hg1  = ss0 + 7506 ;
   double*  gh0  = ss0 + 7821 ;
   double*  ig0  = ss0 + 8136 ;
   double*  hh0  = ss0 + 8976 ;
   double*  ih0  = ss0 + 9417 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 6 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 5 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 4 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 3 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 2 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 5, 1 >( ih0 , ig0 , if0 , hg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  gp0  = ss0 + 633 ;
   double*  gp1  = ss0 + 678 ;
   double*  fd0  = ss0 + 768 ;
   double*  fd1  = ss0 + 828 ;
   double*  ff0  = ss0 + 888 ;
   double*  hp0  = ss0 + 988 ;
   double*  hp1  = ss0 + 1051 ;
   double*  gd0  = ss0 + 1177 ;
   double*  gd1  = ss0 + 1267 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  hs1  = ss0 + 301 ;
   double*  is0  = ss0 + 406 ;
   double*  sp3  = ss0 + 546 ;
   double*  pp2  = ss0 + 549 ;
   double*  pd2  = ss0 + 567 ;
   double*  dp1  = ss0 + 585 ;
   double*  fp0  = ss0 + 639 ;
   double*  fp1  = ss0 + 669 ;
   double*  dd1  = ss0 + 759 ;
   double*  df1  = ss0 + 831 ;
   double*  gp0  = ss0 + 891 ;
   double*  gp1  = ss0 + 936 ;
   double*  fd0  = ss0 + 1071 ;
   double*  fd1  = ss0 + 1131 ;
   double*  ff0  = ss0 + 1251 ;
   double*  ff1  = ss0 + 1351 ;
   double*  hp0  = ss0 + 1451 ;
   double*  hp1  = ss0 + 1514 ;
   double*  gd0  = ss0 + 1703 ;
   double*  gd1  = ss0 + 1793 ;
   double*  fg0  = ss0 + 1973 ;
   double*  ip0  = ss0 + 2123 ;
   double*  gf0  = ss0 + 2459 ;
   double*  gf1  = ss0 + 2609 ;
   double*  hd0  = ss0 + 2759 ;
   double*  hd1  = ss0 + 2885 ;
   double*  gg0  = ss0 + 3137 ;
   double*  hf0  = ss0 + 3362 ;
   double*  hf1  = ss0 + 3572 ;
   double*  id0  = ss0 + 3782 ;
   double*  hg0  = ss0 + 4286 ;
   double*  if0  = ss0 + 4601 ;
   double*  ig0  = ss0 + 5161 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 5 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 4 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 3 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 2 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 1 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps2  = ss0 + 18 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  hs1  = ss0 + 336 ;
   double*  is0  = ss0 + 462 ;
   double*  sp3  = ss0 + 630 ;
   double*  sd3  = ss0 + 636 ;
   double*  pp2  = ss0 + 642 ;
   double*  pd2  = ss0 + 669 ;
   double*  dp1  = ss0 + 705 ;
   double*  pf2  = ss0 + 777 ;
   double*  dd1  = ss0 + 807 ;
   double*  fp0  = ss0 + 915 ;
   double*  fp1  = ss0 + 945 ;
   double*  df1  = ss0 + 1065 ;
   double*  gp0  = ss0 + 1185 ;
   double*  gp1  = ss0 + 1230 ;
   double*  fd0  = ss0 + 1410 ;
   double*  fd1  = ss0 + 1470 ;
   double*  dg1  = ss0 + 1650 ;
   double*  gd0  = ss0 + 1740 ;
   double*  gd1  = ss0 + 1830 ;
   double*  ff0  = ss0 + 2100 ;
   double*  ff1  = ss0 + 2200 ;
   double*  hp0  = ss0 + 2400 ;
   double*  hp1  = ss0 + 2463 ;
   double*  hd0  = ss0 + 2715 ;
   double*  hd1  = ss0 + 2841 ;
   double*  ip0  = ss0 + 3219 ;
   double*  gf0  = ss0 + 3639 ;
   double*  gf1  = ss0 + 3789 ;
   double*  fg0  = ss0 + 4089 ;
   double*  fg1  = ss0 + 4239 ;
   double*  id0  = ss0 + 4389 ;
   double*  hf0  = ss0 + 5061 ;
   double*  hf1  = ss0 + 5271 ;
   double*  gg0  = ss0 + 5691 ;
   double*  gg1  = ss0 + 5916 ;
   double*  fh0  = ss0 + 6141 ;
   double*  if0  = ss0 + 6351 ;
   double*  hg0  = ss0 + 7191 ;
   double*  hg1  = ss0 + 7506 ;
   double*  gh0  = ss0 + 7821 ;
   double*  ig0  = ss0 + 8136 ;
   double*  hh0  = ss0 + 8976 ;
   double*  ih0  = ss0 + 9417 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 6 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 5 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 4 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 3 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 2 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 5, 1 >( ih0 , ig0 , if0 , hg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 13 ;
   double*  ps2  = ss0 + 19 ;
   double*  ds0  = ss0 + 49 ;
   double*  ds1  = ss0 + 55 ;
   double*  fs0  = ss0 + 115 ;
   double*  fs1  = ss0 + 125 ;
   double*  gs0  = ss0 + 215 ;
   double*  gs1  = ss0 + 230 ;
   double*  hs0  = ss0 + 350 ;
   double*  hs1  = ss0 + 371 ;
   double*  is0  = ss0 + 518 ;
   double*  sp3  = ss0 + 714 ;
   double*  sd3  = ss0 + 723 ;
   double*  pp2  = ss0 + 735 ;
   double*  sf3  = ss0 + 771 ;
   double*  pd2  = ss0 + 781 ;
   double*  dp1  = ss0 + 835 ;
   double*  pf2  = ss0 + 925 ;
   double*  dd1  = ss0 + 985 ;
   double*  fp0  = ss0 + 1129 ;
   double*  fp1  = ss0 + 1159 ;
   double*  pg2  = ss0 + 1309 ;
   double*  df1  = ss0 + 1354 ;
   double*  fd0  = ss0 + 1534 ;
   double*  fd1  = ss0 + 1594 ;
   double*  gp0  = ss0 + 1834 ;
   double*  gp1  = ss0 + 1879 ;
   double*  dg1  = ss0 + 2104 ;
   double*  ff0  = ss0 + 2284 ;
   double*  ff1  = ss0 + 2384 ;
   double*  hp0  = ss0 + 2684 ;
   double*  hp1  = ss0 + 2747 ;
   double*  gd0  = ss0 + 3062 ;
   double*  gd1  = ss0 + 3152 ;
   double*  dh1  = ss0 + 3512 ;
   double*  hd0  = ss0 + 3638 ;
   double*  hd1  = ss0 + 3764 ;
   double*  gf0  = ss0 + 4268 ;
   double*  gf1  = ss0 + 4418 ;
   double*  fg0  = ss0 + 4868 ;
   double*  fg1  = ss0 + 5018 ;
   double*  ip0  = ss0 + 5318 ;
   double*  fh0  = ss0 + 5822 ;
   double*  fh1  = ss0 + 6032 ;
   double*  id0  = ss0 + 6242 ;
   double*  hf0  = ss0 + 7082 ;
   double*  hf1  = ss0 + 7292 ;
   double*  gg0  = ss0 + 7922 ;
   double*  gg1  = ss0 + 8147 ;
   double*  gh0  = ss0 + 8597 ;
   double*  gh1  = ss0 + 8912 ;
   double*  fi0  = ss0 + 9227 ;
   double*  if0  = ss0 + 9507 ;
   double*  hg0  = ss0 + 10627 ;
   double*  hg1  = ss0 + 10942 ;
   double*  gi0  = ss0 + 11572 ;
   double*  ig0  = ss0 + 11992 ;
   double*  hh0  = ss0 + 13252 ;
   double*  hh1  = ss0 + 13693 ;
   double*  ih0  = ss0 + 14134 ;
   double*  hi0  = ss0 + 15310 ;
   double*  ii0  = ss0 + 15898 ;
   execute_VRR1_gpu< VTS, 1, 0, 12 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 11 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 10 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 9 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 8 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 7 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 6 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 5 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 6 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 5 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 4 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 4 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 3 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 4, 3 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 5, 2 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 5, 2 >( ih0 , ig0 , if0 , hg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 6, 1 >( hi0 , hh0 , hg0 , gh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 6, 1 >( ii0 , ih0 , ig0 , hh1 , QC_WQ, zcd );
}
















































#include <vector>

template < int VTS >
__device__ constexpr auto get_vrr_kernel(int Lindex ){
   void (*(vrr_kernels[4*4*4*4]))(double *, double*) = { 
      ssss<VTS>,sssp<VTS>,sssd<VTS>,sssf<VTS>,ssps<VTS>,sspp<VTS>,sspd<VTS>,sspf<VTS>,ssds<VTS>,ssdp<VTS>,ssdd<VTS>,ssdf<VTS>,ssfs<VTS>,ssfp<VTS>,ssfd<VTS>,ssff<VTS>,
      spss<VTS>,spsp<VTS>,spsd<VTS>,spsf<VTS>,spps<VTS>,sppp<VTS>,sppd<VTS>,sppf<VTS>,spds<VTS>,spdp<VTS>,spdd<VTS>,spdf<VTS>,spfs<VTS>,spfp<VTS>,spfd<VTS>,spff<VTS>,
      sdss<VTS>,sdsp<VTS>,sdsd<VTS>,sdsf<VTS>,sdps<VTS>,sdpp<VTS>,sdpd<VTS>,sdpf<VTS>,sdds<VTS>,sddp<VTS>,sddd<VTS>,sddf<VTS>,sdfs<VTS>,sdfp<VTS>,sdfd<VTS>,sdff<VTS>,
      sfss<VTS>,sfsp<VTS>,sfsd<VTS>,sfsf<VTS>,sfps<VTS>,sfpp<VTS>,sfpd<VTS>,sfpf<VTS>,sfds<VTS>,sfdp<VTS>,sfdd<VTS>,sfdf<VTS>,sffs<VTS>,sffp<VTS>,sffd<VTS>,sfff<VTS>,
      psss<VTS>,pssp<VTS>,pssd<VTS>,pssf<VTS>,psps<VTS>,pspp<VTS>,pspd<VTS>,pspf<VTS>,psds<VTS>,psdp<VTS>,psdd<VTS>,psdf<VTS>,psfs<VTS>,psfp<VTS>,psfd<VTS>,psff<VTS>,
      ppss<VTS>,ppsp<VTS>,ppsd<VTS>,ppsf<VTS>,ppps<VTS>,pppp<VTS>,pppd<VTS>,pppf<VTS>,ppds<VTS>,ppdp<VTS>,ppdd<VTS>,ppdf<VTS>,ppfs<VTS>,ppfp<VTS>,ppfd<VTS>,ppff<VTS>,
      pdss<VTS>,pdsp<VTS>,pdsd<VTS>,pdsf<VTS>,pdps<VTS>,pdpp<VTS>,pdpd<VTS>,pdpf<VTS>,pdds<VTS>,pddp<VTS>,pddd<VTS>,pddf<VTS>,pdfs<VTS>,pdfp<VTS>,pdfd<VTS>,pdff<VTS>,
      pfss<VTS>,pfsp<VTS>,pfsd<VTS>,pfsf<VTS>,pfps<VTS>,pfpp<VTS>,pfpd<VTS>,pfpf<VTS>,pfds<VTS>,pfdp<VTS>,pfdd<VTS>,pfdf<VTS>,pffs<VTS>,pffp<VTS>,pffd<VTS>,pfff<VTS>,
      dsss<VTS>,dssp<VTS>,dssd<VTS>,dssf<VTS>,dsps<VTS>,dspp<VTS>,dspd<VTS>,dspf<VTS>,dsds<VTS>,dsdp<VTS>,dsdd<VTS>,dsdf<VTS>,dsfs<VTS>,dsfp<VTS>,dsfd<VTS>,dsff<VTS>,
      dpss<VTS>,dpsp<VTS>,dpsd<VTS>,dpsf<VTS>,dpps<VTS>,dppp<VTS>,dppd<VTS>,dppf<VTS>,dpds<VTS>,dpdp<VTS>,dpdd<VTS>,dpdf<VTS>,dpfs<VTS>,dpfp<VTS>,dpfd<VTS>,dpff<VTS>,
      ddss<VTS>,ddsp<VTS>,ddsd<VTS>,ddsf<VTS>,ddps<VTS>,ddpp<VTS>,ddpd<VTS>,ddpf<VTS>,ddds<VTS>,dddp<VTS>,dddd<VTS>,dddf<VTS>,ddfs<VTS>,ddfp<VTS>,ddfd<VTS>,ddff<VTS>,
      dfss<VTS>,dfsp<VTS>,dfsd<VTS>,dfsf<VTS>,dfps<VTS>,dfpp<VTS>,dfpd<VTS>,dfpf<VTS>,dfds<VTS>,dfdp<VTS>,dfdd<VTS>,dfdf<VTS>,dffs<VTS>,dffp<VTS>,dffd<VTS>,dfff<VTS>,
      fsss<VTS>,fssp<VTS>,fssd<VTS>,fssf<VTS>,fsps<VTS>,fspp<VTS>,fspd<VTS>,fspf<VTS>,fsds<VTS>,fsdp<VTS>,fsdd<VTS>,fsdf<VTS>,fsfs<VTS>,fsfp<VTS>,fsfd<VTS>,fsff<VTS>,
      fpss<VTS>,fpsp<VTS>,fpsd<VTS>,fpsf<VTS>,fpps<VTS>,fppp<VTS>,fppd<VTS>,fppf<VTS>,fpds<VTS>,fpdp<VTS>,fpdd<VTS>,fpdf<VTS>,fpfs<VTS>,fpfp<VTS>,fpfd<VTS>,fpff<VTS>,
      fdss<VTS>,fdsp<VTS>,fdsd<VTS>,fdsf<VTS>,fdps<VTS>,fdpp<VTS>,fdpd<VTS>,fdpf<VTS>,fdds<VTS>,fddp<VTS>,fddd<VTS>,fddf<VTS>,fdfs<VTS>,fdfp<VTS>,fdfd<VTS>,fdff<VTS>,
      ffss<VTS>,ffsp<VTS>,ffsd<VTS>,ffsf<VTS>,ffps<VTS>,ffpp<VTS>,ffpd<VTS>,ffpf<VTS>,ffds<VTS>,ffdp<VTS>,ffdd<VTS>,ffdf<VTS>,fffs<VTS>,fffp<VTS>,fffd<VTS>,ffff<VTS>,
   };
   return vrr_kernels[Lindex];
}

// protoype of a v3 with templated global call

template< int NVT, int VTS, int la, int lb, int lc, int ld >
__global__ void compute_VRR_v3_batched_gpu_low(
      const int Ncells, 
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int numV, int numVC, const int Ng ){

   constexpr int L = la+lb+lc+ld;
   constexpr int Li = 64*la+16*lb+4*lc+ld;
   constexpr int F_size = Fsize_but_constexpr(L);
                         
   constexpr auto vrr_kernel = get_vrr_kernel<VTS>(Li);

   int my_vrr_rank = threadIdx.x % VTS ;
   int my_vrr_team = threadIdx.x / VTS ;

   for( int block=blockIdx.x; block < Ncells*Ng ; block += gridDim.x ){

      int p = block / Ng; 
      int n3 = block % Ng;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];

      for ( unsigned i = my_vrr_team; i < n_prm ;  i += NVT ){

//         // Screening on the (ab.n1|cd.n2@n3) fondamental integrals
         bool found = false;
         unsigned int Of = 0;

         double * pr_mem, * pqz;
         __shared__ double PQZ[ NVT * 17 ];

         while ( not found and i < n_prm ){
            Of = ((Ov+i) * Ng + n3 ) * F_size;
            // copy Fm[0] ( the ssss(0) integral ) to AC for later screening in ECO
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            pr_mem[0] = Fm[Of];
            // Immediate screening
            if (Fm[Of] > -1+1.e-30 ){ found = true ; }
            else { i += NVT; }
         }

         if ( found and i < n_prm ){ 

            // Copy the sss(m) integrals
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            for( int il=0; il < L+1; il++ ){ pr_mem[il] = Fm[Of+il]; }
   
            // Copy PA WP QC WQ z1-5 to shared memory for each team
            for( int ii = my_vrr_rank; ii < 17 ; ii += VTS ){ PQZ[my_vrr_team*17+ii] = Fm[Of+L+1+ii]; }
            pqz = &PQZ[my_vrr_team*17];
         }

         __syncthreads();


         if ( found and i < n_prm ){ 
//            if ( my_vrr_rank == 0 ){ 
//               printf(" Tx %d Bx %d | Computing p %d %p \n", threadIdx.x, blockIdx.x, i, pr_mem );
//            }


            vrr_kernel( pr_mem, pqz );
         }
      }
   }   
}







// defines a new type called vrr_global_t
// this type is actually a (global) function, that takes that list 
// as arguments and returns void
typedef void (*(vrr_global_t))( const int , 
      const unsigned int* const __restrict__ ,
      const unsigned int* const __restrict__ ,
      const double* const __restrict__ ,
      const double* const __restrict__ ,
      double* const __restrict__ ,
      double* const __restrict__ ,
      int , int , int , int , const int  ) ;

vrr_global_t get_vrr( int vrr_index ){
   vrr_global_t vrr[4*4*4*4] = {
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 3 >
      };
   return vrr[vrr_index];
}

void compute_VRR_v3(
      const int Ncells, 
      const int vrr_index,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int numV, int numVC, const int Ng, hipStream_t hip_stream ){
   get_vrr( vrr_index )<<<Ncells*Ng, 64, 0, hip_stream >>>( Ncells, PMX,FVH,Fm,data,AC,ABCD,vrr_blocksize,hrr_blocksize,numV,numVC,Ng );
}





__global__ void compute_VRR_v2_batched_gpu_low(
      const int Ncells, const int vrr_kernel_index,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){
   
   int F_size = Fsize(L);

   constexpr int VTS = 16;
   constexpr int NVT =  4;

   auto vrr_kernel = get_vrr_kernel<VTS>(vrr_kernel_index);

   int my_vrr_rank = threadIdx.x % VTS ;
   int my_vrr_team = threadIdx.x / VTS ;
 
   for( int block=blockIdx.x; block < Ncells*Ng ; block += gridDim.x ){

      unsigned int p = block / Ng; 
      int n3 = block % Ng;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];

      for ( unsigned i = my_vrr_team; i < n_prm ;  i += NVT ){

         // Screening on the (ab.n1|cd.n2@n3) fondamental integrals
         bool found = false;
         unsigned int Of = 0;

         double * pr_mem, * pqz;
         __shared__ double PQZ[ NVT * 17 ];

         while ( not found and i < n_prm ){
            Of = ((Ov+i) * Ng + n3 ) * F_size;
            // copy Fm[0] ( the ssss(0) integral ) to AC for later screening in ECO
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            pr_mem[0] = Fm[Of];
            // Immediate screening
            if (Fm[Of] > 1.e-24 ){ found = true ; }
            else { i += NVT; }
         }

         if ( found and i < n_prm ){ 

            // Copy the sss(m) integrals
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            for( int il=0; il < L+1; il++ ){ pr_mem[il] = Fm[Of+il]; }
   
            // Copy PA WP QC WQ z1-5 to shared memory for each team
            for( int ii = my_vrr_rank; ii < 17 ; ii += VTS ){ PQZ[my_vrr_team*17+ii] = Fm[Of+L+1+ii]; }
            pqz = &PQZ[my_vrr_team*17];
         }

         __syncthreads();

         if ( found and i < n_prm ){

            vrr_kernel( pr_mem, pqz );
         }
//         __syncthreads();
      }
   }
}

