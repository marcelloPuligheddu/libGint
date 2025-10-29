#include <iostream>
#include "compute_VRR1_omp.h"
#include "util.h"
#include "define.h"
#include <omp.h>
using std::cout;
using std::endl;

// Given an angular momentum L and linear index i (0 <= i < N),
// where N = (L+1)(L+2)/2, this function maps i to the corresponding
// integer triplet (lx, ly, lz)
//
// Implementation note:
// Each "row" in the enumeration begins at a triangular number T_r = r(r+1)/2.
// So to find which row i belongs to, we need the largest r such that T_r <= i.
// We do this by solving r(r+1)/2 <= i for r:
//       r2 + r - 2i <= 0 
//       -1/2 sqrt(8 i + 1) - 1/2<= r <=1/2 sqrt(8 i + 1) - 1/2
//       So the largest (int) r is the one we are after
//       r = floor( (sqrt(8*i + 1) - 1) / 2 )
void xyz( const int L, const int i, int * lx, int * ly, int * lz ){

    int row = static_cast<int>(floor((sqrt(8.0 * i + 1) - 1) / 2));
    int row_start = row * (row + 1) / 2;
    int col = i - row_start;
    *lx = L - row;
    *ly = row - col;
    *lz = col;
}

// Simplest VRR, computes psss(m) = PA ssss(m) + WP ssss(m+1)
void execute_VRR1_gpu( // int AL, int CL, int m, 
      const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double PA_WP[6] ){

   for( int imm=0; imm < 3*m; imm++ ){

      int mm = imm / 3 ;
      int i  = imm % 3 ;

      int d  = i ;
      int im = 0 ; 

      int idx_000 = i  + 3*mm;
      int idx_m00 = im + 1*mm;

      a0c0m0[idx_000] = PA_WP[d] * amc0m0[idx_m00] + PA_WP[d+3] * amc0m0[1+idx_m00];
   }
}

// TODO template AL
// VRR of the type nsss(m) = PA ksss(m) + WP ksss(m+1) + e ( k1 wsss(m) + k2 wssss(m+1) )
// where k = n-1, w = n-2
void execute_VRR2_gpu(
      const int AL, const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ amc0mp,
      const double* __restrict__ awc0m0,
      const double* __restrict__ awc0mp,
      const double PA[3], const double WP[3], const double inv_2zab, const double min_rho_zab2 ){
 

   const int NcoA   = (AL+1)*(AL+2)/2;
   const int NcoAxx = (AL+0)*(AL-1)/2;
   const int NcoAx  = (AL+1)*(AL+0)/2;
   const int NcoAyy = (AL+3)*(AL+0)/2 - 1;
   const int NcoAy  = (AL+1)*(AL+2)/2 - 2 ;
   const int NcoAm  = (AL+1)*(AL+0)/2;
   const int NcoAw  = (AL-1)*(AL+0)/2;

   for ( int imm = 0; imm < m*NcoA; imm++ ){

      int i  = imm % NcoA ;
      int mm = imm / NcoA ;
      int d, im, iw, e2, ex,ey,ez ;

      xyz( AL, i, &ex,&ey,&ez );

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


// more general case, for al>0 and cl>1
void execute_VRR5_gpu(
      const int AL, const int CL, const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ a0cwm0,
      const double* __restrict__ amcmmp,
      const double QC_WQ[6],
      const double inv_2zcd[3] ){

   const int NcoA = NLco_dev(AL);
   const int NcoAm= NLco_dev(AL-1);
   const int NcoC = NLco_dev(CL);
   const int NcoCm = NLco_dev(CL-1);
   const int NcoCw = NLco_dev(CL-2);
   const int NcoCy = (CL+3)*CL/2-1;
   // k is the faster variable, followed by i, then m
   for ( int mik=0; mik < NcoA*NcoC*m; mik++ ){
      int ex, ey, ez, fx, fy, fz, f2, e2;
      int i, k, mm, d, km, kw, im;
      int idx_000, idx_0m0, idx_0w0, idx_mmp;
      double i_0m0, i_0mp, i_0w0, i_0wp, i_mmp;

      k  = mik % NcoC;
      i  = (mik/NcoC)%NcoA;
      mm = mik/NcoC/NcoA;

      xyz(AL,i,&ex,&ey,&ez);
      xyz(CL,k,&fx,&fy,&fz);

      // if the x moment of k is more than zero, than apply the vrr along x
      if (k < NcoCw){ 
         d  = 0;
         km = k;
         kw = k;
         f2 = fx-1;
         im = i;
         e2 = ex;
      // x[....]
      } else if (k < NcoCm){
         d  = 0;
         km = k;
         kw = k; // not used since f2 = fx-1 = 0 
         f2 = fx-1; // == 0
         im = i;
         e2 = ex;
      // yy[...]
      } else if (k < NcoCy){
         d  = 1;
         km = k - CL; // k - (fy+fz) = k - (CL-fx) = k - CL
         kw = k - 2 * CL + 1;
         f2 = fy-1;
         im = i - (ey+ez); // we need the general version since ex may not be zero
         e2 = ey;
      // y[...]
      } else if (k == NcoC - 2){
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

      idx_000 = mik ; // k +i *NcoC + NcoA *NcoC *(mm)
      idx_0m0 = km+i *NcoCm + NcoA *NcoCm*mm;
      idx_0w0 = kw+i *NcoCw + NcoA *NcoCw*mm;
      idx_mmp = km+im*NcoCm + NcoAm*NcoCm*mm;

      i_0m0 = a0cmm0[ idx_0m0 ];
      i_0mp = a0cmm0[ idx_0m0 + NcoA *NcoCm ];
      i_0w0 = a0cwm0[ idx_0w0 ];
      i_0wp = a0cwm0[ idx_0w0 + NcoA *NcoCw ];
      i_mmp = amcmmp[ idx_mmp ];

      a0c0m0[ idx_000 ] = QC_WQ[d] * i_0m0 + QC_WQ[d+3] * i_0mp + f2*( inv_2zcd[0] * i_0w0 + inv_2zcd[1] * i_0wp ) + e2*inv_2zcd[2]*i_mmp;
   }
}


// case for a>0, c=1
void execute_VRR6_gpu( 
      const int AL, const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ a0cmmp,
      const double* __restrict__ amcmmp,
      const double QC[3], const double WQ[3], const double inv_2z ){
     
   const int NcoA = NLco_dev(AL);
   const int NcoAm= NLco_dev(AL-1);
   const int NcoC = 3 ; // NLco_dev(CL);
   // k is the faster variable, followed by i, then m
   for ( int mik=0; mik < NcoA*NcoC*m; mik++ ){

      int i, k, mm, d, im, e2, ex,ey,ez ;
      int idx_000, idx_0m0, idx_mmp;
      double i_0m0, i_0mp, i_mmp;

      k  = mik % NcoC;
      i  = (mik/NcoC)%NcoA;
      mm = mik/NcoC/NcoA;

      xyz(AL,i,&ex,&ey,&ez);

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

void compute_VRR1_omp(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){
   
   int F_size = Fsize(L);

   #pragma omp target teams distribute is_device_ptr(PMX,FVH,Fm,data,AC) depend( in:PMX,FVH,Fm,data ) depend( out:AC ) 
   for( int block=0; block < Ncells*Ng ; block ++ ){

      int p = block / Ng; 
      int n3 = block % Ng;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];
      double PQZ[ 17 ];

//      #pragma omp parallel for 
      for ( unsigned prm = 0; prm < n_prm;  prm ++ ){

         int i = prm ;
         unsigned int Of = ((Ov+i) * Ng + n3 ) * F_size;
         double * pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
         pr_mem[0] = Fm[Of];

         // Copy the sss(m) integrals
         pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
         for( int il=0; il < L+1; il++ ){ pr_mem[il] = Fm[Of+il]; }

         // Copy PA WP QC WQ z1-5
         const double * pqz = &Fm[Of+L+1]; 

         for ( int op=0; op < numVC; op++ ){

            const int t  = plan[ op*OP_SIZE + T__OFFSET ];
            const int la = plan[ op*OP_SIZE + LA_OFFSET ];
            const int lc = plan[ op*OP_SIZE + LC_OFFSET ];
            const int max_m  = plan[ op*OP_SIZE + M__OFFSET ];
            const int min_m  = plan[ op*OP_SIZE + H__OFFSET ];
            const int m = max_m - min_m + 1 ; 
            const int off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
            const int off_m2 = plan[ op*OP_SIZE + M2_OFFSET ];
            const int off_m3 = plan[ op*OP_SIZE + M3_OFFSET ];
            const int off_m4 = plan[ op*OP_SIZE + M4_OFFSET ];
            const int off_m5 = plan[ op*OP_SIZE + M5_OFFSET ];
            const int off_m6 = plan[ op*OP_SIZE + M6_OFFSET ];
            double* m1 = &pr_mem[off_m1];
            double* m2 = &pr_mem[off_m2];
            double* m3 = &pr_mem[off_m3];
            double *m4, *m5, *m6 ;

            // TODO is it worth it to unify the VRRs types as a single, maybe individually less efficient, call ?
            if ( t == VRR1 ){
               execute_VRR1_gpu( m, m1, m2, pqz+0 );
            } else if ( t == VRR2 ){
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5]; 
               execute_VRR2_gpu( la, m, m1, m2, m3, m4, m5, pqz+0, pqz+3, pqz[12], pqz[13] );
            } else if ( t == VRR3 ){ 
               execute_VRR1_gpu( m, m1, m2, pqz+6 );
            } else if ( t == VRR4 ){ 
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5];
               execute_VRR2_gpu( lc, m, m1, m2, m3, m4, m5, pqz+6, pqz+9, pqz[14], pqz[15] );
            } else if ( t == VRR5 ){ 
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5]; m6 = &pr_mem[off_m6];
               execute_VRR5_gpu( la, lc, m, m1, m2, m4, m6, pqz+6, pqz+14 );
            } else if ( t == VRR6 ){ 
               m4 = &pr_mem[off_m4];
               execute_VRR6_gpu( la, m, m1, m2, m3, m4, pqz+6, pqz+9, pqz[16] );
            } // end of loop over op  
         }
      }
   }
}



