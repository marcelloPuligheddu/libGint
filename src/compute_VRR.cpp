#include <iostream>
#include "compute_VRR.h"
#include "util.h"
#include "define.h"

using std::cout;
using std::endl;


__device__ __host__ void execute_VRR1( // int AL, int CL, int m, 
      const int m,
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ amc0mp,
      const double PA[3], const double WP[3], const int my_vrr_rank=0, const int vrr_team_size=1 ){

   for( int imm=my_vrr_rank; imm < 3*m; imm+=vrr_team_size ){

      int mm = imm / 3 ;
      int i  = imm % 3 ;

      int d  = i ;
      int im = 0 ; 

      int idx_000 = i  + 3*mm;
      int idx_m00 = im + 1*mm;

      a0c0m0[idx_000] = PA[d] * amc0m0[idx_m00] + WP[d] * amc0mp[idx_m00];
   }
}


// Specialization of VRR for a>1, c=0. Applies
// <a+1i,sss>(m) = PA_i <asss>(m) + WP_i <asss>(m+1) 
//               + 1/2zab N_i(a) ( <a-1,sss>(m) - rho/zab <a-1,sss>(m+1) )
__device__ __host__ void execute_VRR2(
      const int AL, const int m,
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

      int ex = lx(i,AL);
      int ey = ly(i,AL);
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

// more general case, for al>0 and cl>1
__device__ __host__ void execute_VRR5(
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
   //
   const int NcoA = NLco_dev(AL);
   const int NcoAm= NLco_dev(AL-1);
   const int NcoC = NLco_dev(CL);
   const int NcoCm = NLco_dev(CL-1);
   const int NcoCw = NLco_dev(CL-2);
   const int NcoCy = (CL+3)*CL/2-1;
   // k is the faster variable, followed by i, then m
   for ( int mik=my_vrr_rank; mik < NcoA*NcoC*m; mik+=vrr_team_size ){
      int ex, ey, ez, fx, fy, fz;
      int i, k, mm, d, km, kw, f2, im, e2 ;
      int idx_000, idx_0m0, idx_0w0, idx_mmp;
      double i_0m0, i_0mp, i_0w0, i_0wp, i_mmp;


      k  = mik % NcoC;
      i  = (mik/NcoC)%NcoA;
      mm = mik/NcoC/NcoA;

      ex = lx(i,AL);
      ey = ly(i,AL);
      ez = AL - ex - ey ; // lz(i,AL);
      fx = lx(k,CL);
      fy = ly(k,CL);
      fz = CL- fx - fy ; // lz(k,CL);
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
         kw = 0; // not used since f2 = fx-1 = 0 
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
      i_0mp = a0cmmp[ idx_0m0 ];
      i_0w0 = a0cwm0[ idx_0w0 ];
      i_0wp = a0cwmp[ idx_0w0 ];
      i_mmp = amcmmp[ idx_mmp ];

      a0c0m0[ idx_000 ] = QC[d] * i_0m0 + WQ[d] * i_0mp + f2*( inv_2zcd * i_0w0 + min_rho_zcd2 * i_0wp ) + e2*inv_2z*i_mmp;
   }
}

// case for a>0, c=1
__device__ __host__ void execute_VRR6( 
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

      ex = lx(i,AL);
      ey = ly(i,AL);
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


__device__ void execute_CP2S_gpu( 
      const int AL, const int CL, 
      const double* __restrict__ pr_mem,
      double* sh_mem, // sh is -very much- not restrict-ed. It is only written/read to by this block, however
      const int my_vrr_rank, const int vrr_team_size, const int hrr_blocksize,
      const unsigned int ipa, const unsigned int ipb, const unsigned int ipc, const unsigned int ipd,
      const unsigned int nla, const int unsigned nlb, const int unsigned nlc, const int unsigned nld, 
      const int unsigned nga, const int unsigned ngb, const int unsigned ngc, const int unsigned ngd,
      const double* const __restrict__ Ka, const double* const __restrict__ Kb,
      const double* const __restrict__ Kc, const double* const __restrict__ Kd ){

   // index over the contract basis sets
   double K = Ka[ ipa ] * Kb[ ipb ] * Kc[ ipc ] * Kd[ ipd ];

   const int NcoA = NLco_dev(AL);
   const int NcoC = NLco_dev(CL);
   const int NcoAC = NcoA*NcoC;
   for( int i=my_vrr_rank; i < NcoAC; i+=vrr_team_size){
      // must be block atomic 
      atomicAdd( &sh_mem[ i ] , K * pr_mem[i]);
   }

/*

   const unsigned int nl___d = nld;
   const unsigned int nl__cd = nlc*nl___d;
   const unsigned int nl_bcd = nlb*nl__cd;
   const unsigned int nlabcd = nla*nl_bcd;

   for( unsigned int ilabcd = 0; ilabcd < nlabcd; ilabcd++ ){
      unsigned int a = (ilabcd / nl_bcd) ;
      unsigned int b = (ilabcd / nl__cd) % nlb ;
      unsigned int c = (ilabcd / nl___d) % nlc ;
      unsigned int d = ilabcd % nld ;
      double K = Ka[ a*nga + ipa ] * Kb[ b*ngb + ipb ] * Kc[ c*ngc + ipc ] * Kd[ d*ngd + ipd ];

      const int NcoA = NLco_dev(AL);
      const int NcoC = NLco_dev(CL);
      const int NcoAC = NcoA*NcoC;
      for( int i=my_vrr_rank; i < NcoAC; i+=vrr_team_size){
        // must be atomic
        atomicAdd( &sh_mem[ ilabcd*hrr_blocksize + i ] , K * pr_mem[i]);
      }
   }
*/
}


void execute_CP2S_v2( 
      const int AL, const int CL, 
      const double* __restrict__ pr_mem,
      double* sh_mem, // sh is -very much- not restrict-ed. It is only written/read to by this block, however
      const int my_vrr_rank, const int vrr_team_size, const int hrr_blocksize,
      const unsigned int ipa, const unsigned int ipb, const unsigned int ipc, const unsigned int ipd,
      const unsigned int nla, const int unsigned nlb, const int unsigned nlc, const int unsigned nld, 
      const int unsigned nga, const int unsigned ngb, const int unsigned ngc, const int unsigned ngd,
      const double* const __restrict__ Ka, const double* const __restrict__ Kb,
      const double* const __restrict__ Kc, const double* const __restrict__ Kd ){

   // Early returns 
   if (nla+nlb+nlc+nld == 4 ){
      double K = Ka[ipa] * Kb[ipb] * Kc[ipc] * Kd[ipd];
      if ( AL+CL == 0 ){
         // must be atomic
//         cout << "Adding " << K << " * " << pr_mem[0] << " to " << sh_mem[ 0 ] << endl;
//         cout << "IP: " << ipa << " " << ipb << " " << ipc << " " << ipd << endl;
//         cout << Ka[ipa] << " " <<  Kb[ipb] << " " <<  Kc[ipc] << " " << Kd[ipd] << endl;
         sh_mem[ 0 ] += K * pr_mem[0];
      } else {
         const int NcoA = NLco_dev(AL);
         const int NcoC = NLco_dev(CL);
         const int NcoAC = NcoA*NcoC;
         for( int i=my_vrr_rank; i < NcoAC; i+=vrr_team_size){
            // must be atomic
            sh_mem[ i ] += K * pr_mem[i];
         }
      }
      return;
   }

   // index over the contract basis sets
   const unsigned int nl___d = nld;
   const unsigned int nl__cd = nlc*nl___d;
   const unsigned int nl_bcd = nlb*nl__cd;
   const unsigned int nlabcd = nla*nl_bcd;

   for( unsigned int ilabcd = 0; ilabcd < nlabcd; ilabcd++ ){
      unsigned int a = (ilabcd / nl_bcd) ;
      unsigned int b = (ilabcd / nl__cd) % nlb ;
      unsigned int c = (ilabcd / nl___d) % nlc ;
      unsigned int d = ilabcd % nld ;
      double K = Ka[ a*nga + ipa ] * Kb[ b*ngb + ipb ] * Kc[ c*ngc + ipc ] * Kd[ d*ngd + ipd ];

      if ( AL+CL == 0 ){
         // must be atomic 
         sh_mem[ ilabcd*hrr_blocksize ] += K * pr_mem[0];
      } else {
         const int NcoA = NLco_dev(AL);
         const int NcoC = NLco_dev(CL);
         const int NcoAC = NcoA*NcoC;
         for( int i=my_vrr_rank; i < NcoAC; i+=vrr_team_size){
            // must be atomic
            sh_mem[ ilabcd*hrr_blocksize + i ] += K * pr_mem[i];
         }
      }
   }
}


__global__ void compute_VRR_batched_gpu_low(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ Pm_input_list,
      const unsigned int* const __restrict__ HRR,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, int vrr_team_size ){

   
   const int num_vrr_teams = blockDim.x / vrr_team_size;
   int my_vrr_team = threadIdx.x / vrr_team_size;
   int my_vrr_rank = threadIdx.x % vrr_team_size;
//   assert(num_vrr_teams*vrr_team_size == blockDim.x);

   for( int block=blockIdx.x; block < Ncells ; block += gridDim.x ){

      unsigned int Ov     = HRR[block*HRR_SIZE+HRR_OFFSET_OV];
      unsigned int n_prm  = HRR[block*HRR_SIZE+HRR_OFFSET_NPRM];
      unsigned int Og     = HRR[block*HRR_SIZE+HRR_OFFSET_OG];
//        Oq     = HRR[block*HRR_SIZE+HRR_OFFSET_OQ]
//        idx_AB = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_AB]
//        idx_CD = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_CD]
      unsigned int nlabcd = HRR[block*HRR_SIZE+HRR_OFFSET_NLABCD];
      unsigned int plabcd = HRR[block*HRR_SIZE+HRR_OFFSET_PLABCD];
      unsigned int ppabcd = HRR[block*HRR_SIZE+HRR_OFFSET_PPABCD];
      unsigned int idx_Ka = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KA];
      unsigned int idx_Kb = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KB];
      unsigned int idx_Kc = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KC];
      unsigned int idx_Kd = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KD];
      unsigned int Lmax   = HRR[block*HRR_SIZE+HRR_OFFSET_LMAX];
      unsigned int Of0    = HRR[block*HRR_SIZE+HRR_OFFSET_OF0];
      unsigned int Op     = HRR[block*HRR_SIZE+HRR_OFFSET_OP];

      const double* Ka = &data[idx_Ka];
      const double* Kb = &data[idx_Kb];
      const double* Kc = &data[idx_Kc];
      const double* Kd = &data[idx_Kd];

      int F_size = Lmax+1;
      if (Lmax > 0){ F_size += 4*3+5; }
 
      unsigned int nla,nlb,nlc,nld,npa,npb,npc,npd;
      decode4( plabcd, &nla,&nlb,&nlc,&nld );
      decode4( ppabcd, &npa,&npb,&npc,&npd );
      double* sh_mem = &ABCD[ Og * hrr_blocksize ];

      for ( unsigned i= threadIdx.x; i < hrr_blocksize * nlabcd ; i+= blockDim.x ){
         sh_mem[i] = 0.0 ;
      }
      __syncthreads();

      for ( unsigned i = my_vrr_team; i < n_prm;  i += num_vrr_teams ){
         unsigned int Of   = Of0 + i * F_size;
         unsigned int ipzn = Pm_input_list[(Op + i)*PM_SIZE+PMA_OFFSET_IPZN ];
         unsigned int ipa,ipb,ipc,ipd;

         decode_ipabcd_none( ipzn, &ipa,&ipb,&ipc,&ipd );

         double* pr_mem = &AC[ (Ov + i) * vrr_blocksize ];

         for( int il=0; il < L+1; il++ ){
            pr_mem[il] = Fm[Of+il];
         }
         const double* PA = nullptr;
         const double* WP = nullptr;
         const double* QC = nullptr;
         const double* WQ = nullptr;
         double inv_2zab = 0.;
         double min_rho_zab2 = 0.;
         double inv_2zcd = 0.;
         double min_rho_zcd2 = 0.;
         double inv_2z = 0.;

         if (L > 0){
            PA = &Fm[Of+Lmax+1];
            WP = &Fm[Of+Lmax+4];
            QC = &Fm[Of+Lmax+7];
            WQ = &Fm[Of+Lmax+10];
            inv_2zab = Fm[Of+Lmax+13];
            min_rho_zab2  = Fm[Of+Lmax+14]; // - rho/zab**2
            inv_2zcd = Fm[Of+Lmax+15];
            min_rho_zcd2  = Fm[Of+Lmax+16]; // - rho/zcd**2
            inv_2z = Fm[Of+Lmax+17];
         }

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

//            const int my_vrr_rank = 0;
//            const int vrr_team_size = 1;

            if ( t == VRR1 ){ 
               execute_VRR1( m, m1, m2, m3, PA, WP, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR2 ){
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5]; 
               execute_VRR2( la, m, m1, m2, m3, m4, m5, PA, WP, inv_2zab, min_rho_zab2, my_vrr_rank, vrr_team_size);
            } else if ( t == VRR3 ){ 
               execute_VRR1( m, m1, m2, m3, QC, WQ, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR4 ){ 
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5];
               execute_VRR2( lc, m, m1, m2, m3, m4, m5, QC, WQ, inv_2zcd, min_rho_zcd2, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR5 ){ 
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5]; m6 = &pr_mem[off_m6];
               execute_VRR5(
                  la, lc, m, m1, m2, m3, m4, m5, m6, 
                  QC, WQ, inv_2zcd, min_rho_zcd2, inv_2z, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR6 ){ 
               m4 = &pr_mem[off_m4];
               execute_VRR6( la, m, m1, m2, m3, m4, QC, WQ, inv_2z, my_vrr_rank, vrr_team_size);
            } else if ( t == CP2S){
               m2 = &sh_mem[off_m2];
               execute_CP2S_gpu( 
                  la, lc, m1, m2, my_vrr_rank, vrr_team_size, hrr_blocksize,
                  ipa, ipb, ipc, ipd, nla, nlb, nlc, nld, npa, npb, npc, npd, Ka, Kb, Kc, Kd );
            } else if ( t == SYTM ){
//               __syncthreads();
            }
         } // end of loop over op  
      }
   }
}




void compute_VRR_batched_low(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ Pm_input_list,
      const unsigned int* const __restrict__ HRR,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC ){
//#pragma omp target teams num_teams(Ncells) distribute map(to:plan,Pm_input_list,HRR,Fm,data ) map(from:AC,ABCD)
   for( int block=0; block < Ncells ; block++ ){

      unsigned int Ov     = HRR[block*HRR_SIZE+HRR_OFFSET_OV];
      unsigned int n_prm  = HRR[block*HRR_SIZE+HRR_OFFSET_NPRM];
      unsigned int Og     = HRR[block*HRR_SIZE+HRR_OFFSET_OG];
//        Oq     = HRR[block*HRR_SIZE+HRR_OFFSET_OQ]
//        idx_AB = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_AB]
//        idx_CD = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_CD]
      unsigned int nlabcd = HRR[block*HRR_SIZE+HRR_OFFSET_NLABCD];
      unsigned int plabcd = HRR[block*HRR_SIZE+HRR_OFFSET_PLABCD];
      unsigned int ppabcd = HRR[block*HRR_SIZE+HRR_OFFSET_PPABCD];
      unsigned int idx_Ka = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KA];
      unsigned int idx_Kb = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KB];
      unsigned int idx_Kc = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KC];
      unsigned int idx_Kd = HRR[block*HRR_SIZE+HRR_OFFSET_IDX_KD];
      unsigned int Lmax   = HRR[block*HRR_SIZE+HRR_OFFSET_LMAX];
      unsigned int Of0    = HRR[block*HRR_SIZE+HRR_OFFSET_OF0];
      unsigned int Op     = HRR[block*HRR_SIZE+HRR_OFFSET_OP];

      const double* Ka = &data[idx_Ka];
      const double* Kb = &data[idx_Kb];
      const double* Kc = &data[idx_Kc];
      const double* Kd = &data[idx_Kd];

      int F_size = Lmax+1;
      if (Lmax > 0){ F_size += 4*3+5; }
 
      unsigned int nla,nlb,nlc,nld,npa,npb,npc,npd;
      decode4( plabcd, &nla,&nlb,&nlc,&nld );
      decode4( ppabcd, &npa,&npb,&npc,&npd );
      double* sh_mem = &ABCD[ Og * hrr_blocksize ];

      for ( unsigned i=0; i < hrr_blocksize * nlabcd ; i++ ){
         sh_mem[i] = 0.0 ;
      }
//#pragma omp parallel for
      for ( unsigned thread=0; thread < n_prm; thread++ ){
         unsigned int Of   = Of0 + thread * F_size;
         unsigned int ipzn = Pm_input_list[(Op + thread)*PM_SIZE+PMA_OFFSET_IPZN ];
         unsigned int ipa,ipb,ipc,ipd;

         decode_ipabcd_none( ipzn, &ipa,&ipb,&ipc,&ipd );

         double* pr_mem = &AC[ (Ov + thread) * vrr_blocksize ];

         for( int il=0; il < L+1; il++ ){
            pr_mem[il] = Fm[Of+il];
         }
         const double* PA = nullptr;
         const double* WP = nullptr;
         const double* QC = nullptr;
         const double* WQ = nullptr;
         double inv_2zab = 0.;
         double min_rho_zab2 = 0.;
         double inv_2zcd = 0.;
         double min_rho_zcd2 = 0.;
         double inv_2z = 0.;

         if (L > 0){
            PA = &Fm[Of+Lmax+1];
            WP = &Fm[Of+Lmax+4];
            QC = &Fm[Of+Lmax+7];
            WQ = &Fm[Of+Lmax+10];
            inv_2zab = Fm[Of+Lmax+13];
            min_rho_zab2  = Fm[Of+Lmax+14]; // - rho/zab**2
            inv_2zcd = Fm[Of+Lmax+15];
            min_rho_zcd2  = Fm[Of+Lmax+16]; // - rho/zcd**2
            inv_2z = Fm[Of+Lmax+17];
         }

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

            const int my_vrr_rank = 0;
            const int vrr_team_size = 1;

            if ( t == VRR1 ){ 
               execute_VRR1( m, m1, m2, m3, PA, WP, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR2 ){
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5]; 
               execute_VRR2( la, m, m1, m2, m3, m4, m5, PA, WP, inv_2zab, min_rho_zab2, my_vrr_rank, vrr_team_size);
            } else if ( t == VRR3 ){ 
               execute_VRR1( m, m1, m2, m3, QC, WQ, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR4 ){ 
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5];
               execute_VRR2( lc, m, m1, m2, m3, m4, m5, QC, WQ, inv_2zcd, min_rho_zcd2, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR5 ){ 
               m4 = &pr_mem[off_m4]; m5 = &pr_mem[off_m5]; m6 = &pr_mem[off_m6];
               execute_VRR5(
                  la, lc, m, m1, m2, m3, m4, m5, m6, 
                  QC, WQ, inv_2zcd, min_rho_zcd2, inv_2z, my_vrr_rank, vrr_team_size );
            } else if ( t == VRR6 ){ 
               m4 = &pr_mem[off_m4];
               execute_VRR6( la, m, m1, m2, m3, m4, QC, WQ, inv_2z, my_vrr_rank, vrr_team_size);
            } else if ( t == CP2S){
               m2 = &sh_mem[off_m2];
               execute_CP2S_v2( 
                  la, lc, m1, m2, my_vrr_rank, vrr_team_size, hrr_blocksize,
                  ipa, ipb, ipc, ipd, nla, nlb, nlc, nld, npa, npb, npc, npd, Ka, Kb, Kc, Kd );
            }
         } // end of loop over op  
      }
   }
}



void compute_VRR_batched(
      const int Ncells, const std::vector<int>& plan, const std::vector<unsigned int>& Pm_input_list,
      const std::vector<unsigned int>& HRR, const std::vector<double>& Fm, const std::vector<double>& data,
      std::vector<double>& AC, std::vector<double>& ABCD, int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC ){

   compute_VRR_batched_low(
      Ncells, plan.data(), Pm_input_list.data(), HRR.data(), Fm.data(),
      data.data(), AC.data(), ABCD.data(), vrr_blocksize, hrr_blocksize, L, numV, numVC );
}

