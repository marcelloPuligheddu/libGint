#include <iostream>
#include "compute_VRR.h"
#include "util.h"
#include "define.h"
#include <cassert>


using std::cout;
using std::endl;


__global__ void compute_ECO_batched_gpu_low(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm, // unused
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){

   unsigned int Nop = numVC - numV + 1;



   for( int block=blockIdx.x; block < Ncells*Nop ; block += gridDim.x ){

      unsigned int ibk   =  block / (Nop); 
      int op              =  block % Nop + numV ;

      unsigned int Ov     = FVH[ibk*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int Og     = FVH[ibk*FVH_SIZE+FVH_OFFSET_OG];
      unsigned int n_prm  = FVH[ibk*FVH_SIZE+FVH_OFFSET_NPRM];
      unsigned int nlabcd = FVH[ibk*FVH_SIZE+FVH_OFFSET_NLABCD];
      unsigned int npabcd = FVH[ibk*FVH_SIZE+FVH_OFFSET_NPABCD];
      unsigned int idx_Ka = FVH[ibk*FVH_SIZE+FVH_OFFSET_IDX_KA];
      unsigned int idx_Kb = FVH[ibk*FVH_SIZE+FVH_OFFSET_IDX_KB];
      unsigned int idx_Kc = FVH[ibk*FVH_SIZE+FVH_OFFSET_IDX_KC];
      unsigned int idx_Kd = FVH[ibk*FVH_SIZE+FVH_OFFSET_IDX_KD];

      const double* Ka = &data[idx_Ka];
      const double* Kb = &data[idx_Kb];
      const double* Kc = &data[idx_Kc];
      const double* Kd = &data[idx_Kd];

      unsigned int nla,nlb,nlc,nld,npa,npb,npc,npd;
      decode_shell( nlabcd, &nla,&nlb,&nlc,&nld, &npa,&npb);
      decode4( npabcd, &npa,&npb,&npc,&npd );
      const unsigned int nl___d = nld;
      const unsigned int nl__cd = nlc*nl___d;
      const unsigned int nl_bcd = nlb*nl__cd;
      nlabcd = nla*nl_bcd;

      // Find the contraction we are doing
      const int t  = plan[ op*OP_SIZE + T__OFFSET ];
      if ( t != CP2S){ continue; } // TODO probably off by 1, so block 0 is skipped
      const int la = plan[ op*OP_SIZE + LA_OFFSET ];
      const int lc = plan[ op*OP_SIZE + LC_OFFSET ];
      const int off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
      const int off_m2 = plan[ op*OP_SIZE + M2_OFFSET ];

      const int NcoA = NLco_dev(la);
      const int NcoC = NLco_dev(lc);
      const int NcoAC = NcoA*NcoC;

      double * out = &ABCD[ Og*hrr_blocksize + off_m2];
      double * inp = &AC[Ov*Ng*vrr_blocksize + off_m1];

      __shared__ double sKa[MAX_N_L * MAX_N_PRM];
      __shared__ double sKb[MAX_N_L * MAX_N_PRM];
      __shared__ double sKc[MAX_N_L * MAX_N_PRM];
      __shared__ double sKd[MAX_N_L * MAX_N_PRM];

      for( unsigned int idx=threadIdx.x; idx < nla * npa ; idx += blockDim.x ){ sKa[idx] = Ka[idx]; }
      for( unsigned int idx=threadIdx.x; idx < nlb * npb ; idx += blockDim.x ){ sKb[idx] = Kb[idx]; }
      for( unsigned int idx=threadIdx.x; idx < nlc * npc ; idx += blockDim.x ){ sKc[idx] = Kc[idx]; }
      for( unsigned int idx=threadIdx.x; idx < nld * npd ; idx += blockDim.x ){ sKd[idx] = Kd[idx]; }

      __syncthreads();



      const int TS_l = 2;
      const int TS_j = 2;
      const int F1 = 8;
      const int F2 = 8;
      const int BS_p = 8;
      const int dim = F1*F2; // 64
      const int BS_l = F1 * TS_l; // 32
      const int BS_j = F2 * TS_j; // 32
      const int totResBlkT = BS_l * BS_j; // 32*32 = 1024
      const int numThrBlkT = totResBlkT / (TS_l*TS_j); // = F1 * F2 = 64 = 1024 / 4 / 4 
      const int strideK = numThrBlkT / BS_p; // = BS_l * BS_j / TS_l*TS_j / BS_p = dim / BS_p = 64 / 64 = 1
      const int strideI = numThrBlkT / BS_j; // = BS_l * BS_j / TS_l*TS_j / BS_j = BS_l / TS_l / TS_j = F1 / TS_j = 64 / 32 = 2


      assert( numThrBlkT == dim );
      assert( numThrBlkT == blockDim.x );
      assert( BS_l * BS_p >= dim );
      assert( BS_p * BS_j >= dim );
      assert( strideK > 0 );
      assert( strideI > 0 );
      

      __shared__ double sK[BS_l*BS_p];
      __shared__ double sI[BS_p*BS_j];

      const int tRow = threadIdx.x / ( BS_j/TS_j ); // thr / F2
      const int tCol = threadIdx.x % ( BS_j/TS_j );
      const int iRoK = threadIdx.x / ( BS_p );
      const int iCoK = threadIdx.x % ( BS_p );
      const int iRoI = threadIdx.x / ( BS_j );
      const int iCoI = threadIdx.x % ( BS_j );

      double regL[TS_l];
      double regJ[TS_j];
      double tRes[TS_l*TS_j];

//      if ( blockIdx.x == 1 and threadIdx.x == 0 ){ printf("Computing %d(%d %d %d %d ) %d = %d %d(%d %d %d %d ) @ %d %d at l %d %d \n", 
//            nlabcd,nla,nlb,nlc,nld, NcoAC, nlabcd, n_prm,npa,npb,npc,npd, n_prm, NcoAC, la, lc ); }

//      if ( blockIdx.x == 1 and threadIdx.x == 0 ){
//         for( unsigned int idx=0; idx < nla * npa ; idx ++ ){ printf( " Ka %d  = %lg \n", idx, sKa[idx]); }
//         for( unsigned int idx=0; idx < nlb * npb ; idx ++ ){ printf( " Kb %d  = %lg \n", idx, sKb[idx]); }
//         for( unsigned int idx=0; idx < nlc * npc ; idx ++ ){ printf( " Kc %d  = %lg \n", idx, sKc[idx]); }
//         for( unsigned int idx=0; idx < nld * npd ; idx ++ ){ printf( " Kd %d  = %lg \n", idx, sKd[idx]); }
//      }


//      printf("Computing %d %d = %d %d @ %d %d at l %d %d \n", nlabcd, NcoAC, nlabcd, n_prm, n_prm, NcoAC, la, lc );

      // Loop over nlabcd, then over primitives then over cartesian components
      for( unsigned int IB_l = 0 ; IB_l  < (nlabcd+BS_l-1) / BS_l ; IB_l ++ ){
         for( unsigned int IB_p = 0 ; IB_p  < (n_prm+BS_p-1) / BS_p ; IB_p ++ ){

            // collaborate to compute and load a BS_l * BS_p block of K in shared memory
            for ( int offK = 0 ; offK < BS_l; offK+=strideK ){

               unsigned int iB_l = iRoK + offK;
               unsigned int iB_p = iCoK;
               unsigned int l = iB_l + IB_l*BS_l;
               unsigned int p = iB_p + IB_p*BS_p;

               if ( p < n_prm and l < nlabcd ){
                  unsigned int a = (l / nl_bcd ) % nla;
                  unsigned int b = (l / nl__cd ) % nlb ;
                  unsigned int c = (l / nl___d ) % nlc ;
                  unsigned int d =  l            % nld ;
                  unsigned int ipzn = PMX[Ov+p];
                  unsigned int ipa,ipb,ipc,ipd;
                  decode4( ipzn, &ipa,&ipb,&ipc,&ipd );
                  double K = sKa[a*npa + ipa] * sKb[b*npb + ipb] * sKc[c*npc + ipc] * sKd[d*npd + ipd];
                  sK[iB_l*BS_p+iB_p ] = K;
//                  if ( blockIdx.x == 1 ){
//                     printf("K [%d(%d %d %d %d ) %d(%d %d %d %d )] : %lf : %lf %lf %lf %lf \n", l, a,b,c,d, p,ipa,ipb,ipc,ipd, K, sKa[a*npa + ipa], sKb[b*npb + ipb], sKc[c*npc + ipc], sKd[d*npd + ipd] ); 
//                  }
               } else {sK[iB_l*BS_p+iB_p] = 0.0;}
            }

            __syncthreads(); // __sync after writing sI, will mean sync after sK

            // Uses this sK to run through the full [BS_l,ALL_J] input vector
            for ( unsigned int IB_j = 0 ; IB_j  < (NcoAC+BS_j-1) / BS_j ; IB_j ++ ){

               // Loads a BS_p * BS_j block of the input matrix
               for ( int offI = 0 ; offI < BS_p; offI+=strideI ){
                  unsigned int iB_p = iRoI + offI;
                  unsigned int iB_j = iCoI;
                  unsigned int p = iB_p + IB_p*BS_p;
                  unsigned int j = iB_j + IB_j*BS_j;
                  
                  if ( p < n_prm and j < NcoAC ){
//                     if ( blockIdx.x == 1 and threadIdx.x == 0 ){ printf("Loading %lg %d %d \n", inp[p*Ng*vrr_blocksize+j], p, j);}
                     sI[iB_p*BS_j+iB_j] = inp[p*Ng*vrr_blocksize+j];
                  } else { sI[iB_p*BS_j+iB_j] = 0.0; }

               }

               __syncthreads(); // __sync after writing sI, will mean sync after sK

               // Zeroes the register local results.
               for( int iT_lj = 0 ; iT_lj < TS_l*TS_j; iT_lj ++ ){ tRes[iT_lj] = 0.0; }
               // Sums over the block of primitives for all TS_l and TS_j
               for ( unsigned int iB_p = 0; iB_p < BS_p; iB_p++ ){
                  // sets local register caches
                  for( int iT_l = 0 ; iT_l < TS_l; iT_l ++ ){
                     regL[iT_l] = sK[ (tRow*TS_l+iT_l)*BS_p+iB_p ];
                  }
                  for( int iT_j = 0 ; iT_j < TS_j; iT_j ++ ){
                     regJ[iT_j] = sI[ (iB_p)*BS_j + (tCol*TS_j+iT_j) ];
                  }

                  for ( unsigned int res_l = 0 ; res_l < TS_l ; res_l++ ){
                     for ( unsigned int res_j = 0 ; res_j < TS_j ; res_j++ ){
//                        if ( blockIdx.x == 1 and threadIdx.x == 0 ){ printf("Adding %lg %lg to %lg \n", regL[res_l], regJ[res_j], tRes[res_l*TS_j+res_j] );}
                        tRes[res_l*TS_j+res_j] += regL[res_l] * regJ[res_j];
                     }
                  }
               }
               __syncthreads(); // __sync after writing sI, will mean sync after sK

               // Writes the results to output. Each thread is writing to a different location
               // No block can write to the same braket (for better or worse)
               // So no atomic is necessary
               for ( unsigned int res_l = 0 ; res_l < TS_l ; res_l++ ){
                  for ( unsigned int res_j = 0 ; res_j < TS_j ; res_j++ ){
                     unsigned int l = tRow * TS_l + res_l + IB_l * BS_l;
                     unsigned int j = tCol * TS_j + res_j + IB_j * BS_j;
                     if ( l < nlabcd and j < NcoAC ){
                        out[l*hrr_blocksize+j] += tRes[res_l*TS_j+res_j];
                     }
                  }
               }
               __syncthreads(); // __sync after using sI
            }
            __syncthreads();
         }
      } // end of strange gemm
   }
}

__global__ void compute_SFT_batched_gpu_low(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm, // unused
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){

   unsigned int Nop = numVC - numV + 1;


   for( int block=blockIdx.x; block < Ncells*Nop ; block += gridDim.x ){

      unsigned int p      =  block / (Nop);
      int op              =  block % Nop + numV ;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];
      // Find the contraction we are doing
      const int t  = plan[ op*OP_SIZE + T__OFFSET ];
      if ( t != CP2S){ continue; }
      const int la = plan[ op*OP_SIZE + LA_OFFSET ];
      const int lc = plan[ op*OP_SIZE + LC_OFFSET ];
      const int off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
      const int NcoA = NLco_dev(la);
      const int NcoC = NLco_dev(lc);
      const int NcoAC = NcoA*NcoC;
      const int VBS = vrr_blocksize;

      // arguable
      const int best_eco_team_size = NcoAC ;
      int eco_team_size = blockDim.x;
      while ( eco_team_size > best_eco_team_size ){ eco_team_size /= 2; }

      int num_eco_teams = blockDim.x / eco_team_size;
      int my_eco_team = threadIdx.x / eco_team_size;
      int my_eco_rank = threadIdx.x % eco_team_size;

      double * const pr_0 = &AC[ Ov*Ng*VBS + off_m1];
      const int pr_ld_i = Ng*VBS;
      // PR[i,0,j] = sum( PR[i,n,j] )
      // pr_ld_i = Ng*VBS
      // pr_ld_n = VBS
      for ( int idx_prm = my_eco_team ; idx_prm < n_prm ; idx_prm += num_eco_teams ){
         double * pr = pr_0 + pr_ld_i*idx_prm;
         for( int j = my_eco_rank; j < NcoAC; j+= eco_team_size ){
            double s = 0.0;
            for( int n3 = 1 ; n3 < Ng; n3++ ){ s += pr[ VBS*n3 + j ]; }
            pr[j] += s;
         }
      }
   }
}

__global__ void compute_ECO_batched_gpu_low_old(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm, // unused
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){

//   int F_size = L+1;
//   if (L > 0){ F_size += 4*3+5; }

//   // TODO to constant array
//   unsigned int L1 = L / 2;
//   unsigned int L2 = (L+1) / 2;
//   unsigned int max_NcoAC = (L1+1)*(L1+2)*(L2+1)*(L2+2) / 4;

   unsigned int Nop = numVC - numV + 1;
   
   // arguable
   const int best_eco_team_size = (L+1) * (L+2) ; // max_NcoAC ;
   int eco_team_size = blockDim.x;
   while ( eco_team_size > best_eco_team_size ){ eco_team_size /= 2; }

   int num_eco_teams = blockDim.x / eco_team_size;
   int my_eco_team = threadIdx.x / eco_team_size;
   int my_eco_rank = threadIdx.x % eco_team_size;


   for( int block=blockIdx.x; block < Ncells*Nop ; block += gridDim.x ){

      unsigned int p      =  block / (Nop); 
      int n3              =  0; // (block / Nop ) % Ng;
      int op              =  block % Nop + numV ;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int Og     = FVH[p*FVH_SIZE+FVH_OFFSET_OG];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];
      unsigned int nlabcd = FVH[p*FVH_SIZE+FVH_OFFSET_NLABCD];
      unsigned int npabcd = FVH[p*FVH_SIZE+FVH_OFFSET_NPABCD];
      unsigned int idx_Ka = FVH[p*FVH_SIZE+FVH_OFFSET_IDX_KA];
      unsigned int idx_Kb = FVH[p*FVH_SIZE+FVH_OFFSET_IDX_KB];
      unsigned int idx_Kc = FVH[p*FVH_SIZE+FVH_OFFSET_IDX_KC];
      unsigned int idx_Kd = FVH[p*FVH_SIZE+FVH_OFFSET_IDX_KD];

      const double* Ka = &data[idx_Ka];
      const double* Kb = &data[idx_Kb];
      const double* Kc = &data[idx_Kc];
      const double* Kd = &data[idx_Kd];

      unsigned int nla,nlb,nlc,nld,npa,npb,npc,npd;
      decode_shell( nlabcd, &nla,&nlb,&nlc,&nld, &npa,&npb);
      decode4( npabcd, &npa,&npb,&npc,&npd );
      const unsigned int nl___d = nld;
      const unsigned int nl__cd = nlc*nl___d;
      const unsigned int nl_bcd = nlb*nl__cd;
      nlabcd = nla*nl_bcd;

      double* sh_mem = &ABCD[ Og * hrr_blocksize ];

      // Find the contraction we are doing
      const int t  = plan[ op*OP_SIZE + T__OFFSET ];
      if ( t != CP2S){ continue; }
      const int la = plan[ op*OP_SIZE + LA_OFFSET ];
      const int lc = plan[ op*OP_SIZE + LC_OFFSET ];
      const int off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
      const int off_m2 = plan[ op*OP_SIZE + M2_OFFSET ];

      const int NcoA = NLco_dev(la);
      const int NcoC = NLco_dev(lc);
      const int NcoAC = NcoA*NcoC;

      double* m2 = &sh_mem[off_m2];

      __shared__ double sKa[MAX_N_L * MAX_N_PRM];
      __shared__ double sKb[MAX_N_L * MAX_N_PRM];
      __shared__ double sKc[MAX_N_L * MAX_N_PRM];
      __shared__ double sKd[MAX_N_L * MAX_N_PRM];

      for( unsigned int idx=threadIdx.x; idx < nla * npa ; idx += blockDim.x ){ sKa[idx] = Ka[idx]; }
      for( unsigned int idx=threadIdx.x; idx < nlb * npb ; idx += blockDim.x ){ sKb[idx] = Kb[idx]; }
      for( unsigned int idx=threadIdx.x; idx < nlc * npc ; idx += blockDim.x ){ sKc[idx] = Kc[idx]; }
      for( unsigned int idx=threadIdx.x; idx < nld * npd ; idx += blockDim.x ){ sKd[idx] = Kd[idx]; }

      __syncthreads();

      for ( unsigned idx_prm = my_eco_team; idx_prm < n_prm ;  idx_prm += num_eco_teams ){

         bool found = false;
         double * pr_mem = nullptr;
         while ( not found and idx_prm < n_prm ){
            // Find the AC value we need to contract
            pr_mem = &AC[ ((Ov+idx_prm) * Ng + n3) * vrr_blocksize ];
            if (pr_mem[0] > 1.e-30 ){ found = true ; }
            else { idx_prm += num_eco_teams ; }
         }
         if ( not found or idx_prm >= n_prm ){ break; }

         double* m1 = &pr_mem[off_m1];
         unsigned int ipzn = PMX[Ov+idx_prm];
         unsigned int ipa,ipb,ipc,ipd;
         decode4( ipzn, &ipa,&ipb,&ipc,&ipd );

         // Loop over (a|c) integrals to contract, linear contractions, and components of these integrals
         unsigned int n_nl_AC = nlabcd * NcoAC;
         for ( unsigned int i = my_eco_rank; i < n_nl_AC ; i+= eco_team_size ){

            unsigned int ilabcd    = i / NcoAC;
            unsigned int j         = i % NcoAC;
           
            unsigned int a = (ilabcd / nl_bcd ) % nla;
            unsigned int b = (ilabcd / nl__cd ) % nlb ;
            unsigned int c = (ilabcd / nl___d ) % nlc ;
            unsigned int d =  ilabcd            % nld ;

            double K = sKa[ a*npa + ipa ] * sKb[ b*npb + ipb ] * sKc[ c*npc + ipc ] * sKd[ d*npd + ipd ];

            // must be atomic if different warps/blocks share the ABCD [i + j] array
            // printf("CP2S %d %d adding %lg %lg @ %p : %lg \n", blockIdx.x, threadIdx.x, K , m1[j], &m2[ilabcd*hrr_blocksize+j], m2[ilabcd*hrr_blocksize+j] );
            atomicAdd( &m2[ ilabcd*hrr_blocksize + j ] , K * m1[j]);
//            m2[ ilabcd*hrr_blocksize + j ] += K * m1[j];
         }
      }
   }
}


