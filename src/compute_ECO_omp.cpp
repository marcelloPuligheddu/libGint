#include <iostream>
#include "compute_ECO_omp.h"
#include "util.h"
#include "define.h"
#include <cassert>
#include <omp.h>

using std::cout;
using std::endl;


// Sums the primitive along the N3 cell vector, goes from AC to AC
void compute_SFT_omp(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm, // unused
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD, // unused
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){

   unsigned int Nop = numVC - numV + 1;


   #pragma omp target teams distribute depend( in:plan,PMX,FVH,data,AC ) depend( out:AC )
   for( int block=0; block < Ncells*Nop ; block += 0 ){

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
      const int NcoA = (la+1)*(la+2)/2; // NLco_dev(la);
      const int NcoC = (lc+1)*(lc+2)/2; // NLco_dev(lc);
      const int NcoAC = NcoA*NcoC;
      const int VBS = vrr_blocksize;

      // arguable
//      constexpr int ETS = 16;
//      constexpr int NET =  8;
//      const int best_eco_team_size = NcoAC ;
//      int eco_team_size = blockDim.x;
//      while ( eco_team_size > best_eco_team_size ){ eco_team_size /= 2; }

//      int num_eco_teams = blockDim.x / eco_team_size;
//      int my_eco_team = omp_get_thread_num() / ETS; // threadIdx.x / ETS;
//      int my_eco_rank = omp_get_thread_num() % ETS; // threadIdx.x % ETS;

      double * const pr_0 = &AC[ Ov*Ng*VBS + off_m1];
      const int pr_ld_i = Ng*VBS;
      // PR[i,0,j] = sum( PR[i,n,j] )
      // pr_ld_i = Ng*VBS
      // pr_ld_n = VBS
      #pragma omp parallel for
      for ( int idx_prm = 0 ; idx_prm < n_prm ; idx_prm++ ){
         double * pr = pr_0 + pr_ld_i*idx_prm;
         for( int j = 0; j < NcoAC; j++ ){
            double s = 0.0;
            for( int n3 = 1 ; n3 < Ng; n3++ ){ s += pr[ VBS*n3 + j ]; }
            pr[j] += s;
         }
      }
   }
}


// Transforms the NCELLS*NOPS*N1*N2*NGAUSS4 AC into the contracted NCELLS*NOP*N1*N2*NNL4 AC0
// The results are saved into ABCD, since they will be the starting points of the HHR
// Computes the 4 matrix products CC1 @ CC2 @ CC3 @ CC4 @ AC

void compute_ECO_omp(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm, // unused
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){

   unsigned int Nop = numVC - numV + 1;

   #pragma omp target teams distribute depend( in:plan,PMX,FVH,Fm,data,AC ) depend( out:ABCD )
   for( int block=0; block < Ncells*Nop ; block += 0 ){

      unsigned int ibk, op,Ov,Og,n_prm,nlabcd,npabcd,idx_K;
      int t,la,lc,off_m1,off_m2, NcoA,NcoC,NcoAC;

      ibk    = block / (Nop); 
      op     = block % (Nop) + numV ;
      Ov     = FVH[ibk*FVH_SIZE+FVH_OFFSET_OV];
      Og     = FVH[ibk*FVH_SIZE+FVH_OFFSET_OG];
      n_prm  = FVH[ibk*FVH_SIZE+FVH_OFFSET_NPRM];
      nlabcd = FVH[ibk*FVH_SIZE+FVH_OFFSET_NLABCD];
      npabcd = FVH[ibk*FVH_SIZE+FVH_OFFSET_NPABCD];
      idx_K  = FVH[ibk*FVH_SIZE+FVH_OFFSET_IDX_K];
      t  = plan[ op*OP_SIZE + T__OFFSET ];
      la = plan[ op*OP_SIZE + LA_OFFSET ];
      lc = plan[ op*OP_SIZE + LC_OFFSET ];
      off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
      off_m2 = plan[ op*OP_SIZE + M2_OFFSET ];
      NcoA = (la+1)*(la+2)/2; // NLco_dev(la);
      NcoC = (lc+1)*(lc+2)/2; // NLco_dev(lc);
      NcoAC = NcoA*NcoC;

      if ( t != CP2S ){ continue; }

      double * out = &ABCD[ Og*hrr_blocksize + off_m2];
      const double * const inp = &AC[Ov*Ng*vrr_blocksize + off_m1];
      const double * const Kabcd = &data[idx_K];

      constexpr int TS_l = 1;
      constexpr int TS_j = 1;
      constexpr int F1 = 8;
      constexpr int F2 = 16;
      constexpr int BS_p = 16;
//      constexpr int dim = F1*F2;
      constexpr int BS_l = F1 * TS_l;
      constexpr int BS_j = F2 * TS_j;
      constexpr int totResBlkT = BS_l * BS_j;
      constexpr int numThrBlkT = totResBlkT / (TS_l*TS_j);
      constexpr int strideK = numThrBlkT / BS_p;
      constexpr int strideI = numThrBlkT / BS_j;

//      __shared__ double sK[BS_l*BS_p];
//      __shared__ double sI[BS_p*BS_j];
      static double sK[BS_l*BS_p];
      static double sI[BS_p*BS_j];

      #pragma omp parallel 
      {
      const int tid  =omp_get_thread_num();
      const int tRow = tid / ( BS_j/TS_j ); // thr / F2
      const int tCol = tid % ( BS_j/TS_j );
      const int iRoK = tid / ( BS_p );
      const int iCoK = tid % ( BS_p );
      const int iRoI = tid / ( BS_j );
      const int iCoI = tid % ( BS_j );

      double regL[TS_l];
      double regJ[TS_j];
      double tRes[TS_l*TS_j];

      unsigned int iB_p = iCoK;
      unsigned int og_p = 0;

//      if ( threadIdx.x == 0 and blockIdx.x % 100 == 0 ){ printf("ECO < %d %d %d > \n", nlabcd, n_prm, NcoAC ); }

      // Loop over nlabcd, then over primitives then over cartesian components
      for( unsigned int IB_l = 0 ; IB_l  < (nlabcd+BS_l-1) ; IB_l += BS_l ){
         for( unsigned int IB_p = 0 ; IB_p  < (n_prm+BS_p-1) ; IB_p += BS_p ){
            unsigned int p = iB_p + IB_p;
            if ( p < n_prm ){ og_p = PMX[(Ov+p)*PMX_SIZE+PMX_OFFSET_OGP ]; } 
            // collaborate to compute and load a BS_l * BS_p block of K in shared memory
            for ( int offK = 0 ; offK < BS_l; offK+=strideK ){
               unsigned int iB_l = iRoK + offK;
               unsigned int l = iB_l + IB_l;
               if ( p < n_prm and l < nlabcd ){
                  sK[iB_l*BS_p+iB_p ] = Kabcd[l*npabcd + og_p];
               } else {sK[iB_l*BS_p+iB_p] = 0.0;}
            }

//            __syncthreads(); // __sync after writing sK
            #pragma omp barrier

            // Uses this sK to run through the full [BS_l,ALL_J] input vector
            for ( unsigned int IB_j = 0 ; IB_j  < (NcoAC+BS_j-1) ; IB_j += BS_j ){

               // Loads a BS_p * BS_j block of the input matrix
               for ( int offI = 0 ; offI < BS_p; offI+=strideI ){
                  unsigned int iB_p = iRoI + offI;
                  unsigned int iB_j = iCoI;
                  unsigned int p = iB_p + IB_p;
                  unsigned int j = iB_j + IB_j;
                  
                  if ( p < n_prm and j < NcoAC ){
                     sI[iB_p*BS_j+iB_j] = inp[p*Ng*vrr_blocksize+j];
                  } else { sI[iB_p*BS_j+iB_j] = 0.0; }
               }

//               __syncthreads(); // __sync after writing sI
               #pragma omp barrier

               // Zeroes the register local results.
               for( int iT_lj = 0 ; iT_lj < TS_l*TS_j; iT_lj ++ ){ tRes[iT_lj] = 0.0; }
               // Sums over the block of primitives for all TS_l and TS_j
               for ( unsigned int iB_p = 0; iB_p < BS_p; iB_p++ ){
                  // sets local register caches
                  for( int iT_l = 0 ; iT_l < TS_l; iT_l ++ ){ regL[iT_l] = sK[ (tRow*TS_l+iT_l)*BS_p + iB_p             ]; }
                  for( int iT_j = 0 ; iT_j < TS_j; iT_j ++ ){ regJ[iT_j] = sI[ (iB_p          )*BS_j + (tCol*TS_j+iT_j) ]; }

                  for ( unsigned int res_l = 0 ; res_l < TS_l ; res_l++ ){
                     for ( unsigned int res_j = 0 ; res_j < TS_j ; res_j++ ){
                        tRes[res_l*TS_j+res_j] += regL[res_l] * regJ[res_j];
                     }
                  }
               }

               // Writes the results to output. Each thread is writing to a different location
               // No block can write to the same braket (for better or worse)
               // So no atomic is necessary
               for ( unsigned int res_l = 0 ; res_l < TS_l ; res_l++ ){
                  for ( unsigned int res_j = 0 ; res_j < TS_j ; res_j++ ){
                     unsigned int l = tRow * TS_l + res_l + IB_l;
                     unsigned int j = tCol * TS_j + res_j + IB_j;
                     if ( l < nlabcd and j < NcoAC ){
                        out[l*hrr_blocksize+j] += tRes[res_l*TS_j+res_j];
                     }
                  }
               }
//               __syncthreads(); // __sync after using sI
               #pragma omp barrier
            }
//            __syncthreads(); // sync after using sK
            #pragma omp barrier
         }
      } // end of #pragma omp parallel 
      } // end of strange gemm
   }
}


