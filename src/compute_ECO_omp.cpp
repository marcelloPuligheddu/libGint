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
   for( int block=0; block < Ncells*Nop ; block ++ ){

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
      printf("Preparing to sum b %d p %d o %d -> Ov %d @ %p \n", block, p, op, Ov, &pr_0 );
      #pragma omp parallel for
      for ( int idx_prm = 0 ; idx_prm < n_prm ; idx_prm++ ){
         double * pr = pr_0 + pr_ld_i*idx_prm;
         for( int j = 0; j < NcoAC; j++ ){
            double s = 0.0;
            for( int n3 = 1 ; n3 < Ng; n3++ ){ s += pr[ VBS*n3 + j ]; }
            if ( s*s > 1.e-6 ){
               printf("block %d p %d op %d i %d : @ %p j %d pr %lg s %lg \n", block, p, op, idx_prm, &pr, j, pr[j], s );
            }
            pr[j] += s;
         }
      }
   }
}


// Transforms the NCELLS*NOPS*N1*N2*NGAUSS4 AC into the contracted NCELLS*NOP*N1*N2*NNL4 AC0
// Uses the preomputed 4 matrix products CC1 @ CC2 @ CC3 @ CC4 = K
// The results are saved into ABCD, since they will be the starting points of the HHR

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

   #pragma omp target teams distribute parallel for depend( in:plan,PMX,FVH,Fm,data,AC ) depend( out:ABCD )
   for( int block=0; block < Ncells*Nop ; block ++ ){

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

      printf("Preparing to eco b %d o %d t %d  \n", block, op, t );
      if ( t != CP2S ){ continue; }

      double * out = &ABCD[ Og*hrr_blocksize + off_m2];
      const double * const inp = &AC[Ov*Ng*vrr_blocksize + off_m1];
      const double * const Kabcd = &data[idx_K];


      for ( unsigned int p = 0; p < n_prm; p++ ){
         unsigned int og_p = PMX[(Ov+p)*PMX_SIZE+PMX_OFFSET_OGP ];
         for ( unsigned int l = 0; l < nlabcd; l++ ){
            double k = Kabcd[l*npabcd + og_p];
            for( unsigned int j = 0; j < NcoAC; j++ ){
               if ( inp[p*Ng*vrr_blocksize+j]*inp[p*Ng*vrr_blocksize+j] > 1.e-6 ){
                  printf("block %d p %d l %d j %d : out %d inp %lg k %lg \n", block, p, l, j, Og*hrr_blocksize+off_m2+l*hrr_blocksize+j, inp[p*Ng*vrr_blocksize+j], k );
               }
               out[l*hrr_blocksize+j] += inp[p*Ng*vrr_blocksize+j] * k;
            }
         }
      }
   }
}


