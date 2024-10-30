#include <iostream>
#include "compute_VRR.h"
#include "util.h"
#include "define.h"

using std::cout;
using std::endl;

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


   for( int block=blockIdx.x; block < Ncells*Nop ; block += gridDim.x ){

      unsigned int p      =  block / (Nop);
      int op              =  block % Nop + numV ;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];

      for ( int idx_prm = my_eco_team ; idx_prm < n_prm ; idx_prm += num_eco_teams ){
         double * pr = &AC[ Ov*Ng*VBS + off_m1 + Ng*VBS*idx_prm]; 
         for( int j = my_eco_rank; j < NcoAC; j+= eco_team_size ){
            double s = 0.0;
            for( int n3 = 1 ; n3 < Ng; n3++ ){ s += pr[ VBS*n3 + j ]; }
            pr[j] += s;
         }
      }
   }
}

__global__ void compute_ECO_batched_gpu_low(
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


