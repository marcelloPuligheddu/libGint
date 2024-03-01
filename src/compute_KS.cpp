#include "compute_KS.h"
#include "define.h"
#include "util.h"


void compute_KS(
      const int Nqrtt,
      const unsigned int * const __restrict__ KS,
      const int la, const int lb, const int lc, const int ld,
      const double * const __restrict__ P,
      const double * const __restrict__ I,
      double * const __restrict__ K,
      const double * const __restrict__ data ){
   const int nsa = 2*la+1;
   const int nsb = 2*lb+1;
   const int nsc = 2*lc+1;
   const int nsd = 2*ld+1;
   const int nscd = nsc*nsd;
   const int nsbcd = nsb*nscd;
   const int nsabcd = nsa*nsbcd;

   for ( int block = 0 ; block < Nqrtt ; block++ ){
      const unsigned int idx_fac = KS[block * KS_SIZE + KS_OFFSET_IDXFAC ];
      const unsigned int inlabcd = KS[block * KS_SIZE + KS_OFFSET_INLABCD];
      const unsigned int ld_abcd = KS[block * KS_SIZE + KS_OFFSET_LDABCD ];
      const unsigned int offset_L_set_atom_ac = KS[block * KS_SIZE + KS_OFFSET_OFFAC  ];
      const unsigned int offset_L_set_atom_ad = KS[block * KS_SIZE + KS_OFFSET_OFFAD  ];
      const unsigned int offset_L_set_atom_bc = KS[block * KS_SIZE + KS_OFFSET_OFFBC  ];
      const unsigned int offset_L_set_atom_bd = KS[block * KS_SIZE + KS_OFFSET_OFFBD  ];
      const unsigned int Tall    = KS[block * KS_SIZE + KS_OFFSET_TALL   ];

      unsigned int inla,inlb,inlc,inld;
      unsigned int ldac,ldad,ldbc,ldbd;
      unsigned int  Tac, Tad, Tbc, Tbd;

      decode4( inlabcd, &inla, &inlb, &inlc, &inld );
      decode4( ld_abcd, &ldac, &ldad, &ldbc, &ldbd );
      decode4( Tall   , &Tac,  &Tad,  &Tbc,  &Tbd  );

      const unsigned int Oq = block * nsabcd ;
      const double fac = data[idx_fac];
      
      for ( int t = 0 ; t < nsabcd; t++ ){
         const int isa =   t / nsbcd;
         const int isb = ( t / nscd ) % nsb;
         const int isc = ( t / nsd  ) % nsc;
         const int isd =   t          % nsd;

         const int idx_a = inla*nsa + isa;
         const int idx_b = inlb*nsb + isb;
         const int idx_c = inlc*nsc + isc;
         const int idx_d = inld*nsd + isd;

         const int idx_ac = Tac ?  idx_c * ldac + idx_a :  idx_a * ldac + idx_c;
         const int idx_ad = Tad ?  idx_d * ldad + idx_a :  idx_a * ldad + idx_d;
         const int idx_bc = Tbc ?  idx_c * ldbc + idx_b :  idx_b * ldbc + idx_c;
         const int idx_bd = Tbd ?  idx_d * ldbd + idx_b :  idx_b * ldbd + idx_d;

         const int Iac = offset_L_set_atom_ac + idx_ac;
         const int Iad = offset_L_set_atom_ad + idx_ad;
         const int Ibc = offset_L_set_atom_bc + idx_bc;
         const int Ibd = offset_L_set_atom_bd + idx_bd;
         const double iabcd = - fac * I[ Oq + t ];
         const double pac = P [Iac];
         const double pad = P [Iad];
         const double pbc = P [Ibc];
         const double pbd = P [Ibd];
         // must be atomics
         K[ Ibd ] += pac * iabcd;
         K[ Ibc ] += pad * iabcd;
         K[ Iad ] += pbc * iabcd;
         K[ Iac ] += pbd * iabcd;
      }
   }
}


__global__ void compute_KS_gpu(
      const int Nqrtt,
      const unsigned int * const __restrict__ KS,
      const int la, const int lb, const int lc, const int ld,
      const double * const __restrict__ P,
      const double * const __restrict__ I,
      double * const __restrict__ K,
      const double * const __restrict__ data ){
   const int nsa = 2*la+1;
   const int nsb = 2*lb+1;
   const int nsc = 2*lc+1;
   const int nsd = 2*ld+1;
   const int nscd = nsc*nsd;
   const int nsbcd = nsb*nscd;
   const int nsabcd = nsa*nsbcd;

   for ( int block = blockIdx.x ; block < Nqrtt ; block += gridDim.x ){
      const unsigned int idx_fac = KS[block * KS_SIZE + KS_OFFSET_IDXFAC ];
      const unsigned int inlabcd = KS[block * KS_SIZE + KS_OFFSET_INLABCD];
      const unsigned int ld_abcd = KS[block * KS_SIZE + KS_OFFSET_LDABCD ];
      const unsigned int offset_L_set_atom_ac = KS[block * KS_SIZE + KS_OFFSET_OFFAC  ];
      const unsigned int offset_L_set_atom_ad = KS[block * KS_SIZE + KS_OFFSET_OFFAD  ];
      const unsigned int offset_L_set_atom_bc = KS[block * KS_SIZE + KS_OFFSET_OFFBC  ];
      const unsigned int offset_L_set_atom_bd = KS[block * KS_SIZE + KS_OFFSET_OFFBD  ];
      const unsigned int Tall    = KS[block * KS_SIZE + KS_OFFSET_TALL   ];

      unsigned int inla,inlb,inlc,inld;
      unsigned int ldac,ldad,ldbc,ldbd;
      unsigned int  Tac, Tad, Tbc, Tbd;

      decode4( inlabcd, &inla, &inlb, &inlc, &inld );
      decode4( ld_abcd, &ldac, &ldad, &ldbc, &ldbd );
      decode4( Tall   , &Tac,  &Tad,  &Tbc,  &Tbd  );

      const int Oq = block * nsabcd ;
      const double fac = data[idx_fac];
      
      for ( int t = threadIdx.x ; t < nsabcd; t += blockDim.x ){
         const int isa =   t / nsbcd;
         const int isb = ( t / nscd ) % nsb;
         const int isc = ( t / nsd  ) % nsc;
         const int isd =   t          % nsd;

         const int idx_a = inla*nsa + isa;
         const int idx_b = inlb*nsb + isb;
         const int idx_c = inlc*nsc + isc;
         const int idx_d = inld*nsd + isd;

         const int idx_ac = Tac ?  idx_c * ldac + idx_a :  idx_a * ldac + idx_c;
         const int idx_ad = Tad ?  idx_d * ldad + idx_a :  idx_a * ldad + idx_d;
         const int idx_bc = Tbc ?  idx_c * ldbc + idx_b :  idx_b * ldbc + idx_c;
         const int idx_bd = Tbd ?  idx_d * ldbd + idx_b :  idx_b * ldbd + idx_d;

         const int Iac = offset_L_set_atom_ac + idx_ac;
         const int Iad = offset_L_set_atom_ad + idx_ad;
         const int Ibc = offset_L_set_atom_bc + idx_bc;
         const int Ibd = offset_L_set_atom_bd + idx_bd;
         const double iabcd = - fac * I[ Oq + t ];
         const double kbd = iabcd * P [Iac];
         const double kbc = iabcd * P [Iad];
         const double kad = iabcd * P [Ibc];
         const double kac = iabcd * P [Ibd];
         // must be atomics on device, or however P and K are distributed
         atomicAdd( &K[ Ibd ] , kbd);
         atomicAdd( &K[ Ibc ] , kbc);
         atomicAdd( &K[ Iad ] , kad);
         atomicAdd( &K[ Iac ] , kac);
      }
   }
}




















