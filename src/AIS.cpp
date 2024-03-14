#include <vector>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include "plan.h"
#include "util.h"
#include "timer.h"
#include "UniqueArray.h"
#include "compute_Fm.h"
#include "compute_VRR.h"
#include "compute_HRR.h"
#include "compute_SPH.h"
#include "compute_TRA.h"
#include "fgamma.h"
#include "AIS.h"
#include "c2s.h"


using std::max;

void AIS::show_state(){

   for (unsigned int L : encoded_moments ){
      int la,lb,lc,ld;
      decodeL(L,&la,&lb,&lc,&ld);
      cout << " Moments : " << la << " " << lb << " " << lc << " " << ld << endl;
   }

}

inline unsigned int encode_ipabcd_n123( const int ipa, const int ipb, const int ipc, const int ipd, const int n1, const int n2, const int n3 ){
    unsigned int ret = 0;
    ret +=  n3+8;
    ret += (n2+8)* 16;
    ret += (n1+8)* 16 * 16;
    ret += (ipd) * 16 * 16 * 16;
    ret += (ipc) * 16 * 16 * 16 * 32;
    ret += (ipb) * 16 * 16 * 16 * 32 * 32;
    ret += (ipa) * 16 * 16 * 16 * 32 * 32 * 32;    
    return ret;
}

int max( std::vector<int> x ){
   if ( x.size() == 0 ){ return 0; };
   int ret = x[0];
   for( int idx=1; idx<x.size(); idx++ ){
      ret = max(ret, x[idx]);
   }
   return ret;
}

void AIS::set_max_n_prm( int max_n3 ){
   max_n_prm  = max_n3;
   max_n_prm *= max( npa );
   max_n_prm *= max( npb );
   max_n_prm *= max( npc );
   max_n_prm *= max( npd );
   prm_tmp_list.resize( PRM_TMP_SIZE * max_n_prm );
}

void AIS::set_L(){
   for( int la=0; la <= 2; la++ ){
   for( int lb=0; lb <= 2; lb++ ){
   for( int lc=0; lc <= 2; lc++ ){
   for( int ld=0; ld <= 2; ld++ ){
      int L = encodeL( la,lb,lc,ld );
      PMI[L].reserve( 100000 );
   }}}}
}

void AIS::add_prm( const int ipa, const int ipb, const int ipc, const int ipd, const int n1, const int n2, const int n3 ){
   unsigned int piabcdxyz = encode_ipabcd_n123(ipa,ipb,ipc,ipd,n1,n2,n3);
   prm_tmp_list[ n_prm * PRM_TMP_SIZE + PRM_TMP_OFFSET_IPZN] = piabcdxyz;
   n_prm++;
}

////////// A ////////
void AIS::setA( int i, double* A_, double* Za_, int npa_ ){
   if ( i >= idx_A.size() ){ idx_A.resize(i+1); }
   if ( i >= idx_Za.size() ){ idx_Za.resize(i+1); }
   if ( i >= npa.size() ){ npa.resize(i+1); }

   unsigned int tmp_idx_A = ua.add( A_, 3 );
   unsigned int tmp_idx_Za = ua.add( Za_, npa_ );
   idx_A[i] = tmp_idx_A;
   idx_Za[i] = tmp_idx_Za;
   npa[i] = npa_;
}

void AIS::clearAl(){
   all_la.clear();
   all_nla.clear();
   all_idx_Ka.clear();
}

void AIS::setAl( int i, int la_, int nla_, double* Ka_ ){
   if ( i >= all_la.size()     ){ all_la.resize( i+1 ); }
   if ( i >= all_nla.size()    ){ all_nla.resize( i+1 ); }
   if ( i >= all_idx_Ka.size() ){ all_idx_Ka.resize( i+1 ); }
   all_la[i].push_back(la_);
   all_nla[i].push_back(nla_);
   all_idx_Ka[i].push_back( ua.add( Ka_, npa[i]*nla_ ) );
}
////////// B ////////
void AIS::setB( int j, double* B_, double* Zb_, int npb_ ){
   if ( j >= idx_B.size() ){ idx_B.resize( j+1 ); }
   if ( j >= idx_Zb.size() ){ idx_Zb.resize( j+1 ); }
   if ( j >= npb.size() ){ npb.resize( j+1 ); }
   unsigned int tmp_idx_B = ua.add( B_, 3 );
   unsigned int tmp_idx_Zb = ua.add( Zb_, npb_ );
   idx_B[j] = tmp_idx_B;
   idx_Zb[j] = tmp_idx_Zb;
   npb[j] = npb_;
}

void AIS::clearBl(){
   all_lb.clear();
   all_nlb.clear();
   all_idx_Kb.clear();
}

void AIS::setBl( int j, int lb_, int nlb_, double* Kb_ ){
   if ( j >= all_lb.size()     ){ all_lb.resize( j+1 ); }
   if ( j >= all_nlb.size()    ){ all_nlb.resize( j+1 ); }
   if ( j >= all_idx_Kb.size() ){ all_idx_Kb.resize( j+1 ); }
   all_lb[j].push_back(lb_);
   all_nlb[j].push_back(nlb_);
   all_idx_Kb[j].push_back( ua.add( Kb_, npb[j]*nlb_ ) );
}

////////// C ////////
void AIS::setC( int k, double* C_, double* Zc_, int npc_ ){
   if ( k >= idx_C.size() ){ idx_C.resize( k+1 ); }
   if ( k >= idx_Zc.size() ){ idx_Zc.resize( k+1 ); }
   if ( k >= npc.size() ){ npc.resize( k+1 ); }
   unsigned int tmp_idx_C = ua.add( C_, 3 );
   unsigned int tmp_idx_Zc = ua.add( Zc_, npc_ );
   idx_C[k] = tmp_idx_C;
   idx_Zc[k] = tmp_idx_Zc;
   npc[k] = npc_;
}

void AIS::clearCl(){
   all_lc.clear();
   all_nlc.clear();
   all_idx_Kc.clear();
}

void AIS::setCl( int k, int lc_, int nlc_, double* Kc_ ){
   if ( k >= all_lc.size()     ){ all_lc.resize( k+1 ); }
   if ( k >= all_nlc.size()    ){ all_nlc.resize( k+1 ); }
   if ( k >= all_idx_Kc.size() ){ all_idx_Kc.resize( k+1 ); }
   all_lc[k].push_back(lc_);
   all_nlc[k].push_back(nlc_);
   all_idx_Kc[k].push_back( ua.add( Kc_, npc[k]*nlc_ ) );
}

////////// D ////////
void AIS::setD( int l, double* D_, double* Zd_, int npd_ ){
   if ( l >= idx_D.size() ){ idx_D.resize( l+1 ); }
   if ( l >= idx_Zd.size() ){ idx_Zd.resize( l+1 ); }
   if ( l >= npd.size() ){ npd.resize( l+1 ); }
   unsigned int tmp_idx_D = ua.add( D_, 3 );
   unsigned int tmp_idx_Zd = ua.add( Zd_, npd_ );
   idx_D[l] = tmp_idx_D;
   idx_Zd[l] = tmp_idx_Zd;
   npd[l] = npd_;
}

void AIS::clearDl(){
   all_ld.clear();
   all_nld.clear();
   all_idx_Kd.clear();
}

void AIS::setDl( int l, int ld_, int nld_, double* Kd_ ){
   if ( l >= all_ld.size()     ){ all_ld.resize( l+1 ); }
   if ( l >= all_nld.size()    ){ all_nld.resize( l+1 ); }
   if ( l >= all_idx_Kd.size() ){ all_idx_Kd.resize( l+1 ); }
   all_ld[l].push_back(ld_);
   all_nld[l].push_back(nld_);
   all_idx_Kd[l].push_back( ua.add( Kd_, npd[l]*nld_ ) );
}
///////////////////////

void AIS::add_shell ( int i, int j, int k, int l, int n1, int n2 ){
   
//   Timer timer;
//   Timer timer2;

   if (n_prm == 0){
      return;
   }

   int nnla = all_la[i].size();
   int nnlb = all_lb[j].size();
   int nnlc = all_lc[k].size();
   int nnld = all_ld[l].size();

   int idx_la = 0;
   int idx_lb = 0;
   int idx_lc = 0;
   int idx_ld = 0;


//   for( int idx_la=0; idx_la < nnla; idx_la++ ){
      unsigned int idx_Ka = all_idx_Ka[i][idx_la];
      int nla = all_nla[i][idx_la];
      int la = all_la[i][idx_la];
//      for( int idx_lb=0; idx_lb < nnlb; idx_lb++ ){
         unsigned int idx_Kb = all_idx_Kb[j][idx_lb];
         int nlb = all_nlb[j][idx_lb];
         int lb = all_lb[j][idx_lb];
//         for( int idx_lc=0; idx_lc < nnlc; idx_lc++ ){
            unsigned int idx_Kc = all_idx_Kc[k][idx_lc];
            int nlc = all_nlc[k][idx_lc];
            int lc = all_lc[k][idx_lc];

//            for( int idx_ld=0; idx_ld < nnld; idx_ld++ ){
               unsigned int idx_Kd = all_idx_Kd[l][idx_ld];
               int nld = all_nld[l][idx_ld];
               int ld = all_ld[l][idx_ld];

               const unsigned int N_cc = nla*nlb*nlc*nld;
               const unsigned int L = encodeL(la,lb,lc,ld);

               const unsigned int Of = offset_F[L];
               const unsigned int Ov = offset_V[L];
               const unsigned int Og = offset_G[L];
               const unsigned int Oq = offset_Q[L];

               const unsigned int encoded_nlabcd = encode4(nla,nlb,nlc,nld);
               const unsigned int encoded_npabcd = encode4(npa[i],npb[j],npc[k],npd[l]);

               while ( PMI[L].size() < (1 + Ov + n_prm) * PMI_SIZE ) {
//                  cout << " Realloc " << la<<lb<<lc<<ld << " " << PMI[L].size() << endl ;
                  PMI[L].resize( (1 + Ov + n_prm) * PMI_SIZE + 2 * PMI[L].capacity() );
               }

               unsigned int * pmi_l = & (PMI[L][Ov*PMI_SIZE]) ;

               for( unsigned int prm_idx = 0; prm_idx < n_prm; prm_idx++ ){
                  pmi_l[ prm_idx * PMI_SIZE + PMI_OFFSET_OF  ] = Of;
                  pmi_l[ prm_idx * PMI_SIZE + PMI_OFFSET_IPZN] = prm_tmp_list[prm_idx];
               }
//                    PMI[L].push_back(Of);
//                    PMI[L].push_back(prm_tmp_list[prm_idx]);
//               }

               const unsigned int tmp[FVH_SIZE] = {
                  Ov, Og, Oq, n_prm, idx_A[i], idx_B[j], idx_C[k], idx_D[l],
                  idx_Za[i], idx_Zb[j], idx_Zc[k], idx_Zd[l], idx_Ka, idx_Kb, idx_Kc, idx_Kd,
                  encoded_nlabcd, encoded_npabcd
               };
               FVH[L].insert( FVH[L].end(), tmp, tmp+18 );

               const int labcd = la+lb+lc+ld;
               Fm_size[L] += (1+labcd) * n_prm;
               if ( labcd > 0 ){
                  Fm_size[L] += (4*3+5) * n_prm;
               }

               if ( all_moments.count(L) == 0 ){
                  std::vector<int> * plan = NULL ;
                  unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
                  plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );
                  all_vrr_blocksize[L] = vrr_blocksize;
                  all_hrr_blocksize[L] = hrr_blocksize;
                  all_moments.insert(L);
               }

               AC_size[L] += all_vrr_blocksize[L] * n_prm;
               ABCD_size[L] += all_hrr_blocksize[L] * N_cc;

               offset_G[L] += N_cc;
               offset_V[L] += n_prm;
               offset_F[L] ++ ;

               encoded_moments.insert(L);
//}}}}

}

void AIS::add_cell() {

   if (n_prm == 0){
      return;
   }

   cell_in_set++;
   n_prm = 0;

}

int AIS::add_qrt( int la, int lb, int lc, int ld, int nla, int nlb, int nlc, int nld ){
   unsigned int nlabcd = nla*nlb*nlc*nld;
   unsigned int L = encodeL(la,lb,lc,ld);

   unsigned int SPH_idxs[SPH_SIZE] = {0} ;
   SPH_idxs[SPH_OFFSET_Q     ] = offset_Q[L];
   SPH_idxs[SPH_OFFSET_NLABCD] = nlabcd;
   SPH[L].insert(SPH[L].end(), SPH_idxs, SPH_idxs+SPH_SIZE);

   unsigned int nla_as_uint = (unsigned int) nla ;
   unsigned int nlb_as_uint = (unsigned int) nlb ;
   unsigned int nlc_as_uint = (unsigned int) nlc ;
   unsigned int nld_as_uint = (unsigned int) nld ;
   unsigned int TRA_idxs[TRA_SIZE] = {0};
   TRA_idxs[TRA_OFFSET_Q   ] = offset_Q[L];
   TRA_idxs[TRA_OFFSET_NLA ] = nla_as_uint;
   TRA_idxs[TRA_OFFSET_NLB ] = nlb_as_uint;
   TRA_idxs[TRA_OFFSET_NLC ] = nlc_as_uint;
   TRA_idxs[TRA_OFFSET_NLD ] = nld_as_uint;
   TRA_idxs[TRA_OFFSET_DEST] = dest;
   TRA[L].insert(TRA[L].end(), TRA_idxs, TRA_idxs+TRA_SIZE );

   ABCD0_size[L] += compute_Nc(la,lb,lc,ld) * nlabcd ;
   SPHER_size[L] += compute_Ns(la,lb,lc,ld) * nlabcd ;
   OUT_size[L] += compute_Ns(la,lb,lc,ld) * nlabcd ; 
   offset_Q[L] += nlabcd ;
   offset_T[L] += 1 ;

   unsigned int prev_dest = dest;
   dest += compute_Ns(la,lb,lc,ld) * nlabcd ;
   return prev_dest;


}

void AIS::add_qrtt(
      double symm_fac, int la, int lb, int lc, int ld, 
      int inla, int inlb, int inlc, int inld,
      int ld_ac, int ld_ad, int ld_bc, int ld_bd, 
      unsigned int offset_ac_L_set, unsigned int offset_ad_L_set, 
      unsigned int offset_bc_L_set, unsigned int offset_bd_L_set, 
      bool Tac, bool Tad, bool Tbc, bool Tbd ){

  

   unsigned int idx_fac ;
   // The symm_fac can be only 2,1,0.5 or 0.25 (or 0 in dev version). We assign them to 0,1,2 and 3 (and 4) respectively
   // We encode it here, decode it inside compute_KS

   if ( symm_fac < 0.1  ){ idx_fac = 4 ; }
   else if ( symm_fac < 0.3  ){ idx_fac = 3 ; }
   else if ( symm_fac < 0.75 ){ idx_fac = 2 ; }
   else if ( symm_fac < 1.5  ){ idx_fac = 1 ; }
   else { idx_fac = 0 ; }

   unsigned int L = encodeL(la,lb,lc,ld);
   unsigned int KS_idxs[KS_SIZE] = {0};
   KS_idxs[KS_OFFSET_IDXFAC ] = idx_fac;
   KS_idxs[KS_OFFSET_INLABCD] = encode4(  inla, inlb, inlc, inld );
   KS_idxs[KS_OFFSET_LDABCD ] = encode4( ld_ac,ld_ad,ld_bc,ld_bd );
   KS_idxs[KS_OFFSET_OFFAC  ] = offset_ac_L_set;
   KS_idxs[KS_OFFSET_OFFAD  ] = offset_ad_L_set;
   KS_idxs[KS_OFFSET_OFFBC  ] = offset_bc_L_set;
   KS_idxs[KS_OFFSET_OFFBD  ] = offset_bd_L_set;
   KS_idxs[KS_OFFSET_TALL   ] = encode4(  (int)Tac, (int)Tad, (int)Tbc, (int)Tbd );
   KS[L].insert( KS[L].end(), KS_idxs, KS_idxs+KS_SIZE );


}

void AIS::add_set(){
   prm_in_set = 0;
   n_set += 1;
}

double Lcorrection(int L){
   if (L == 0){
       return 1.0;
   } else if ( L == 1 ) {
       return sqrt(3.);
   } else {
        return 2. * sqrt(M_PI);
   }
}

__global__ void apply_correction( int N, double* mem_dev, double corr ){
   for( int i= blockIdx.x*blockDim.x + threadIdx.x ; i < N ; i += blockDim.x*gridDim.x ){
      mem_dev[i] *= corr;
   }
}


void AIS::compute_max_vector_size(){

   max_integral_scratch_size = 0;
   max_plan_size = 0;
   max_PMI_size = 0;
   max_FVH_size = 0;
   max_SPH_size = 0;
   max_KS_size  = 0;
   max_TRA_size = 0;
   out_size = 0;

   for ( unsigned int L : encoded_moments ){

      int la,lb,lc,ld;
      decodeL(L,&la,&lb,&lc,&ld);
      std::vector<int> * plan = NULL ;
      unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
      plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );

      size_t integral_scratch_size = Fm_size[L] + AC_size[L] + ABCD_size[L] + ABCD0_size[L] + SPHER_size[L] ;

      max_integral_scratch_size = max( max_integral_scratch_size, integral_scratch_size );
      max_plan_size  = max(max_plan_size, plan->size());
      max_PMI_size   = max(max_PMI_size,  (size_t)PMI_SIZE * (1+offset_V[L]) );
      max_FVH_size   = max(max_FVH_size,  FVH[L].size());
      max_SPH_size   = max(max_SPH_size,  SPH[L].size());
      max_KS_size    = max( max_KS_size,   KS[L].size());
      max_TRA_size   = max(max_TRA_size,  TRA[L].size());

      out_size += OUT_size[L];

   }

}

size_t AIS::memory_needed( ){
   compute_max_vector_size();
   size_t tmp = (max_plan_size + max_PMI_size + max_FVH_size + max_SPH_size + max_KS_size + max_TRA_size);
   size_t add1 = tmp * sizeof(unsigned int);
   size_t add2 = (max_integral_scratch_size + out_size) * sizeof(double);
   size_t add3 = 2 * FP_size * sizeof(double);
   return add1+add2+add3;
}

void AIS::set_P( std::vector<double> & P_ ){
   cout << " Setting P with size " << P_.size() << endl ;
   nspin = 1;
   P_a = P_.data();
   FP_size = P_.size();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&P_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( P_a_dev, P_.data(), sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
}

void AIS::set_P( std::vector<double> & P_a_, std::vector<double> & P_b_ ){
   cout << " Setting P Polarized with size " << P_a_.size() << " " << P_b_.size() << endl ;
   nspin = 2;
   P_a = P_a_.data();
   P_b = P_b_.data();
   FP_size = P_a_.size();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&P_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( P_a_dev, P_a_.data(), sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&P_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( P_b_dev, P_b_.data(), sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
}

void AIS::set_K( std::vector<double> & K_ ){
   cout << " Setting K with size " << K_.size() << endl ;
   assert( nspin == 1 );
   K_a = K_.data();
   FP_size = K_.size();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_a_dev, K_.data(), sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
}

void AIS::set_K( std::vector<double> & K_a_, std::vector<double> & K_b_ ){
   cout << " Setting K Polarized with size " << K_a_.size() << " " << K_b_.size() << endl ;
   assert( nspin == 2 );
   K_a = K_a_.data();
   K_b = K_b_.data();
   FP_size = K_a_.size();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_a_dev, K_a_.data(), sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_b_dev, K_b_.data(), sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
}

void AIS::get_K( std::vector<double> & K_a_ ){
   assert( nspin == 1 );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_a_.data(), K_a_dev, sizeof(double)*FP_size, cudaMemcpyDeviceToHost ));
}

void AIS::get_K( std::vector<double> & K_a_,  std::vector<double> & K_b_ ){
   assert( nspin == 2 );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_a_.data(), K_a_dev, sizeof(double)*FP_size, cudaMemcpyDeviceToHost ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_b_.data(), K_b_dev, sizeof(double)*FP_size, cudaMemcpyDeviceToHost ));
}


void AIS::reset_indices(){
   max_integral_scratch_size = 0;
   max_plan_size = 0;
   max_PMI_size = 0;
   max_FVH_size = 0;
   max_SPH_size = 0;
   max_TRA_size = 0;
   out_size = 0;

   for ( unsigned int L : encoded_moments ){
      PMI[L].clear();
      FVH[L].clear();
      SPH[L].clear();
      TRA[L].clear();
      offset_F[L] = 0;
      offset_V[L] = 0;
      offset_G[L] = 0;
      offset_Q[L] = 0;
      offset_T[L] = 0;
      Fm_size[L] = 0;
      AC_size[L] = 0;
      ABCD_size[L] = 0;
      ABCD0_size[L] = 0;
      SPHER_size[L] = 0;
      OUT_size[L] = 0;
   }

   dest=0;
   n_set = 0;
   prm_in_set = 0;
   n_prm = 0;
   p0 = 0;
   cell_in_set = 0;

   encoded_moments.clear();

}

void AIS::dispatch( bool skip_cpu ){


   Timer timer;
   Timer timer2;
   timer2.start();

   compute_max_vector_size();

//   cout << "Dispatch: Will compute " << out_size << " values " << endl;
   OUT.resize(out_size);

   int ftable_ld = 0; // ld of table for fgamma
   int nmax = 21;
   double tdelta = 0.1;  // hardcoded in source code
   double tmin = 0.; // hardcoded in source code
   double tmax = 12.;  // hardcoded in source code
   int n = nmax + 6; // 6 is the max taylor expansion. Also hardcoded
   int itabmin = floor(tmin/tdelta); // === 0
   int itabmax = int((tmax - tmin)/tdelta); // === 120
   int nelem = (itabmax - itabmin + 1 ) * (n+1); // === 121*(n+1) == 121*ftable_ld
   double* ftable = create_md_ftable( nmax, tmin, tmax, tdelta, &ftable_ld);

   size_t tot_mem = 0;
   tot_mem += OUT.size()*sizeof(double);
   tot_mem += ua.internal_buffer.size()*sizeof(double) ;
   tot_mem += sizeof(double)*max_integral_scratch_size;
   tot_mem += sizeof(int)*max_plan_size          ;
   tot_mem += sizeof(unsigned int)*max_PMI_size  ; 
   tot_mem += sizeof(unsigned int)*max_FVH_size  ;
   tot_mem += sizeof(unsigned int)*max_SPH_size  ;
   tot_mem += sizeof(unsigned int)*max_KS_size  ;
   tot_mem += sizeof(unsigned int)*max_TRA_size  ;
   tot_mem += sizeof(unsigned int)*(9+nelem+245) ;
   tot_mem += 2 * sizeof(double)*(FP_size) ;

   if ( first ){
      cout << "Memory use: (MB)" << endl;
      cout << " OUT DAT SCRT PLAN PMI FVH SPH KS TRA AUX FK TOT" << endl;
      first = false;
   }
   cout << int( OUT.size()*sizeof(double) *1.e-6 ) << " ";
   cout << int( ua.internal_buffer.size()*sizeof(double) *1.e-6 ) << " ";
   cout << int( sizeof(double)*max_integral_scratch_size *1.e-6 ) << " ";
   cout << int( sizeof(int)*max_plan_size          *1.e-6 ) << "  ";
   cout << int( sizeof(unsigned int)*max_PMI_size  *1.e-6 ) << "  "; 
   cout << int( sizeof(unsigned int)*max_FVH_size  *1.e-6 ) << "  ";
   cout << int( sizeof(unsigned int)*max_SPH_size  *1.e-6 ) << "  ";
   cout << int( sizeof(unsigned int)*max_KS_size   *1.e-6 ) << "  ";
   cout << int( sizeof(unsigned int)*max_TRA_size  *1.e-6 ) << "  ";
   cout << int( sizeof(unsigned int)*(9+nelem+245) *1.e-6 ) << "  "; 
   cout << int( 2 * sizeof(double)*(FP_size)       *1.e-6 ) << "  "; 
   cout << int( tot_mem *1.e-6 ) << endl;

   std::vector<double> integral_scratch(max_integral_scratch_size);

   for ( unsigned int L : encoded_moments ){

      int la,lb,lc,ld,labcd;
      decodeL(L,&la,&lb,&lc,&ld);
      labcd = la+lb+lc+ld;
      int Nc = compute_Nc(la,lb,lc,ld);
      int Ns = compute_Ns(la,lb,lc,ld);

      double corr = Lcorrection(la)*Lcorrection(lb)*Lcorrection(lc)*Lcorrection(ld);

      std::vector<int> * plan = NULL ;
      unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;   
      plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );
      
      unsigned int Nprm   = offset_V[L];
      unsigned int Ncells = offset_F[L];
      unsigned int Nqrtt  = offset_Q[L];
      unsigned int Nshell = offset_T[L];

      double * Fm    = &integral_scratch[0];
      double * AC    = Fm    + Fm_size[L];
      double * ABCD  = AC    + AC_size[L];
      double * ABCD0 = ABCD  + ABCD_size[L];
      double * SPHER = ABCD0 + ABCD0_size[L];

      unsigned int * FVH_L = FVH[L].data();
      unsigned int * PMI_L = PMI[L].data();
      unsigned int * TRA_L = TRA[L].data();
      unsigned int * KS_L  =  KS[L].data();

      int * plan_L = plan->data();
      double * env = ua.internal_buffer.data();

///      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | ";
//      cout << "Computing " << Nprm << " prms " << Ncells << " cells " << Nqrtt << " qrtts " << Nshell << " shells ";
//      cout << " AC: " << AC_size[L] << " ABCD " << ABCD_size[L] << "/" << ABCD0_size[L] ;

      timer.start();

      if ( not skip_cpu ) {

         compute_Fm_batched_low(
            FVH_L, PMI_L, env, Fm, Nprm, labcd, periodic, cell_h, ftable, ftable_ld );

         compute_VRR_batched_low(
            Ncells, plan_L, PMI_L, FVH_L, Fm, env,
            AC, ABCD, vrr_blocksize, hrr_blocksize, labcd, numV, numVC );

         compute_HRR_batched_low(
            Ncells, plan_L, FVH_L, env, ABCD, ABCD0,
            hrr_blocksize, Nc, numVC, numVCH );

         compute_SPH_batched_low( Nqrtt, la, lb, lc, ld, ABCD0, SPHER, ABCD );

         for( unsigned int i=0; i < Nqrtt*Ns; i++ ){ SPHER[i] *= corr; }

         compute_KS( Nqrtt, KS_L, la,lb,lc,ld, P_a, SPHER, K_a, env );
         if ( nspin == 2 ){ compute_KS( Nqrtt, KS_L, la,lb,lc,ld, P_b, SPHER, K_b, env ); }

         compute_TRA_batched_low( Nshell, la, lb, lc, ld, TRA_L, SPHER, OUT.data() );
//         cout << " VALs: " << Fm[0] << " " << AC[0] << "  " << OUT[0] << endl;

      }

      timer.stop();

//      cout << " CPU KRNL " <<  " SL " << labcd << " " ;
//      cout << timer.elapsedMicroseconds() << " us " ;
//      cout << OUT_size[L] / timer.elapsedMicroseconds() * sizeof(double) / 1.e3 << " GB/s" ;
//      cout << endl;


      record_of_out_sizes[L].push_back(OUT_size[L]);
      record_of_times_cpu[L].push_back(timer.elapsedMicroseconds());
   }

   timer2.stop();
//   cout << "DISPATCH CPU " << timer2.elapsedMilliseconds() << endl;


   timer2.start();

   double *data_dev, *cell_h_dev, *ftable_dev, *OUT_dev, *C2S_dev;

   timer.start();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&OUT_dev, sizeof(double)*OUT.size() )); // TODO mvoe alloc to be concurr to compute_Fm
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&data_dev, sizeof(double)*(ua.internal_buffer.size()) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&cell_h_dev, sizeof(double)*(9) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&ftable_dev, sizeof(double)*(nelem) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&C2S_dev, sizeof(double)*245 ));
   timer.stop();
//   cout << "I Malloc " << timer.elapsedMilliseconds() << " ms " << endl;

//   cout << "Copying Ua " << sizeof(unsigned int)*(ua.internal_buffer.size()) * 1.e-6 << " MBytes " << endl;
   timer.start();
   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      data_dev, ua.internal_buffer.data(), sizeof(double)*(ua.internal_buffer.size()), cudaMemcpyHostToDevice ));

   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      cell_h_dev, cell_h, sizeof(double)*(9), cudaMemcpyHostToDevice ));

   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      ftable_dev, ftable, sizeof(double)*(nelem), cudaMemcpyHostToDevice ));

   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      C2S_dev, c2s, sizeof(double)*245, cudaMemcpyHostToDevice ));
   timer.stop();
//   cout << "F COPY " << timer.elapsedMilliseconds() << " ms " << endl;


   double *integral_scratch_dev;
   unsigned int *PMI_dev, *FVH_dev, *SPH_dev, *KS_dev, *TRA_dev;
   int *plan_dev;


   timer.start();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&integral_scratch_dev,    sizeof(double)*max_integral_scratch_size ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&plan_dev,  sizeof(int)*max_plan_size ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&PMI_dev, sizeof(unsigned int)*max_PMI_size )); 
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&FVH_dev, sizeof(unsigned int)*max_FVH_size ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&SPH_dev, sizeof(unsigned int)*max_SPH_size )); 
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&KS_dev , sizeof(unsigned int)*max_KS_size  )); 
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&TRA_dev, sizeof(unsigned int)*max_TRA_size )); 
   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
   timer.stop();
//   cout << "L Malloc " << timer.elapsedMilliseconds() << " ms " << endl;

   timer.start();
   cublasHandle_t cublas_handle;
   CUBLAS_GPU_ERR_CHECK( cublasCreate(&cublas_handle) );
   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
   timer.stop();
//   cout << "CUBLAS HANDLE CREATE " << timer.elapsedMilliseconds() << " ms " << endl;
  
   for ( unsigned int L : encoded_moments ){
      int la,lb,lc,ld,labcd;
      decodeL(L,&la,&lb,&lc,&ld);
      labcd = la+lb+lc+ld;
      int Nc = compute_Nc(la,lb,lc,ld);
      int Ns = compute_Ns(la,lb,lc,ld);
      double corr = Lcorrection(la)*Lcorrection(lb)*Lcorrection(lc)*Lcorrection(ld);

      std::vector<int> * plan = NULL ;
      unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;   
      plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );

      unsigned int Nprm   = offset_V[L];
      unsigned int Ncells = offset_F[L];
      unsigned int Nqrtt  = offset_Q[L];
      unsigned int Nshell = offset_T[L];

      double* Fm_dev    = &integral_scratch_dev[0];
      double* AC_dev    = Fm_dev    + Fm_size[L];
      double* ABCD_dev  = AC_dev    + AC_size[L];
      double* ABCD0_dev = ABCD_dev  + ABCD_size[L];
      double* SPHER_dev = ABCD0_dev + ABCD0_size[L];

//      cout << "Computing " << Ncells << " cells" ;
//      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | " << OUT_size[L] << " | " ;
//      cout << " AC: " << AC_size[L] << " ABCD " << ABCD_size[L] << "/" << ABCD0_size[L] ;
//      cout << " <<< " << Fm_numblocks << "," << Fm_blocksize << " >>> " ;
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         plan_dev, plan->data(), sizeof(int)*(plan->size()), cudaMemcpyHostToDevice ));
//      cout << endl << sizeof(unsigned int)*(PMI_SIZE*Nprm) << " " << sizeof(unsigned int)*max_PMI_size << endl;
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         PMI_dev, PMI[L].data(), sizeof(unsigned int)*(PMI_SIZE*Nprm), cudaMemcpyHostToDevice )); 
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         FVH_dev, FVH[L].data(), sizeof(unsigned int)*(FVH[L].size()), cudaMemcpyHostToDevice ));
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
          KS_dev,  KS[L].data(), sizeof(unsigned int)*( KS[L].size()), cudaMemcpyHostToDevice )); 
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         TRA_dev, TRA[L].data(), sizeof(unsigned int)*(TRA[L].size()), cudaMemcpyHostToDevice ));

      timer.start();

      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
      int Fm_blocksize = 256;
      int Fm_numblocks = (Nprm+Fm_blocksize-1)/Fm_blocksize;

      compute_Fm_batched_low_gpu<<<Fm_numblocks,Fm_blocksize>>>(
         FVH_dev, PMI_dev, data_dev, Fm_dev, Nprm, labcd,
         periodic, cell_h_dev, ftable_dev, ftable_ld );
      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

      compute_VRR_batched_gpu_low<<<Ncells,32>>>(
         Ncells, plan_dev, PMI_dev, FVH_dev, Fm_dev, data_dev,
         AC_dev, ABCD_dev, vrr_blocksize, hrr_blocksize, labcd, numV, numVC );
     
      compute_HRR_batched_gpu_low<<<Ncells,128>>>(
         Ncells, plan_dev, FVH_dev, data_dev, ABCD_dev, ABCD0_dev,
         hrr_blocksize, Nc, numVC, numVCH );

      // Note: we need to DeviceSynchronize before going from kernels to cublas. TODO actually check it is true
      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

      compute_SPH_batched_gpu_alt ( Nqrtt, la, lb, lc, ld, ABCD0_dev, SPHER_dev, ABCD_dev, C2S_dev, cublas_handle );

      int corrBS = 64;
      int corrNB = (Nqrtt*Ns+corrBS-1)/corrBS;
      apply_correction<<<corrNB,corrBS>>>( Nqrtt*Ns, SPHER_dev, corr );

      compute_KS_gpu<<<Nqrtt,128>>>( Nqrtt, KS_dev, la,lb,lc,ld, P_a_dev, SPHER_dev, K_a_dev, data_dev );
      if ( nspin == 2 ){
         compute_KS_gpu<<<Nqrtt,128>>>( Nqrtt, KS_dev, la,lb,lc,ld, P_b_dev, SPHER_dev, K_b_dev, data_dev );
      }

      compute_TRA_batched_gpu_low<<<Nshell,128>>>( Nshell, la, lb, lc, ld, TRA_dev, SPHER_dev, OUT_dev );

      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
      timer.stop();

//      cout << " GPU KRNL " <<  " SL " << labcd << " " ;
//      cout << timer.elapsedMicroseconds() << " us " ;
//      cout << OUT_size[L] / timer.elapsedMicroseconds() * sizeof(double) / 1.e3 << " GB/s" ;
//      cout << endl;
//      cout.flush();

      record_of_times_gpu[L].push_back(timer.elapsedMicroseconds());

   }
   timer.start();
   std::vector<double> OUT_from_gpu( OUT.size() );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( OUT_from_gpu.data(), OUT_dev, sizeof(double)*(OUT.size()), cudaMemcpyDeviceToHost ));
   timer.stop();

   timer.start();
   std::vector<double> F_a_from_gpu( FP_size );
   std::vector<double> F_b_from_gpu( FP_size );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( F_a_from_gpu.data(), K_a_dev, sizeof(double)*(FP_size), cudaMemcpyDeviceToHost ));
   if ( nspin == 2 ){
      CUDA_GPU_ERR_CHECK( cudaMemcpy( F_b_from_gpu.data(), K_b_dev, sizeof(double)*(FP_size), cudaMemcpyDeviceToHost ));
   }

   timer.stop();
//   cout << "IJKL COPY " << sizeof(double)*(OUT.size()) * 1.e-6 << " MB " << timer.elapsedMilliseconds() << " ms " << endl;


   timer2.stop();
//   cout << "DISPATCH GPU " << timer2.elapsedMilliseconds() << " ms" << endl;

   if ( not skip_cpu ){
      double diff_sum = 0.0;
      double adiff_sum = 0.0;
      int nerrors = 0;
      int Nval = int(OUT.size());
      for( int i=0; i < Nval; i++ ){
         double ref = OUT[i];
         double val = OUT_from_gpu[i];
         double diff = ref - val;
         double adiff = abs(diff);
         diff_sum += diff;
         adiff_sum += adiff;

         if ( adiff > 1.e-12 ){
            nerrors++;
            double ratio = 1.0;
            if ( abs(ref) > 0. ){ ratio = val / ref ; }
            cout << "I: CPU - GPU: Error at " << i << " " << ref << " " << val 
                 << " " << diff << " " << ratio << " " << endl ;
            if ( nerrors >= 100 ){
               cout << " TOO MANY ERRORS ! EXITING NOW " << endl;
               exit( EXIT_FAILURE );
            }
         }
      }


      if (nerrors != 0 ){
         cout << "E[ CPU-GPU ] = " <<  diff_sum / Nval << endl;
         cout << "E[|CPU-GPU|] = " << adiff_sum / Nval << endl;
         exit( EXIT_FAILURE );
      }
   } else {
      // copy gpu to cpu to test later against ref
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         OUT.data(), OUT_dev, sizeof(double)*(OUT.size()), cudaMemcpyDeviceToHost ));
   }

   if ( not skip_cpu ){
      double diff_sum = 0.0;
      double adiff_sum = 0.0;
      int nerrors = 0;
      int Nval = int(FP_size);
      for( int i=0; i < Nval; i++ ){
         double ref = K_a[i];
         double val = F_a_from_gpu[i];
         double diff = ref - val;
         double adiff = abs(diff);
         diff_sum += diff;
         adiff_sum += adiff;

         if ( adiff > 1.e-12 ){
            nerrors++;
            double ratio = 1.0;
            if ( abs(ref) > 0. ){ ratio = val / ref ; }
            cout << "F: CPU - GPU: Error at " << i << " " << ref << " " << val 
                 << " " << diff << " " << ratio << " " << endl ;
            if ( nerrors >= 100 ){
               cout << " TOO MANY ERRORS ! EXITING NOW " << endl;
               exit( EXIT_FAILURE );
            }
         }

         if ( nspin == 2 ){
            ref = K_b[i];
            val = F_b_from_gpu[i];
            diff = ref - val;
            adiff = abs(diff);
            diff_sum += diff;
            adiff_sum += adiff;

            if ( adiff > 1.e-12 ){
               nerrors++;
               double ratio = 1.0;
               if ( abs(ref) > 0. ){ ratio = val / ref ; }
               cout << "Fb: CPU - GPU: Error at " << i << " " << ref << " " << val
                    << " " << diff << " " << ratio << " " << endl ;
               if ( nerrors >= 100 ){
                  cout << " TOO MANY ERRORS ! EXITING NOW " << endl;
                  exit( EXIT_FAILURE );
               }
            }         
         }

      }


      if (nerrors != 0 ){
         cout << "E[ CPU-GPU ] = " <<  diff_sum / Nval << endl;
         cout << "E[|CPU-GPU|] = " << adiff_sum / Nval << endl;
         exit( EXIT_FAILURE );
      }
   } else {
   // no point in copying a partial F matrix from this batch
//      // copy gpu to cpu to test later against ref
//      CUDA_GPU_ERR_CHECK( cudaMemcpy(
//         F.data(), OUT_dev, sizeof(double)*(OUT.size()), cudaMemcpyDeviceToHost ));
   }


   CUBLAS_GPU_ERR_CHECK( cublasDestroy(cublas_handle) );
   CUDA_GPU_ERR_CHECK( cudaFree(OUT_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(data_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(cell_h_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(ftable_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(C2S_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(integral_scratch_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(plan_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(PMI_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(FVH_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(SPH_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(TRA_dev) );

   reset_indices();

}


template < class T > 
T sum( std::vector< T > x ){
   T ret = (T)0;
   for( auto i : x ){
      ret += i;
   }
   return ret;
}

void AIS::report_througput(bool skip_cpu){
   cout << " la lb lc ld L OUT(MB) ";
   if ( not skip_cpu ) { cout << "CPU_Throughput(GB/s) " ; }
   cout << " GPU_Throughput(GB/s) " << endl;
   for ( auto L : all_moments ){

      int la,lb,lc,ld,labcd;
      decodeL(L,&la,&lb,&lc,&ld);
      labcd = la+lb+lc+ld;
 
      size_t sum_output_size = sum( record_of_out_sizes[L] );
      double sum_times_cpu   = sum( record_of_times_cpu[L] );
      double sum_times_gpu   = sum( record_of_times_gpu[L] );

      double avg_thr_cpu = sum_output_size / sum_times_cpu * sizeof(double) / 1.e3;
      double avg_thr_gpu = sum_output_size / sum_times_gpu * sizeof(double) / 1.e3;

      cout << la << " " << lb << " " << lc << " " << ld << " " << labcd << " " ;
      cout << sum_output_size / 1.e6 << " " ;
      if ( not skip_cpu ) { cout << avg_thr_cpu << " " ; }
      cout << avg_thr_gpu ;
      cout << endl;

   }
}

