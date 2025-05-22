#include <vector>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include "plan.h"
#include "util.h"
#include "timer.h"
#include "UniqueArray.h"
#include "BW_by_patch.h"
#include "t_c_g0_n.h"
#include "compute_Fm.h"
#include "compute_VRR.h"
#include "compute_VRR2.h"
#include "compute_ECO.h"
#include "compute_HRR.h"
#include "compute_SPH.h"
#include "fgamma.h"
#include "libGint.h"
#include "c2s.h"
#include <omp.h>

#include <iomanip>      // std::setprecision

using std::max;


std::vector<LibGint_shared> libGint::shared_obj_ptr; // static // 

void libGint::show_state(){
   for (unsigned int L : encoded_moments ){
      int la,lb,lc,ld;
      decodeL(L,&la,&lb,&lc,&ld);
      cout << " Moments : " << la << " " << lb << " " << lc << " " << ld << endl;
   }
}


void libGint::init(){
//   PUSH_RANGE("libGint init",1);
   // IF the static (shared over OMP) persistent vector is not enough, (re)create it
   // ELSE read the values (hopefully) already saved there
   if ( shared_obj_ptr.size() < (std::vector<LibGint_shared>::size_type) omp_get_num_threads() ){
   // barrier necessary to avoid thread > 0 coming to check the if after master has already allocated the shared obj memory
#pragma omp barrier
#pragma omp master
      shared_obj_ptr = std::vector<LibGint_shared>(omp_get_num_threads());
#pragma omp barrier
      CUDA_GPU_ERR_CHECK( cudaStreamCreate( &cuda_stream ));
      CUBLAS_GPU_ERR_CHECK( cublasCreate(&cublas_handle) );
      CUBLAS_GPU_ERR_CHECK( cublasSetStream( cublas_handle, cuda_stream ));
      shared_obj_ptr[ my_thr ] = { &cublas_handle, &cuda_stream };
      potential_type = COULOMB; // default
   } else {
      // Use the persistent vector to populate the class members
      cublas_handle = * ( shared_obj_ptr[my_thr].cublas_handle );
      cuda_stream   = * ( shared_obj_ptr[my_thr].cuda_stream   );
      // TODO may want to check if we can save other stuff other than just the stram and cublas_h
      potential_type = COULOMB; // default
   }
//   POP_RANGE; // libGint init
}

void libGint::set_Potential_Truncated( double R_cut_, double * C0_, int ld_C0_, int C0_size_ ){
   C0_size = C0_size_;
   ld_C0 = ld_C0_;
   R_cut = R_cut_;
   C0 = C0_;
   potential_type = TRUNCATED;

#pragma omp single copyprivate(C0_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&C0_dev, C0_size * sizeof(double) ) );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( C0_dev, C0, C0_size * sizeof(double), cudaMemcpyHostToDevice ));   
   }
   size_t to_patch_size = POT_TRUNC_N1 * POT_TRUNC_N2 * sizeof(int);

#pragma omp single copyprivate(x12_to_patch_low_R, x12_to_patch_high_R)
   {
   x12_to_patch_low_R  = (int*) malloc( to_patch_size );
   x12_to_patch_high_R = (int*) malloc( to_patch_size );
   fill_x12_to_patch( POT_TRUNC_N1, POT_TRUNC_N2, x12_to_patch_low_R, x12_to_patch_high_R );
   }

#pragma omp single copyprivate(x12_to_patch_low_R_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&x12_to_patch_low_R_dev, to_patch_size ) );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( x12_to_patch_low_R_dev, x12_to_patch_low_R, to_patch_size, cudaMemcpyHostToDevice ));   
   }

#pragma omp single copyprivate(x12_to_patch_high_R_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&x12_to_patch_high_R_dev, to_patch_size ) );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( x12_to_patch_high_R_dev, x12_to_patch_high_R, to_patch_size, cudaMemcpyHostToDevice ));   
   }

#pragma omp single copyprivate(BW_by_patch_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&BW_by_patch_dev, 207*4 * sizeof(double) ) );
   CUDA_GPU_ERR_CHECK( cudaMemcpy( BW_by_patch_dev, BW_by_patch, 207*4 * sizeof(double), cudaMemcpyHostToDevice ));
   }
}

void libGint::set_hf_fac(double fac){ hf_fac = fac; }

// Set the max gpu memory available to each MPI rank
// this memory will be shared by all omp threads
// TODO add hook to read fraction mem from cp2k input
void libGint::set_max_mem(int max_mem_){
   size_t max_dat_mem = size_t(max_mem_) * 1024 * 1024 * FRACTION_DAT_MEM;
   size_t max_idx_mem = size_t(max_mem_) * 1024 * 1024 * FRACTION_IDX_MEM;
   max_dat_mem_per_thread = max_dat_mem / Nomp;
   max_idx_mem_per_thread = max_idx_mem / Nomp;
}

void libGint::set_Atom( int i, double* R_, double* Z_, int np_ ){
   // why not pushback ?
   if ( i >= (int) idx_R.size() ){ idx_R.resize(i+1); }
   if ( i >= (int) idx_Z.size() ){ idx_Z.resize(i+1); }
   if ( i >= (int) np.size() ){ np.resize(i+1); }

   unsigned int tmp_idx_R = ua.add( R_, 3 );
   unsigned int tmp_idx_Z = ua.add( Z_, np_ );
   idx_R[i] = tmp_idx_R;
   idx_Z[i] = tmp_idx_Z;
   np[i] = np_;
   if ( i < (int) all_l.size() ) {
      all_l[i].clear();
      all_nl[i].clear();
      all_idx_K[i].clear();
   }
   // precompute the max possible number of primitives in a set
   if ( np_ > max_n_prm ){
      max_n_prm = np_;
      prm_tmp_list.resize( max_n_prm*max_n_prm*max_n_prm*max_n_prm*4 );
      new_prm_tmp_list.resize( max_n_prm*max_n_prm*max_n_prm*max_n_prm*PMX_SIZE );
   }
}

void libGint::set_Atom_L( int i, int l_, int nl_, double* K_ ){
   if ( i >= (int) all_l.size()     ){ all_l.resize( i+1 ); }
   if ( i >= (int) all_nl.size()    ){ all_nl.resize( i+1 ); }
   if ( i >= (int) all_idx_K.size() ){ all_idx_K.resize( i+1 ); }
   all_l[i].push_back(l_);
   all_nl[i].push_back(nl_);
   unsigned int idx_K = ua.add( K_, np[i]*nl_);
//   cout << " Idx K of " << i << " " << l_ << " " << nl_ << " " << K_ << " is " << idx_K << endl; cout.flush();

   if ( unique_K_set.count(idx_K) == 0 ){
//      cout << " Adding " << idx_K << " " << nl_ << "." << np[i] << " as # " << unique_K_list.size() << endl;
      unique_K_list.push_back( idx_K );
      l_from_K_list.push_back( nl_);
      p_from_K_list.push_back( np[i] );
      unique_K_set.insert( idx_K );
   }
   
   all_idx_K[i].push_back( idx_K );
   data_size = ua.internal_buffer.size()*sizeof(double);
   
}

// Add the index of the gaussian exponenets, encoded into a single uint
void libGint::add_prm( const int ipa, const int ipb, const int ipc, const int ipd ){
//   unsigned int piabcd = encode4(ipa,ipb,ipc,ipd);
//   prm_tmp_list[ n_prm ] = piabcd ;
   prm_tmp_list[ n_prm*4+0 ] = ipa;
   prm_tmp_list[ n_prm*4+1 ] = ipb;
   prm_tmp_list[ n_prm*4+2 ] = ipc;
   prm_tmp_list[ n_prm*4+3 ] = ipd;
   n_prm++;
}

void libGint::add_shell ( int i, int j, int k, int l, int n1, int n2 ){

   if (n_prm == 0){ return; }

   int npa = np[i];
   int npb = np[j];
   int npc = np[k];
   int npd = np[l];
   int npcd = npc*npd;
   int npbcd = npb*npcd;
   for ( unsigned int i_prm = 0 ; i_prm < n_prm ; i_prm++ ){
      int ipa = prm_tmp_list[ i_prm*4+0 ];
      int ipb = prm_tmp_list[ i_prm*4+1 ];
      int ipc = prm_tmp_list[ i_prm*4+2 ];
      int ipd = prm_tmp_list[ i_prm*4+3 ];
      new_prm_tmp_list[ i_prm * PMX_SIZE + PMX_OFFSET_IPZN] = encode4(ipa,ipb,ipc,ipd);
      new_prm_tmp_list[ i_prm * PMX_SIZE + PMX_OFFSET_OGP ] = ipa*npbcd + ipb*npcd + ipc*npd + ipd;
   }


   // number of angular moments in the i,j,k,l set
   int nnla = all_l[i].size();
   int nnlb = all_l[j].size();
   int nnlc = all_l[k].size();
   int nnld = all_l[l].size();
   // For each combination of angular moment, add work to be done
   // to the candidate vectors this_set_* 
   // TODO transpose all_idx_K[i][idx_la],all_nl[i][idx_la],all_l[i][idx_la] to all_idx_K_nl_l[i][idx_la]
   for( int idx_la=0; idx_la < nnla; idx_la++ ){
      unsigned int idx_Ka = all_idx_K[i][idx_la];
      int nla = all_nl[i][idx_la];
      int la = all_l[i][idx_la];
      for( int idx_lb=0; idx_lb < nnlb; idx_lb++ ){
         unsigned int idx_Kb = all_idx_K[j][idx_lb];
         int nlb = all_nl[j][idx_lb];
         int lb = all_l[j][idx_lb];
         for( int idx_lc=0; idx_lc < nnlc; idx_lc++ ){
            unsigned int idx_Kc = all_idx_K[k][idx_lc];
            int nlc = all_nl[k][idx_lc];
            int lc = all_l[k][idx_lc];
            for( int idx_ld=0; idx_ld < nnld; idx_ld++ ){
               unsigned int idx_Kd = all_idx_K[l][idx_ld];
               int nld = all_nl[l][idx_ld];
               int ld = all_l[l][idx_ld];

               const unsigned int L = encodeL(la,lb,lc,ld);
               const int labcd = la+lb+lc+ld;
               int F_size = Fsize(labcd);

               const unsigned int Of = this_set_offset_F[L];
               const unsigned int Ov = this_set_offset_V[L];
               const unsigned int Og = this_set_offset_G[L];
               const unsigned int Oq = this_set_offset_Q[L];

               const unsigned int elabcd = encode_shell(nla,nlb,nlc,nld,n1,n2);
               const unsigned int epabcd = encode4(np[i],np[j],np[k],np[l]);
               const unsigned int nlabcd = nla*nlb*nlc*nld;
               const unsigned int npabcd = np[i]*np[j]*np[k]*np[l];

               unsigned int idx_Kabcd = all_idx_Kabcd[ four_uint_tuple(idx_Ka, idx_Kb, idx_Kc, idx_Kd) ];
//               unsigned int idx_Kr = offset_K[L];
               unsigned int idx_Kr = 0;
//               cout << " K " << i << " " << j << " " << k << " " << l << " [" << idx_Kabcd << "] = " << ua.internal_buffer[idx_Kabcd] << endl;

//               TODO FUTURE add ij and kl detailed screening coeff
//               const unsigned int bra_Radii_ptr  = MAX_N_PRM * MAX_N_PRM * ( kind_set[i] * number_of_sets + kind_set[j] );
//               const unsigned int ket_Radii_ptr  = MAX_N_PRM * MAX_N_PRM * ( kind_set[k] * number_of_sets + kind_set[l] );
               // if this is the first time we see this combination
               // of angular moments, prepare and save the plan
               if ( all_moments[L] == false ){
                  std::vector<int> * plan = NULL ;
                  unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
                  plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );
                  all_vrr_blocksize[L] = vrr_blocksize;
                  all_hrr_blocksize[L] = hrr_blocksize;
                  all_moments[L] = true;
                  max_plan_size  = max(max_plan_size,  plan->size());     // cell Gvectors     gamma   C0
                  AUX_size = sizeof(int)*(max_plan_size) + sizeof(double)*(2*9+3*max_ncells+3388+245+C0_size/Nomp);
               }
               // many, many idxs for the calculation of the integrals
               // later they will be copied to the gpu memory
               const unsigned int tmp[FVH_SIZE] = {
                  Ov, Og, Oq, n_prm, idx_R[i], idx_R[j], idx_R[k], idx_R[l],
                  idx_Z[i], idx_Z[j], idx_Z[k], idx_Z[l], idx_Kabcd, idx_Kr,
                  elabcd, epabcd, nlabcd, npabcd };

               this_set_L.insert(L);
               this_set_OF[L].insert(  this_set_OF[L].end(), n_prm, Of );
               this_set_PMX[L].insert( this_set_PMX[L].end(), &new_prm_tmp_list[0], &new_prm_tmp_list[n_prm*PMX_SIZE] );
               this_set_FVH[L].insert( this_set_FVH[L].end(), tmp, tmp+FVH_SIZE );

               this_set_Fm_size[L] += (F_size * n_prm * max_ncells)*sizeof(double);
               this_set_AC_size[L] += all_vrr_blocksize[L] * n_prm * max_ncells * sizeof(double);
               this_set_ABCD_size[L] += all_hrr_blocksize[L] * nlabcd * sizeof(double);
//               this_set_KRL_size[L] += n_prm * nlabcd * sizeof(double) ;

               this_set_idx_mem_needed[L] += (n_prm + n_prm*PMX_SIZE + FVH_SIZE) * sizeof(unsigned int);
               this_set_offset_G[L] += nlabcd;
               this_set_offset_V[L] += n_prm;
               this_set_offset_F[L] ++ ;
//               this_set_offset_K[L] += nlabcd*n_prm ;
               encoded_moments.insert(L);
   }}}}

   cell_in_set++;
   n_prm = 0;
}

void libGint::add_cell() {}

void libGint::add_qrt( int la, int lb, int lc, int ld, int nla, int nlb, int nlc, int nld ){
   unsigned int nlabcd = nla*nlb*nlc*nld;
   unsigned int L = encodeL(la,lb,lc,ld);
   this_set_ABCD0_size[L] = this_set_ABCD0_size[L] + compute_Nc(la,lb,lc,ld) * nlabcd * sizeof(double) ;
   this_set_SPHER_size[L] = this_set_SPHER_size[L] + compute_Ns(la,lb,lc,ld) * nlabcd * sizeof(double) ;

//   this_set_dat_mem_needed[L] += compute_Nc(la,lb,lc,ld) * nlabcd * sizeof(double);
//   this_set_dat_mem_needed[L] += compute_Ns(la,lb,lc,ld) * nlabcd * sizeof(double);
   this_set_offset_Q[L] += nlabcd ;
}

void libGint::add_qrtt(
      double symm_fac, int la, int lb, int lc, int ld, 
      int inla, int inlb, int inlc, int inld,
      int ld_ac, int ld_ad, int ld_bc, int ld_bd, 
      unsigned int offset_ac_L_set, unsigned int offset_ad_L_set, 
      unsigned int offset_bc_L_set, unsigned int offset_bd_L_set, 
      bool Tac, bool Tad, bool Tbc, bool Tbd ){ 

   // The symm_fac can be only 2,1,0.5 or 0.25 (or 0 in dev version). 
   // We assign them to 0,1,2 and 3 (and 4) respectively
   // We encode it here, decode it inside compute_KS

   unsigned int idx_fac = 0 ;
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
   KS_idxs[KS_OFFSET_TALL   ] = encode4( (int)Tac, (int)Tad, (int)Tbc, (int)Tbd );

   this_set_KS[L].insert( this_set_KS[L].end(), KS_idxs, KS_idxs+KS_SIZE );
   this_set_idx_mem_needed[L] += KS_SIZE * sizeof(unsigned int);
//   if ( la == 3 and lb == 3 and lc == 3 and ld == 3 ){
//      cout << la<<lb<<lc<<ld<<" " << offset_ac_L_set << " " << offset_ad_L_set << " " << offset_bc_L_set << " " << offset_bd_L_set << endl;
//   }

}

void libGint::add_set(){
   
//   set_timer.start();
   for ( unsigned int L : this_set_L ){

#ifdef LIBGINT_INTERNAL_DEBUG
      // Check if this set is even possible
      size_t this_set_dat_mem_needed_1 = max( this_set_Fm_size[L], max( this_set_ABCD_size[L], this_set_SPHER_size[L] ));
      size_t this_set_dat_mem_needed_2 = max( this_set_AC_size[L], 2 * this_set_ABCD0_size[L] );
      this_set_dat_mem_needed[L] = this_set_dat_mem_needed_1 + this_set_dat_mem_needed_2;
      if ( this_set_dat_mem_needed[L] >= max_dat_mem_per_thread ){
         cout << " Calculation may fail: this set L: " << L << " requires at least " <<  this_set_dat_mem_needed[L]/1024/1024 << " MiB for dat " 
              << this_set_Fm_size[L] << " " <<  this_set_AC_size[L] << " " <<  this_set_ABCD_size[L] << " " << this_set_ABCD0_size[L] << endl;
      }
      if ( this_set_idx_mem_needed[L] >= max_idx_mem_per_thread ){
         cout << " Calculation may fail: this set L: " << L << " requires at least " <<  this_set_idx_mem_needed[L]/1024/1024 << " MiB for idx" << endl;
      }
#endif // LIBGINT_INTERNAL_DEBUG

      // Check if we can fit this set in the list of sets to do
      size_t Fm_size_L_p = Fm_size[L]+this_set_Fm_size[L];
      size_t AC_size_L_p = AC_size[L]+this_set_AC_size[L];
      size_t ABCD_size_L_p = ABCD_size[L]+this_set_ABCD_size[L];
      size_t ABCD0_size_L_p = ABCD0_size[L]+this_set_ABCD0_size[L];
      size_t SPHER_size_L_p = SPHER_size[L]+this_set_SPHER_size[L];
      size_t dat_mem_needed_L_1 = max( Fm_size_L_p, max( ABCD_size_L_p, SPHER_size_L_p ));
      size_t dat_mem_needed_L_2 = max( AC_size_L_p, 2*ABCD0_size_L_p );
      size_t dat_mem_req_L = dat_mem_needed_L_1 + dat_mem_needed_L_2 ; 
      size_t idx_mem_req_L = idx_mem_needed[L] + this_set_idx_mem_needed[L];
      bool enough_dat_mem_L = (dat_mem_req_L <= max_dat_mem_per_thread);
      bool enough_idx_mem_L = (idx_mem_req_L <= max_idx_mem_per_thread);
      bool enough_mem_L = enough_dat_mem_L and enough_idx_mem_L;

      if ( enough_mem_L ){
         // Move the indices for this set forward, then include them in the larger lists
         for ( unsigned int idx1 = 0; idx1 < this_set_OF[L].size(); idx1++ ){
            this_set_OF[L][idx1] += offset_F[L];
         }
         for ( unsigned int idx2 = 0; idx2 < this_set_FVH[L].size()/FVH_SIZE; idx2++ ){
            this_set_FVH[L][idx2*FVH_SIZE+0] += offset_V[L];
            this_set_FVH[L][idx2*FVH_SIZE+1] += offset_G[L];
            this_set_FVH[L][idx2*FVH_SIZE+2] += offset_Q[L];
         }
         // TODO move to ::add_set_L(L);
         // Gotta love the stl
          OF[L].insert(  OF[L].end(),  this_set_OF[L].begin(),  this_set_OF[L].end() );
         PMX[L].insert( PMX[L].end(), this_set_PMX[L].begin(), this_set_PMX[L].end() );
         FVH[L].insert( FVH[L].end(), this_set_FVH[L].begin(), this_set_FVH[L].end() );
          KS[L].insert(  KS[L].end(),  this_set_KS[L].begin(),  this_set_KS[L].end() );
         // TODO transpose L from fast to slow variable
            Fm_size[L] +=    this_set_Fm_size[L];
            AC_size[L] +=    this_set_AC_size[L];
          ABCD_size[L] +=  this_set_ABCD_size[L];
         ABCD0_size[L] += this_set_ABCD0_size[L];
         SPHER_size[L] += this_set_SPHER_size[L];

         dat_mem_needed_L_1 = max( Fm_size[L], max( ABCD_size[L], SPHER_size[L] ));
         dat_mem_needed_L_2 = max( AC_size[L], 2*ABCD0_size[L] );
         dat_mem_needed[L] = dat_mem_needed_L_1 + dat_mem_needed_L_2;
         assert( dat_mem_needed[L] <= max_dat_mem_per_thread && " Strange. Pre check dat mem failed "); 
//         dat_mem_needed[L] += this_set_dat_mem_needed[L];
         idx_mem_needed[L] += this_set_idx_mem_needed[L];
         this_set_dat_mem_needed[L] = 0;
         this_set_idx_mem_needed[L] = 0;
         offset_V[L] += this_set_offset_V[L];
         offset_F[L] += this_set_offset_F[L];
         offset_G[L] += this_set_offset_G[L];
         offset_Q[L] += this_set_offset_Q[L];
         //
      } else {
//         cout << " L: " << L << " added to after list " << dat_mem_req_L << " " << idx_mem_req_L << endl;
         add_L_after_dispatch.insert(L);
      }
      // we still need to max check to make sure we dispatch early correctly
      max_dat_mem_needed = max(max_dat_mem_needed, dat_mem_req_L);
      max_idx_mem_needed = max(max_idx_mem_needed, idx_mem_req_L);
   }

   bool out_of_dat_mem = max_dat_mem_needed > max_dat_mem_per_thread;
   bool out_of_idx_mem = max_idx_mem_needed > max_idx_mem_per_thread;
   bool out_of_mem = out_of_dat_mem or out_of_idx_mem;

   if ( out_of_mem ){
         dispatch(false);
   }

   for ( unsigned int L : add_L_after_dispatch ){
      // TODO move to ::add_set_L(L);
       OF[L].insert(  OF[L].end(),  this_set_OF[L].begin(),  this_set_OF[L].end() );
      PMX[L].insert( PMX[L].end(), this_set_PMX[L].begin(), this_set_PMX[L].end() );
      FVH[L].insert( FVH[L].end(), this_set_FVH[L].begin(), this_set_FVH[L].end() );
       KS[L].insert(  KS[L].end(),  this_set_KS[L].begin(),  this_set_KS[L].end() );
         Fm_size[L] +=    this_set_Fm_size[L];
         AC_size[L] +=    this_set_AC_size[L];
       ABCD_size[L] +=  this_set_ABCD_size[L];
      ABCD0_size[L] += this_set_ABCD0_size[L];
      SPHER_size[L] += this_set_SPHER_size[L];
      dat_mem_needed[L] += this_set_dat_mem_needed[L];
      idx_mem_needed[L] += this_set_idx_mem_needed[L];
      this_set_dat_mem_needed[L] = 0;
      this_set_idx_mem_needed[L] = 0;
      offset_V[L] += this_set_offset_V[L];
      offset_F[L] += this_set_offset_F[L];
      offset_G[L] += this_set_offset_G[L];
      offset_Q[L] += this_set_offset_Q[L];

      max_dat_mem_needed = max(max_dat_mem_needed, dat_mem_needed[L]);
      max_idx_mem_needed = max(max_idx_mem_needed, idx_mem_needed[L]);
   }
   // TODO move to reset_set
   for ( unsigned int L : this_set_L ){
       this_set_OF[L].clear();
      this_set_PMX[L].clear();
      this_set_FVH[L].clear();
       this_set_KS[L].clear();
         this_set_Fm_size[L] = 0;
         this_set_AC_size[L] = 0;
       this_set_ABCD_size[L] = 0;
      this_set_ABCD0_size[L] = 0;
      this_set_SPHER_size[L] = 0;
      this_set_offset_V[L] = 0;
      this_set_offset_F[L] = 0;
      this_set_offset_G[L] = 0;
      this_set_offset_Q[L] = 0;

      idx_mem_needed[L] = 0;

   }

   add_L_after_dispatch.clear();
   this_set_L.clear();

}


void libGint::set_cell( bool periodic_, double * cell_h_,  double * cell_i_ ){
//#pragma omp critical
//   {cout << " Setting cell " << periodic_ << " " << cell_h[0] << " " << cell_i_[8] << endl; cout.flush();}
   periodic = periodic_;
   for (int i=0;i<9;i++){ cell_h[CELL_HMAT_OFF+i] = cell_h_[i]; }
   for (int i=0;i<9;i++){ cell_h[CELL_HINV_OFF+i] = cell_i_[i]; }
}

void libGint::set_neighs( double * neighs_, int nneighs_ ){
//#pragma omp critical
//   {cout << " Setting neigh " << nneighs_ << " " << neighs_[0] << " -  " << neighs_[3*nneighs_-1] << endl; cout.flush();}
   max_ncells = nneighs_;
   neighs.resize(3*nneighs_);
   for (int i=0;i<3*nneighs_;i++){ neighs[i] = neighs_[i]; }
}


// TODO ? get zero and set could be async ?
void libGint::set_P( double * P_, int P_size ){
   nspin = 1;
   P_a = P_ ;
   FP_size = P_size;

#pragma omp single copyprivate(P_a_dev)
   {
   // if this is the first call, P_a_dev is zero and no operation is performed.
   CUDA_GPU_ERR_CHECK( cudaFree(P_a_dev));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&P_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( P_a_dev, P_a, sizeof(double)*FP_size, cudaMemcpyHostToDevice ));  
   }

   FP_size_omp = 2 * nspin * sizeof(double)*(FP_size) / Nomp ;
//#pragma omp critical
//   cout << " Setting P from " << P_ << " to " << P_a_dev << " x " << FP_size << endl;

   allocate_on_GPU();
  
}
void libGint::set_P( std::vector<double> & P_ ){ set_P(P_.data(), P_.size()); }

void libGint::set_P( double * P_a_, double * P_b_, int P_size ){
//   cout << " Setting P Polarized with size " << P_a_.size() << " " << P_b_.size() << endl ;
   nspin = 2;
   P_a = P_a_;
   P_b = P_b_;
   FP_size = P_size;

#pragma omp single copyprivate(P_a_dev) // nowait
   {
   // if this is the first call, P_a_dev is zero and no operation is performed.
   CUDA_GPU_ERR_CHECK( cudaFree(P_a_dev));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&P_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( P_a_dev, P_a_, sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
   }
#pragma omp single copyprivate(P_b_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaFree(P_b_dev));     
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&P_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( P_b_dev, P_b_, sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
   }

   FP_size_omp = 2 * nspin * sizeof(double)*(FP_size) / Nomp ;
   allocate_on_GPU();
}
void libGint::set_P( std::vector<double> & P_a_, std::vector<double> & P_b_ ){ set_P( P_a_.data(), P_b_.data(), P_a_.size()); }

void libGint::allocate_on_GPU(){
//   PUSH_RANGE("dispatch malloc",1);

   // TODO move at init
   ftable_ld = 0; // ld of table for fgamma === n+1 === nmax + 6 + 1 === 21 + 6 + 1 === 28
   int nmax = 21;
   double tdelta = 0.1;  // hardcoded in source code
   double tmin = 0.; // hardcoded in source code
   double tmax = 12.;  // hardcoded in source code
   int n = nmax + 6; // 6 is the max taylor expansion. Also hardcoded
   int itabmin = floor(tmin/tdelta); // === 0
   int itabmax = int((tmax - tmin)/tdelta); // === 120
   int nelem = (itabmax - itabmin + 1 ) * (n+1); // === 121*(n+1) == 121*ftable_ld === 121*28 === 3388
   double* ftable = create_md_ftable( nmax, tmin, tmax, tdelta, &ftable_ld);

//   cout << " Starting at " << ua.internal_buffer.size() << endl;
   if ( not all_idx_Kabcd_ready ) {
      for ( unsigned int ia = 0; ia < unique_K_list.size(); ia++ ){
      unsigned int iKa = unique_K_list[ia];
      unsigned int nla = l_from_K_list[ia];
      unsigned int npa = p_from_K_list[ia];
      for ( unsigned int ib = 0; ib < unique_K_list.size(); ib++ ){
      unsigned int iKb = unique_K_list[ib];
      unsigned int nlb = l_from_K_list[ib];
      unsigned int npb = p_from_K_list[ib];
      for ( unsigned int ic = 0; ic < unique_K_list.size(); ic++ ){
      unsigned int iKc = unique_K_list[ic];
      unsigned int nlc = l_from_K_list[ic];
      unsigned int npc = p_from_K_list[ic];
      for ( unsigned int id = 0; id < unique_K_list.size(); id++ ){
      unsigned int iKd = unique_K_list[id];
      unsigned int nld = l_from_K_list[id];
      unsigned int npd = p_from_K_list[id];

         double * Ka = &ua.internal_buffer[iKa];
         double * Kb = &ua.internal_buffer[iKb];
         double * Kc = &ua.internal_buffer[iKc];
         double * Kd = &ua.internal_buffer[iKd];

         int nl = nla*nlb*nlc*nld;
         int np = npa*npb*npc*npd;
         std::vector<double> tmp_K( nl * np );

         int ilabcd = 0;
         for ( unsigned int ila=0; ila < nla; ila++ ){
         for ( unsigned int ilb=0; ilb < nlb; ilb++ ){
         for ( unsigned int ilc=0; ilc < nlc; ilc++ ){
         for ( unsigned int ild=0; ild < nld; ild++ ){
            int ipabcd = 0;
            for ( unsigned int ipa=0; ipa < npa ; ipa++){
            for ( unsigned int ipb=0; ipb < npb ; ipb++){
            for ( unsigned int ipc=0; ipc < npc ; ipc++){
            for ( unsigned int ipd=0; ipd < npd ; ipd++){
               tmp_K[ ilabcd * np + ipabcd ]  = Ka[ ila * npa + ipa ];
               tmp_K[ ilabcd * np + ipabcd ] *= Kb[ ilb * npb + ipb ];
               tmp_K[ ilabcd * np + ipabcd ] *= Kc[ ilc * npc + ipc ];
               tmp_K[ ilabcd * np + ipabcd ] *= Kd[ ild * npd + ipd ];
//               cout << " Precomputing K " << ila<<ilb<<ilc<<ild<<"."<<ipa<<ipb<<ipc<<ipd<< " " << tmp_K[ ilabcd * np + ipabcd ] << endl;
               ipabcd++;
            }}}}
            ilabcd++;
         }}}}
      unsigned int iKabcd = ua.add( tmp_K.data(), nl*np );
//      cout << " Saving precomputed K " << nl << "." << np << " from " << iKa << " " << iKb << " " << iKc << " " << iKd << " @ " << iKabcd << endl;
      all_idx_Kabcd[ four_uint_tuple(iKa, iKb, iKc, iKd) ] = iKabcd;
      }}}}
      all_idx_Kabcd_ready = true;
   }
//   cout << " Ending at " << ua.internal_buffer.size() << endl;

   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&data_dev, sizeof(double)*(ua.internal_buffer.size()) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&dat_mem_dev, max_dat_mem_per_thread ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&idx_mem_dev , max_idx_mem_per_thread ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&cell_h_dev, sizeof(double)*(2*9) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&neighs_dev, sizeof(double)*(3*max_ncells) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&ftable_dev, sizeof(double)*(nelem) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&C2S_dev, sizeof(double)*245 ));
   // TODO do
   #define max_plan_size_possible 10000000
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&plan_dev,sizeof(int)*max_plan_size_possible ));
   CUDA_GPU_ERR_CHECK( cudaHostAlloc( (void**)&idx_mem_stg, max_idx_mem_per_thread, cudaHostAllocPortable ));
   CUDA_GPU_ERR_CHECK( cudaHostAlloc( (void**)&plan_stg, sizeof(int)*max_plan_size_possible, cudaHostAllocPortable ));
   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//   POP_RANGE; // dispatch malloc

//   PUSH_RANGE("dispatch memcpy",1);
   CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
      data_dev, ua.internal_buffer.data(), sizeof(double)*(ua.internal_buffer.size()), cudaMemcpyHostToDevice, cuda_stream ));
   CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
      cell_h_dev, cell_h, sizeof(double)*(2*9), cudaMemcpyHostToDevice, cuda_stream ));
   CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
      neighs_dev, neighs.data(), sizeof(double)*(3*max_ncells), cudaMemcpyHostToDevice, cuda_stream )); 
   CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
      ftable_dev, ftable, sizeof(double)*(nelem), cudaMemcpyHostToDevice, cuda_stream ));
   CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
      C2S_dev, c2s, sizeof(double)*245, cudaMemcpyHostToDevice, cuda_stream ));
//   POP_RANGE; // dispatch memcpy
//   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
   // ! needed after async memcpy TODO move to dispatch
//   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
   CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
}

void libGint::free_on_GPU(){
   // TODO move to some resize / delete function at get_K time
   CUDA_GPU_ERR_CHECK( cudaFree(data_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(dat_mem_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(idx_mem_dev) ); 
   CUDA_GPU_ERR_CHECK( cudaFree(cell_h_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(neighs_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(ftable_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(C2S_dev) );
   CUDA_GPU_ERR_CHECK( cudaFree(plan_dev) );
   CUDA_GPU_ERR_CHECK( cudaFreeHost(idx_mem_stg) );
   CUDA_GPU_ERR_CHECK( cudaFreeHost(plan_stg) );
#pragma omp single
   CUDA_GPU_ERR_CHECK( cudaFree(K_a_dev));
#pragma omp single
   CUDA_GPU_ERR_CHECK( cudaFree(P_a_dev));
   if ( nspin == 2 ){
#pragma omp single
      CUDA_GPU_ERR_CHECK( cudaFree(K_b_dev));
#pragma omp single
      CUDA_GPU_ERR_CHECK( cudaFree(P_b_dev));
   }
}

void libGint::zero_K( int K_size ){
   assert( nspin == 1 );
   FP_size = K_size;
#pragma omp single copyprivate(K_a_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemset( K_a_dev, 0, sizeof(double)*FP_size ));
   }
}

void libGint::zero_K( int K_size, int K_size_ ){
   assert( nspin == 2 );
   assert( K_size == K_size_ );
   FP_size = K_size;
#pragma omp single copyprivate(K_a_dev) // nowait
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemset( K_a_dev, 0, sizeof(double)*FP_size ));
   }
#pragma omp single copyprivate(K_b_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemset( K_b_dev, 0, sizeof(double)*FP_size ));
   }
}

void libGint::set_K( double * K_ , int K_size ){
   assert( nspin == 1 );
   K_a = K_;
   FP_size = K_size;

#pragma omp single copyprivate(K_a_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_a_dev, K_, sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
   }
}

void libGint::set_K( std::vector<double> & K_ ){ set_K( K_.data(), K_.size()); }

void libGint::set_K( double * K_a_, double * K_b_, int K_size ){
   assert( nspin == 2 );
   K_a = K_a_;
   K_b = K_b_;
   FP_size = K_size;
#pragma omp single copyprivate(K_a_dev) // nowait
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_a_dev, K_a_, sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
   }
#pragma omp single copyprivate(K_b_dev)
   {
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&K_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_b_dev, K_b_, sizeof(double)*FP_size, cudaMemcpyHostToDevice ));
   }
}
void libGint::set_K( std::vector<double> & K_a_ , std::vector<double> & K_b_ ){ set_K( K_a_.data(), K_b_.data(), K_a_.size()); }

void libGint::get_K( double * K_ ){
   assert( nspin == 1 );
   // make sure every thread is done with its calculations
//   CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
   dispatch(true);
//   CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
#pragma omp barrier

#pragma omp single
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_, K_a_dev, sizeof(double)*FP_size, cudaMemcpyDeviceToHost ));

   free_on_GPU();

//#pragma omp critical
//   cout << " Getting K from " << K_a_dev << " to " << K_ << " x " << FP_size << endl;

}

void libGint::get_K( std::vector<double> & K_a_ ){ get_K( K_a_.data()); }

void libGint::get_K( double * K_a_,  double * K_b_ ){
   assert( nspin == 2 );
   // make sure every thread is done with its calculations
   dispatch(true);
#pragma omp barrier
 
#pragma omp single nowait
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_a_, K_a_dev, sizeof(double)*FP_size, cudaMemcpyDeviceToHost ));
#pragma omp single
   CUDA_GPU_ERR_CHECK( cudaMemcpy( K_b_, K_b_dev, sizeof(double)*FP_size, cudaMemcpyDeviceToHost ));

   free_on_GPU();

}
void libGint::get_K( std::vector<double> & K_a_,  std::vector<double> & K_b_ ){ get_K( K_a_.data(), K_b_.data()); }

void libGint::reset_indices(unsigned int L){

    OF[L].clear();
   PMX[L].clear();
   FVH[L].clear();
    KS[L].clear();

   offset_F[L] = 0;
   offset_V[L] = 0;
   offset_G[L] = 0;
   offset_Q[L] = 0;

   Fm_size[L] = 0;
   AC_size[L] = 0;
   ABCD_size[L] = 0;
   ABCD0_size[L] = 0;
   SPHER_size[L] = 0;

//   n_set = 0;
//   prm_in_set = 0;
//   n_prm = 0;
//   p0 = 0;
//   cell_in_set = 0;

//   encoded_moments.erase(L);

}

void libGint::dispatch( bool dispatch_all ){

//   dis_timer.start();
//   #pragma omp critical
//   { cout << "Dispatch on stream " << cuda_stream << " @ " << &cuda_stream << " F: " << dispatch_all << endl; cout.flush(); }

//   CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//   CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );


//   #pragma omp critical
//   { cout << "Dispatch: at max dat " << max_dat_mem_needed/1024/1024 << " MiB values at thr " << omp_get_thread_num() << endl; cout.flush(); }
//   { cout << "Dispatch: at max idx " << max_idx_mem_needed << " B values at thr " << omp_get_thread_num() << endl; cout.flush(); }
   
//   OUT.resize(out_size);

   // Main cycle.
   // 0) Decide if we are to run this L
   // 1) Get the plan
   // 2) Copy the input vectors to device memory
   // 3) Run

   // There is a (small) chance we get to this point before the previous dispatch on the same stream
   // has finished. To avoid problems, we wait
   CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );

   // Flag to sync before we overwrite the RAM side idx
   bool first_loop_this_dispatch = true;

//   PUSH_RANGE("dispatch all L",3);
   for ( unsigned int L : encoded_moments ){
 
//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );

  
      // Do we have to compute this batch of integrals:
      // 2 possible reasons: 
      // (1) We have asked to dispatch all, likely because we got a get_K request
      // (2) adding the next set would mean the the next time we try the batch would require
      //       too much memory
      bool must_be_computed = dispatch_all or (add_L_after_dispatch.count(L) > 0);
      bool can_be_skipped = not must_be_computed;

      // Early exit moments with a small number of integrals
      // No worry, they are guaranteed to be computed before get_K returns
      bool is_too_small = ABCD_size[L] < MIN_INT_BATCH_SIZE;
      if ( can_be_skipped and is_too_small ) { continue; }
      if ( SPHER_size[L] == 0 ){ continue; }

      size_t dat_mem_needed_L_1 = max( Fm_size[L], max( ABCD_size[L], SPHER_size[L] ));
      size_t idx_mem_needed_L = OF[L].size() + PMX[L].size() + FVH[L].size() + KS[L].size();
#ifdef LIBGINT_INTERNAL_DEBUG
      size_t dat_mem_needed_L_2 = max( AC_size[L], 2*ABCD0_size[L] );
      size_t dat_mem_needed_L = dat_mem_needed_L_1 + dat_mem_needed_L_2; 
//      cout << L << " " << dat_mem_needed_L << " . " << dat_mem_needed[L] << " . " << max_dat_mem_per_thread << endl;
      assert( dat_mem_needed_L <= max_dat_mem_per_thread && " Uff. Check dat mem failed "); 
      assert( idx_mem_needed_L <= max_idx_mem_per_thread && " Uff. Check idx mem failed "); 
#endif // LIBGINT_INTERNAL_DEBUG

      int la,lb,lc,ld,labcd;
      decodeL(L,&la,&lb,&lc,&ld);
      labcd = la+lb+lc+ld;
      int Nc = compute_Nc(la,lb,lc,ld);
      int Ns = compute_Ns(la,lb,lc,ld);

//      std::string Lname = std::to_string(la) + "_" + std::to_string(lb) + "_" + std::to_string(lc) + "_" + std::to_string(ld);
//      cout << "Running " << Lname << " " << SPHER_size[L] << endl;

      std::vector<int> * plan = NULL ;
      unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
      plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );
      unsigned int Nop = numVC - numV + 1;

      unsigned int Nprm   = offset_V[L];
      unsigned int Ncells = offset_F[L];
      unsigned int Nqrtt  = offset_Q[L];

      double* Fm_dev    = &dat_mem_dev[0];
      double* AC_dev    = &dat_mem_dev[0] + dat_mem_needed_L_1 / sizeof(double);
      double* ABCD_dev  = &dat_mem_dev[0];
      double* ABCD0_dev = &dat_mem_dev[0] + dat_mem_needed_L_1 / sizeof(double);
      double* SPTMP_dev = &dat_mem_dev[0] + dat_mem_needed_L_1 / sizeof(double) + ABCD0_size[L] / sizeof(double);
      double* SPHER_dev = &dat_mem_dev[0]; 

      unsigned int*  OF_dev = &idx_mem_dev[0];
      unsigned int* PMX_dev =  OF_dev +  OF[L].size();
      unsigned int* FVH_dev = PMX_dev + PMX[L].size();
      unsigned int*  KS_dev = FVH_dev + FVH[L].size();

      unsigned int*  OF_stg = &idx_mem_stg[0];
      unsigned int* PMX_stg =  OF_stg +  OF[L].size();
      unsigned int* FVH_stg = PMX_stg + PMX[L].size();
      unsigned int*  KS_stg = FVH_stg + FVH[L].size();


//#pragma omp critical
//      {
//      size_t tot_mem_L = Fm_size[L] + AC_size[L] + ABCD_size[L] + ABCD0_size[L] + SPHER_size[L];
      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | ";
      cout << Nprm << " prms " << Ncells << " cells " << Nqrtt << " qrtts " << max_ncells << " Ng " ;
      cout <<    Fm_size[L]/1024/1024 << " " ;
      cout <<    AC_size[L]/1024/1024 << " " ;
      cout <<  ABCD_size[L]/1024/1024 << " ";
      cout << ABCD0_size[L]/1024/1024 << " ";
      cout << SPHER_size[L]/1024/1024 << " | ";
      cout << dispatch_all << " " << must_be_computed << " " << is_too_small << " " << first_loop_this_dispatch << " | ";
      cout << dat_mem_needed_L_1/1024/1024 << " MB | " << dat_mem_needed_L_2/1024/1024 << " MB | ";
      cout << endl;
//      }

//      PUSH_RANGE(Lname.c_str(),3);

//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );

      // it is (very) possible that we reach this point before the previous loop completed, so we sync
      // before overwriting index arrays on device
      // TODO ? not necessary ?
      if ( not first_loop_this_dispatch ){
         cout << " SYNCING " << endl;
         CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
      } else {
         first_loop_this_dispatch = false;
      }

      // Stage the idx array for async copy on device
      memcpy(  OF_stg,  OF[L].data(), sizeof(unsigned int)*( OF[L].size()) );
      memcpy( PMX_stg, PMX[L].data(), sizeof(unsigned int)*(PMX[L].size()) );
      memcpy( FVH_stg, FVH[L].data(), sizeof(unsigned int)*(FVH[L].size()) );
      memcpy(  KS_stg,  KS[L].data(), sizeof(unsigned int)*( KS[L].size()) );
      memcpy(plan_stg,  plan->data(), sizeof(int)*( plan->size()) );

      CUDA_GPU_ERR_CHECK( cudaMemcpyAsync( 
         idx_mem_dev, idx_mem_stg, sizeof(unsigned int)*idx_mem_needed_L, cudaMemcpyHostToDevice, cuda_stream));
//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );

     
//      PUSH_RANGE("transfer indeces",4);
      CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
         plan_dev, plan_stg, sizeof(int)*(plan->size()), cudaMemcpyHostToDevice, cuda_stream));
//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );


//      CUDA_GPU_ERR_CHECK( cudaMemcpyAsync( 
//          OF_dev,  OF[L].data(), sizeof(unsigned int)*(Nprm), cudaMemcpyHostToDevice, cuda_stream ));  
//      CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
//         PMX_dev, PMX[L].data(), sizeof(unsigned int)*(Nprm), cudaMemcpyHostToDevice, cuda_stream )); 
//      CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
//         FVH_dev, FVH[L].data(), sizeof(unsigned int)*(FVH[L].size()), cudaMemcpyHostToDevice, cuda_stream ));
//      CUDA_GPU_ERR_CHECK( cudaMemcpyAsync(
//          KS_dev,  KS[L].data(), sizeof(unsigned int)*( KS[L].size()), cudaMemcpyHostToDevice, cuda_stream )); 

//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );

      // (nvidia?) GPUs adhere to IEEE-754, so a pattern of all 0s represents a floating-point zero.
      CUDA_GPU_ERR_CHECK( cudaMemsetAsync( dat_mem_dev, 0, dat_mem_needed_L , cuda_stream ) );

//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );     
//      POP_RANGE; // transfer indeces

//      PUSH_RANGE("compute",5);
      int prep_Fm_blocksize = 128;
      int prep_Fm_numblocks = (Nprm+prep_Fm_blocksize-1)/prep_Fm_blocksize;

      CUDA_GPU_ERR_CHECK( cudaMemsetAsync( Fm_dev, 0, Fm_size[L] , cuda_stream ) );
      prepare_Fm_batched_gpu_low_private<<<prep_Fm_numblocks,prep_Fm_blocksize,0,cuda_stream>>>(
         FVH_dev, OF_dev, PMX_dev, data_dev, Fm_dev, Nprm, labcd,
         periodic, cell_h_dev, neighs_dev, max_ncells );

//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

//      std::vector<double> FM0_on_cpu(Fm_size[L]);
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( FM0_on_cpu.data(),  Fm_dev, sizeof(double)*(Fm_size[L]), cudaMemcpyDeviceToHost) );
//      cout << " FM " << endl;
//      for( unsigned int ifm=0; ifm < Fm_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << FM0_on_cpu[ifm] << endl;
//      } cout << endl;

      int Fm_blocksize = 32;
      int Fm_numblocks = Nprm; // (Nprm+Fm_blocksize-1)/Fm_blocksize;

      compute_Fm_batched_gpu_low_private<<<Fm_numblocks,Fm_blocksize,0,cuda_stream>>>(
         Fm_dev, Nprm, labcd, periodic, neighs_dev, 
         ftable_dev, ftable_ld,R_cut,C0_dev,ld_C0,
         x12_to_patch_low_R_dev, x12_to_patch_high_R_dev, BW_by_patch_dev,
         potential_type, max_ncells );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );

      int Vm_blocksize = 64;
      int Vm_numblocks = Nprm;
      compute_Vm_batched_gpu_low_private<<<Vm_numblocks,Vm_blocksize,0,cuda_stream>>>(
         Fm_dev, Nprm, labcd, periodic, neighs_dev, 
         ftable_dev, ftable_ld,R_cut,C0_dev,ld_C0,
         x12_to_patch_low_R_dev, x12_to_patch_high_R_dev, BW_by_patch_dev,
         potential_type, max_ncells );

//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      std::vector<double> FM_on_cpu(Fm_size[L]);
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( FM_on_cpu.data(),  Fm_dev, sizeof(double)*(Fm_size[L]), cudaMemcpyDeviceToHost) );
//      cout << " FM " << endl;
//      for( unsigned int ifm=0; ifm < Fm_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << FM_on_cpu[ifm] << endl;
//      } cout << endl;

//      std::vector<unsigned int> FVH_on_cpu(FVH[L].size());
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( FVH_on_cpu.data(),  FVH_dev, sizeof(unsigned int )*(FVH[L].size()), cudaMemcpyDeviceToHost) );
//      cout << " FVH <- " << la << " " << lb << " " << lc << " " << ld << endl;
//      for( int ifm=0; ifm < FVH[L].size(); ifm++ ){
//         cout << " " << FVH_on_cpu[ifm] ;
//         if (ifm % FVH_SIZE == FVH_SIZE-1 ){ cout << endl ; }
//      } cout << endl;
//
//      // (nvidia?) GPUs adhere to IEEE-754, so a pattern of all 0s represents a floating-point zero.
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

//      cout << " Dev is " << dat_mem_dev << " of size " << max_dat_mem_per_thread/1024/1024 << " AC is " << AC_dev << " of size " << AC_size[L]/1024/1024 << " L: " << L << endl;

      CUDA_GPU_ERR_CHECK( cudaMemsetAsync( AC_dev, 0, AC_size[L] , cuda_stream ) );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

      int vrr_index = 64*la+16*lb+4*lc+ld;
//      compute_VRR_v2_batched_gpu_low<<<Ncells*max_ncells,64,0,cuda_stream>>>(
//        Ncells, vrr_index, PMX_dev, FVH_dev, Fm_dev, data_dev,
//        AC_dev, nullptr, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 

      compute_VRR_v3(
        Ncells, vrr_index, PMX_dev, FVH_dev, Fm_dev, data_dev,
        AC_dev, nullptr, vrr_blocksize, hrr_blocksize, numV, numVC, max_ncells, cuda_stream ); 


//         compute_VRR_batched_gpu_low<<<Ncells*max_ncells,64,0,cuda_stream>>>(
//            Ncells, plan_dev, PMX_dev, FVH_dev, Fm_dev, data_dev,
//            AC_dev, nullptr, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 


//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

//      std::vector<double> AC0_on_cpu(AC_size[L]);
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( AC0_on_cpu.data(),  AC_dev, AC_size[L], cudaMemcpyDeviceToHost) );
//      cout << " AC " << la << " " << lb << " " << lc << " " << ld << " " << AC_size[L] << endl;
//      for( unsigned int ifm=0; ifm < AC_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << AC0_on_cpu[ifm] << endl;
//      } cout << endl;    

      CUDA_GPU_ERR_CHECK( cudaMemsetAsync( ABCD_dev, 0, ABCD_size[L] , cuda_stream ) );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

      compute_SFT_batched_gpu_low<<<Ncells*Nop,128,0,cuda_stream>>>(
         Ncells, plan_dev, PMX_dev, FVH_dev, nullptr, data_dev,
         AC_dev, ABCD_dev, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

      compute_ECO_batched_gpu_low<<<Ncells*Nop,128,0,cuda_stream>>>(
         Ncells, plan_dev, PMX_dev, FVH_dev, nullptr, data_dev,
         AC_dev, ABCD_dev, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

//      std::vector<double> AC_on_cpu(AC_size[L]);
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( AC_on_cpu.data(),  AC_dev, sizeof(double)*(AC_size[L]), cudaMemcpyDeviceToHost) );
//      cout << " AC " << AC_size[L] << endl;
//      for( unsigned int ifm=0; ifm < AC_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << AC_on_cpu[ifm] << endl;
//      } cout << endl;

//      std::vector<double> ABCD_on_cpu(ABCD_size[L]);
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( ABCD_on_cpu.data(),  ABCD_dev, sizeof(double)*(ABCD_size[L]), cudaMemcpyDeviceToHost) );
//      cout << " ABCD " << ABCD_size[L] << endl;
//      for( unsigned int ifm=0; ifm < ABCD_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << ABCD_on_cpu[ifm] << endl;
//      } cout << endl;

//      std::vector<unsigned int> FVH2_on_cpu(FVH[L].size());
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( FVH2_on_cpu.data(),  FVH_dev, sizeof(unsigned int )*(FVH[L].size()), cudaMemcpyDeviceToHost) );
//      cout << " FVH 2 <- " << endl;
//      for( int ifm=0; ifm < FVH[L].size(); ifm++ ){
//         cout << " " << FVH2_on_cpu[ifm] ;
//         if (ifm % FVH_SIZE == FVH_SIZE-1 ){ cout << endl ; }
//      } cout << endl;

//      cout << "FVH" << endl;
//      for ( int ii = 0 ; ii < FVH[L].size() ; ii++ ){
//         cout << FVH[L][ii] << " " ;
//         if ( ii % FVH_SIZE == FVH_SIZE-1 ){ cout << endl ; }
//      } cout << endl;

      CUDA_GPU_ERR_CHECK( cudaMemsetAsync( ABCD0_dev, 0, ABCD0_size[L] , cuda_stream ) );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

      compute_HRR_batched_gpu_low<<<Ncells,128,0,cuda_stream>>>(
         Ncells, plan_dev, FVH_dev, data_dev, ABCD_dev, ABCD0_dev,
         periodic, cell_h_dev, neighs_dev,
         hrr_blocksize, Nc, numVC, numVCH );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

//      std::vector<double> ABCD0_on_cpu(ABCD0_size[L]);
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( ABCD0_on_cpu.data(),  ABCD0_dev, sizeof(double)*(ABCD0_size[L]), cudaMemcpyDeviceToHost) );
//      cout << " ABCD0 " << endl;
//      for( int ifm=0; ifm < ABCD0_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << ABCD0_on_cpu[ifm] << endl;
//      } cout << endl;


      // Note: we need to DeviceSynchronize before going from kernels to cublas. TODO actually check it is true
      // TODO it should not be necessary since this cublas handle has been assigned to this stream
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
      // note: uses ABCD as a scratch space

//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
      // (nvidia?) GPUs adhere to IEEE-754, so a pattern of all 0s represents a floating-point zero.
//      CUDA_GPU_ERR_CHECK( cudaMemsetAsync( SPHER_dev, 0, SPHER_size[L]*sizeof(double) , cuda_stream ) );

      compute_SPH_batched_gpu_alt ( Nqrtt, la, lb, lc, ld, ABCD0_dev, SPHER_dev, SPTMP_dev, C2S_dev, cublas_handle );

//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      std::vector<double> SPHER_on_cpu(SPHER_size[L]);
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( SPHER_on_cpu.data(),  SPHER_dev, sizeof(double)*(SPHER_size[L]), cudaMemcpyDeviceToHost) );
//      cout << " SPHER " << endl;
//      for( int ifm=0; ifm < SPHER_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << SPHER_on_cpu[ifm] << endl;
//      } cout << endl;

//      #pragma omp critical
//      {
//         cout << " Preparing to compute KS " << Nqrtt << " " << KS_dev << " [" << KS[L].size() << "] "
//              << P_a_dev << " [" << FP_size      << " ]" << SPHER_dev << " [" << SPHER_size[L] << "]"
//              << K_a_dev << " " << hf_fac << " on stream " << cuda_stream << endl;
//
//         for (int q=0; q < KS[L].size() ; q++ ){
//            cout << KS[L][q] << "  ";
//            if ( q%KS_SIZE==KS_SIZE-1) { cout << endl; }
//         }
//      }
      compute_KS_gpu<<<Nqrtt,128,0,cuda_stream>>>( Nqrtt, KS_dev, la,lb,lc,ld, P_a_dev, SPHER_dev, K_a_dev, data_dev, hf_fac );
      if ( nspin == 2 ){
         compute_KS_gpu<<<Nqrtt,128,0,cuda_stream>>>( Nqrtt, KS_dev, la,lb,lc,ld, P_b_dev, SPHER_dev, K_b_dev, data_dev, hf_fac );
      }

//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

//      cout << endl;

//      CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );

//      POP_RANGE; // compute
//      POP_RANGE; // Lname
      reset_indices(L);
   }
//   POP_RANGE; // compute all L

// #pragma omp barrier

//   std::vector<double> OUT_from_gpu( OUT.size() );
//   CUDA_GPU_ERR_CHECK( cudaMemcpy( OUT_from_gpu.data(), OUT_dev, sizeof(double)*(OUT.size()), cudaMemcpyDeviceToHost ));

//   std::vector<double> F_a_from_gpu( FP_size );
//   cout << endl;
//   cout << " Copying K back from gpu to cpu. " << endl;
//   cout << " Gpu address is " << K_a_dev << endl;
//   cout << " Cpu address is " << F_a_from_gpu.data() << endl;
//   cout << " Copying " << FP_size << " elements " << endl;
//   cout << endl;
//   CUDA_GPU_ERR_CHECK( cudaMemcpy( F_a_from_gpu.data(), K_a_dev, sizeof(double)*(FP_size), cudaMemcpyDeviceToHost ));
//   if ( nspin == 2 ){
//      std::vector<double> F_b_from_gpu( FP_size );
//      CUDA_GPU_ERR_CHECK( cudaMemcpy( F_b_from_gpu.data(), K_b_dev, sizeof(double)*(FP_size), cudaMemcpyDeviceToHost ));
//   }
//   for ( int ipf=0; ipf < FP_size; ipf++ ){ cout <<  ipf << " " << F_a_from_gpu[ipf] << endl ; } cout << endl;

   // Wait for all kernels to finish before returning control to caller
//   CUDA_GPU_ERR_CHECK( cudaStreamSynchronize(cuda_stream) );
//   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );


//   dis_timer.stop();
//   dis_ms += dis_timer.elapsedMilliseconds(); 
//   dis_cnt++;
}


template < class T > 
T sum( std::vector< T > x ){
   T ret = (T)0;
   for( auto i : x ){
      ret += i;
   }
   return ret;
}

void libGint::report_througput(bool skip_cpu){
#pragma omp single
   cout << " report_througput not implemented  ";
}

