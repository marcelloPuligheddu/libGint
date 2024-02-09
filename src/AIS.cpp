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

void AIS::add_prm( const int ipa, const int ipb, const int ipc, const int ipd, const int n1, const int n2, const int n3 ){
//   cout << "Adding prm " << prm_in_set << " " << prm << " " << ipd << endl ;
   unsigned int piabcdxyz = encode_ipabcd_n123(ipa,ipb,ipc,ipd,n1,n2,n3);

   unsigned int prm_tmp_list_idxs[PRM_TMP_SIZE] = {0};
   prm_tmp_list_idxs[PRM_TMP_OFFSET_IPZN] = piabcdxyz;
   prm_tmp_list.insert(prm_tmp_list.end(), prm_tmp_list_idxs, prm_tmp_list_idxs+PRM_TMP_SIZE);

   prm_in_set++;
   prm++;
}

void AIS::add_shell (
      int la, int lb, int lc, int ld,
      double* Ka, double* Kb, double* Kc, double* Kd,
      double* A, double* B, double* C, double* D,
      double* Za, double* Zb, double* Zc, double* Zd,
      int npa, int npb, int npc, int npd, int nla, int nlb, int nlc, int nld )
{

   unsigned int n_prm = prm_tmp_list.size() / PRM_TMP_SIZE ;
   
   if (n_prm == 0){
      return;
   }

   std::vector<int> * plan = NULL ;
   unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
   plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );

   unsigned int idx_Ka = ua.add( Ka, npa*nla );
   unsigned int idx_Kb = ua.add( Kb, npb*nlb );
   unsigned int idx_Kc = ua.add( Kc, npc*nlc );
   unsigned int idx_Kd = ua.add( Kd, npd*nld );

   unsigned int idx_A = ua.add( A, 3 );
   unsigned int idx_B = ua.add( B, 3 );
   unsigned int idx_C = ua.add( C, 3 );
   unsigned int idx_D = ua.add( D, 3 );

   unsigned int idx_Za = ua.add( Za, npa*nla );
   unsigned int idx_Zb = ua.add( Zb, npb*nlb );
   unsigned int idx_Zc = ua.add( Zc, npc*nlc );
   unsigned int idx_Zd = ua.add( Zd, npd*nld );

   unsigned int N_cc = nla*nlb*nlc*nld;
   unsigned int L = encodeL(la,lb,lc,ld);

   unsigned int Of = offset_F[L];
   unsigned int Ov = offset_V[L];
   unsigned int Og = offset_G[L];
   unsigned int Oq = offset_Q[L];

   unsigned int encoded_nlabcd = encode4(nla,nlb,nlc,nld);
   unsigned int encoded_npabcd = encode4(npa,npb,npc,npd);


   for( unsigned int pi = 0; pi < n_prm; pi++ ){
      unsigned int pm_idxs[PMI_SIZE] = {0};
      unsigned int ipabcd_n123 = prm_tmp_list[pi*PRM_TMP_SIZE+PRM_TMP_OFFSET_IPZN];
      pm_idxs[PMI_OFFSET_OF] = Of;
      pm_idxs[PMI_OFFSET_IPZN] = ipabcd_n123;
      PMI[L].insert(PMI[L].end(), pm_idxs, pm_idxs+PMI_SIZE);
   }

   unsigned int fvh_idxs[FVH_SIZE] = {0};
   fvh_idxs[FVH_OFFSET_OV    ] = Ov;
   fvh_idxs[FVH_OFFSET_OG    ] = Og;
   fvh_idxs[FVH_OFFSET_OQ    ] = Oq;
   fvh_idxs[FVH_OFFSET_NPRM  ] = n_prm;
   fvh_idxs[FVH_OFFSET_IDX_A ] = idx_A;
   fvh_idxs[FVH_OFFSET_IDX_B ] = idx_B;
   fvh_idxs[FVH_OFFSET_IDX_C ] = idx_C;
   fvh_idxs[FVH_OFFSET_IDX_D ] = idx_D;
   fvh_idxs[FVH_OFFSET_IDX_ZA] = idx_Za;
   fvh_idxs[FVH_OFFSET_IDX_ZB] = idx_Zb;
   fvh_idxs[FVH_OFFSET_IDX_ZC] = idx_Zc;
   fvh_idxs[FVH_OFFSET_IDX_ZD] = idx_Zd;
   fvh_idxs[FVH_OFFSET_IDX_KA] = idx_Ka;
   fvh_idxs[FVH_OFFSET_IDX_KB] = idx_Kb;
   fvh_idxs[FVH_OFFSET_IDX_KC] = idx_Kc;
   fvh_idxs[FVH_OFFSET_IDX_KD] = idx_Kd;
   fvh_idxs[FVH_OFFSET_NLABCD] = encoded_nlabcd;
   fvh_idxs[FVH_OFFSET_NPABCD] = encoded_npabcd;

   FVH[L].insert(FVH[L].end(), fvh_idxs, fvh_idxs+FVH_SIZE);
   
   int labcd = la+lb+lc+ld;
   Fm_size[L] += (1+labcd) * n_prm;
   if ( labcd > 0 ){
      Fm_size[L] += (4*3+5) * n_prm;
   }
   AC_size[L] += vrr_blocksize * n_prm;
   ABCD_size[L] += hrr_blocksize * N_cc;

   offset_G[L] += N_cc;
   offset_V[L] += n_prm;
   offset_F[L] ++ ;

   encoded_moments.insert(L);

}

void AIS::add_cell() {

   unsigned int n_prm = prm_tmp_list.size() / PRM_TMP_SIZE ;

   if (n_prm == 0){
      return;
   }

   cell_in_set++;
   prm_tmp_list.clear();
}

int AIS::add_qrt( int la, int lb, int lc, int ld, int nla, int nlb, int nlc, int nld ){
   unsigned int nlabcd = nla*nlb*nlc*nld;

   if (prm_in_set > 0 ){
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
   return -1;
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


void AIS::dispatch(){

   Timer timer;
   Timer timer2;
   timer2.start();

   out_size = 0;
   for( unsigned int L : encoded_moments ){
      out_size += OUT_size[L];
   }
   cout << "Dispatch: Will compute " << out_size << " values " << endl;
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


   unsigned int max_integral_scratch_size = 0;
   unsigned int max_plan_size = 0;
   unsigned int max_PMI_size = 0;
   unsigned int max_FVH_size = 0;
   unsigned int max_SPH_size = 0;
   unsigned int max_TRA_size = 0;
   for ( unsigned int L : encoded_moments ){

      int la,lb,lc,ld;
      decodeL(L,&la,&lb,&lc,&ld);
      std::vector<int> * plan = NULL ;
      unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
      plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );

      int integral_scratch_size = Fm_size[L] + AC_size[L] + ABCD_size[L] + ABCD0_size[L] + SPHER_size[L] ;

      max_integral_scratch_size = max( max_integral_scratch_size, integral_scratch_size );

      max_plan_size  = max(max_plan_size, (unsigned int) plan->size());
      max_PMI_size   = max(max_PMI_size,  (unsigned int) PMI[L].size());
      max_FVH_size   = max(max_FVH_size,  (unsigned int) FVH[L].size());
      max_SPH_size   = max(max_SPH_size,  (unsigned int) SPH[L].size());
      max_TRA_size   = max(max_TRA_size,  (unsigned int) TRA[L].size());
   }
   

   cout << "Memory use: " << endl;
   cout << " OUT   " << OUT.size()*sizeof(double) *1.e-6 << "MB " << endl;
   cout << " DAT   " << ua.internal_buffer.size()*sizeof(double) *1.e-6 << "MB " << endl;

   cout << " Scratch " << sizeof(double)*max_integral_scratch_size *1.e-6 << "MB " << endl;
   cout << " PLAN  " << sizeof(int)*max_plan_size          *1.e-6 << "MB "<< endl;

   cout << " PMI   " << sizeof(unsigned int)*max_PMI_size  *1.e-6 << "MB " << endl; 
   cout << " FVH   " << sizeof(unsigned int)*max_FVH_size  *1.e-6 << "MB "<< endl;
   cout << " SPH   " << sizeof(unsigned int)*max_SPH_size  *1.e-6 << "MB "<< endl;
   cout << " TRA   " << sizeof(unsigned int)*max_TRA_size  *1.e-6 << "MB "<< endl;
   cout << " AUX   " << sizeof(unsigned int)*(9+nelem+245) *1.e-6 << "MB "<< endl;


   std::vector<double> integral_scratch(max_integral_scratch_size);

   for ( unsigned int L : encoded_moments ){


      cout << " L " << L << " | ";

      int la,lb,lc,ld,labcd;
      decodeL(L,&la,&lb,&lc,&ld);
      cout.flush();
      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | ";
      cout.flush();

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
      int * plan_L = plan->data();
      double * env = ua.internal_buffer.data();

      cout << "Computing " << Nprm << " prms " << Ncells << " cells" << Nqrtt << " qrtts " << Nshell << " shells ";
      cout << " AC: " << AC_size[L] << " ABCD " << ABCD_size[L] << "/" << ABCD0_size[L] ;

      timer.start();

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

      compute_TRA_batched_low( Nshell, la, lb, lc, ld, TRA_L, SPHER, OUT.data() );

      timer.stop();

      cout << " CPU KRNL " <<  " SL " << labcd << " " ;
      cout << timer.elapsedMicroseconds() << " us " ;
      cout << OUT_size[L] / timer.elapsedMicroseconds() * sizeof(double) / 1.e3 << " GB/s" ;
      cout << endl;

   }

   timer2.stop();
   cout << "DISPATCH CPU " << timer2.elapsedMilliseconds() << endl;


   timer2.start();

   double *data_dev, *cell_h_dev, *ftable_dev, *OUT_dev, *C2S_dev;

   timer.start();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&OUT_dev, sizeof(double)*OUT.size() )); // TODO mvoe alloc to be concurr to compute_Fm
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&data_dev, sizeof(double)*(ua.internal_buffer.size()) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&cell_h_dev, sizeof(double)*(9) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&ftable_dev, sizeof(double)*(nelem) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&C2S_dev, sizeof(double)*245 ));
   timer.stop();
   cout << "I Malloc " << timer.elapsedMilliseconds() << " ms " << endl;

   cout << "Copying Ua " << sizeof(unsigned int)*(ua.internal_buffer.size()) * 1.e-6 << " MBytes " << endl;
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
   cout << "F COPY " << timer.elapsedMilliseconds() << " ms " << endl;


   double *integral_scratch_dev;
   unsigned int *PMI_dev, *FVH_dev, *SPH_dev, *TRA_dev;
   int *plan_dev;


   timer.start();
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&integral_scratch_dev,    sizeof(double)*max_integral_scratch_size ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&plan_dev,  sizeof(int)*max_plan_size ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&PMI_dev, sizeof(unsigned int)*max_PMI_size )); 
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&FVH_dev, sizeof(unsigned int)*max_FVH_size ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&SPH_dev, sizeof(unsigned int)*max_SPH_size )); 
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&TRA_dev, sizeof(unsigned int)*max_TRA_size )); 
   timer.stop();
   cout << "L Malloc " << timer.elapsedMilliseconds() << " ms " << endl;

   timer.start();
   cublasHandle_t cublas_handle;
   CUBLAS_GPU_ERR_CHECK( cublasCreate(&cublas_handle) );
   timer.stop();
   cout << "CUBLAS HANDLE CREATE " << timer.elapsedMilliseconds() << " ms " << endl;
  
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

      cout << "Computing " << Ncells << " cells" ;
      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | " << OUT_size[L] << " | " ;
      cout << " AC: " << AC_size[L] << " ABCD " << ABCD_size[L] << "/" << ABCD0_size[L] ;

      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         plan_dev, plan->data(), sizeof(int)*(plan->size()), cudaMemcpyHostToDevice ));
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         PMI_dev, PMI[L].data(), sizeof(unsigned int)*(PMI[L].size()), cudaMemcpyHostToDevice )); 
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         FVH_dev, FVH[L].data(), sizeof(unsigned int)*(FVH[L].size()), cudaMemcpyHostToDevice ));
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         TRA_dev, TRA[L].data(), sizeof(unsigned int)*(TRA[L].size()), cudaMemcpyHostToDevice ));

      timer.start();

      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
      int Fm_blocksize = 256;
      int Fm_numblocks = (prm+Fm_blocksize-1)/Fm_blocksize;
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

      compute_TRA_batched_gpu_low<<<Nshell,128>>>( Nshell, la, lb, lc, ld, TRA_dev, SPHER_dev, OUT_dev );

      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
      timer.stop();
      cout << " GPU KRNL " <<  " SL " << labcd << " " ;
      cout << timer.elapsedMicroseconds() << " us " ;
      cout << OUT_size[L] / timer.elapsedMicroseconds() * sizeof(double) / 1.e3 << " GB/s" ;
      cout << endl;
   }
   timer.start();
   std::vector<double> OUT_from_gpu( OUT.size() );
   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      OUT_from_gpu.data(), OUT_dev, sizeof(double)*(OUT.size()), cudaMemcpyDeviceToHost ));
   timer.stop();
   cout << "IJKL COPY " << sizeof(double)*(OUT.size()) * 1.e-6 << " MB " << timer.elapsedMilliseconds() << " ms " << endl;


   timer2.stop();
   cout << "DISPATCH GPU " << timer2.elapsedMilliseconds() << " ms" << endl;


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
         cout << "CPU - GPU: Error at " << i << " " << ref << " " << val << " " << diff << " " << ratio << " " << endl ;
         if ( nerrors >= 100 ){
            cout << " TOO MANY ERRORS ! EXITING NOW " << endl;
            exit( EXIT_FAILURE );
         }
      }

   }
   cout << "E[ CPU-GPU ] = " <<  diff_sum / Nval << endl;
   cout << "E[|CPU-GPU|] = " << adiff_sum / Nval << endl;


   CUBLAS_GPU_ERR_CHECK( cublasDestroy(cublas_handle) );

}



