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

using std::max;

void AIS::show_state(){
   cout << "Fm_offset " << Fm_offset << endl;

   for (unsigned int L : encoded_moments ){
      int la,lb,lc,ld;
      decodeL(L,&la,&lb,&lc,&ld);
      cout << " Moments : " << la << " " << lb << " " << lc << " " << ld << endl;
   }

}

void AIS::add_prm( int ipa, int ipb, int ipc, int ipd,  int Lmax, int n1, int n2, int n3 ){
   unsigned int piabcdxyz = encode_ipabcd_n123(ipa,ipb,ipc,ipd,n1,n2,n3);
   unsigned int vrr_tmp_list_idxs[VRR_SIZE] = {0};
   vrr_tmp_list_idxs[VRR_OFFSET_OF] = Fm_offset;
   vrr_tmp_list_idxs[VRR_OFFSET_LMAX] = Lmax;
   vrr_tmp_list_idxs[VRR_OFFSET_IPZN] = piabcdxyz;
   vrr_tmp_list.insert(vrr_tmp_list.end(), vrr_tmp_list_idxs, vrr_tmp_list_idxs+VRR_SIZE);

//   cout << "Adding prm " << Fm_offset << " " << Lmax << " " << piabcdxyz << endl ;

   prm_in_set++;
   prm++;
   Fm_offset += Lmax+1;
   if (Lmax > 0){ Fm_offset += 4*3+5; }
}

void AIS::add_shell (
      double* AB, double* CD, int la, int lb, int lc, int ld,
      double* Ka, double* Kb, double* Kc, double* Kd,
      int npa, int npb, int npc, int npd, int nla, int nlb, int nlc, int nld )
{
   unsigned int n_prm = vrr_tmp_list.size() / VRR_SIZE ;
   
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
   unsigned int idx_AB = ua.add( AB, 3 );
   unsigned int idx_CD = ua.add( CD, 3 );

   unsigned int Op = p0 ;
   unsigned int nlabcd = nla*nlb*nlc*nld;
   unsigned int L = encodeL(la,lb,lc,ld);
   unsigned int Og = offset_G[L];
   unsigned int Oq = offset_Q[L];
   unsigned int Of_s =  vrr_tmp_list[0*VRR_SIZE+VRR_OFFSET_OF];
   unsigned int Lmax =  vrr_tmp_list[0*VRR_SIZE+VRR_OFFSET_LMAX];
   unsigned int Ov = offset_V[L];

   unsigned int plabcd = encode4(nla,nlb,nlc,nld);
   unsigned int ppabcd = encode4(npa,npb,npc,npd);

   unsigned int hrr_idxs[HRR_SIZE] = {0};
   hrr_idxs[HRR_OFFSET_OV    ] = Ov ;
   hrr_idxs[HRR_OFFSET_NPRM  ] = n_prm;
   hrr_idxs[HRR_OFFSET_OG    ] = Og;
   hrr_idxs[HRR_OFFSET_OQ    ] = Oq;
   hrr_idxs[HRR_OFFSET_IDX_AB] = idx_AB;
   hrr_idxs[HRR_OFFSET_IDX_CD] = idx_CD;
   hrr_idxs[HRR_OFFSET_NLABCD] = nlabcd;
   hrr_idxs[HRR_OFFSET_PLABCD] = plabcd;
   hrr_idxs[HRR_OFFSET_PPABCD] = ppabcd;
   hrr_idxs[HRR_OFFSET_IDX_KA] = idx_Ka;
   hrr_idxs[HRR_OFFSET_IDX_KB] = idx_Kb;
   hrr_idxs[HRR_OFFSET_IDX_KC] = idx_Kc;
   hrr_idxs[HRR_OFFSET_IDX_KD] = idx_Kd;
   hrr_idxs[HRR_OFFSET_LMAX  ] = Lmax;
   hrr_idxs[HRR_OFFSET_OF0   ] = Of_s;
   hrr_idxs[HRR_OFFSET_OP    ] = Op;
   HRR[L].insert(HRR[L].end(), hrr_idxs, hrr_idxs+HRR_SIZE);

   AC_size[L] += vrr_blocksize * n_prm;
   ABCD_size[L] += hrr_blocksize * nlabcd;
   offset_G[L] += nlabcd;
   offset_V[L] += n_prm;
   NG[L] += 1;

   encoded_moments.insert(L);

}

void AIS::add_cell(
      double* A, double* B, double* C, double* D, 
      double* Za, double* Zb, double* Zc, double* Zd,
      int npa, int npb, int npc, int npd )
{
   unsigned int n_prm = vrr_tmp_list.size() / VRR_SIZE ;

   if (n_prm == 0){
      return;
   }

   unsigned int idx_A = ua.add( A, 3 );
   unsigned int idx_B = ua.add( B, 3 );
   unsigned int idx_C = ua.add( C, 3 );
   unsigned int idx_D = ua.add( D, 3 );
   unsigned int idx_Za = ua.add( Za, npa );
   unsigned int idx_Zb = ua.add( Zb, npb );
   unsigned int idx_Zc = ua.add( Zc, npc );
   unsigned int idx_Zd = ua.add( Zd, npd );

   unsigned int Of_s = vrr_tmp_list[0*VRR_SIZE+VRR_OFFSET_OF];
   unsigned int Lmax =  vrr_tmp_list[0*VRR_SIZE+VRR_OFFSET_LMAX];

   unsigned int fm_idxs[FM_SIZE] = {0};
   fm_idxs[FMA_OFFSET_LMAX  ] = Lmax;
   fm_idxs[FMA_OFFSET_IDX_A ] = idx_A;
   fm_idxs[FMA_OFFSET_IDX_B ] = idx_B;
   fm_idxs[FMA_OFFSET_IDX_C ] = idx_C;
   fm_idxs[FMA_OFFSET_IDX_D ] = idx_D;
   fm_idxs[FMA_OFFSET_IDX_ZA] = idx_Za;
   fm_idxs[FMA_OFFSET_IDX_ZB] = idx_Zb;
   fm_idxs[FMA_OFFSET_IDX_ZC] = idx_Zc;
   fm_idxs[FMA_OFFSET_IDX_ZD] = idx_Zd;
   fm_idxs[FMA_OFFSET_OF0   ] = Of_s;
   fm_idxs[FMA_OFFSET_P0    ] = p0;

   Fm_input_list.insert(Fm_input_list.end(), fm_idxs, fm_idxs+FM_SIZE);

   for( unsigned int pi = 0; pi < n_prm; pi++ ){
      unsigned int pm_idxs[PM_SIZE] = {0};
      unsigned int ipabcd_n123 = vrr_tmp_list[pi*VRR_SIZE+VRR_OFFSET_IPZN];
      pm_idxs[PMA_OFFSET_IF] = cell_in_set;
      pm_idxs[PMA_OFFSET_IPZN] = ipabcd_n123;
      Pm_input_list.insert(Pm_input_list.end(), pm_idxs, pm_idxs+PM_SIZE);
   }

   cell_in_set++;
   p0 += n_prm;
   vrr_tmp_list.clear();
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

   out_size = 0;
   for( unsigned int L : encoded_moments ){
      out_size += OUT_size[L];
   }
   cout << "Dispatch: Will compute " << out_size << " values " << endl;
   OUT.resize(out_size);

   cout << "Dispatch: Computing " << prm << " Foundamental integrals " << endl;
   Fm.resize( Fm_offset );

   timer.start();
   timer2.start();
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

   compute_Fm_batched( Fm_input_list, Pm_input_list, ua.internal_buffer, Fm, prm, periodic, cell_h, ftable, ftable_ld );
   timer.stop();
   cout << "Fm CPU " << timer.elapsedMilliseconds() << endl;

   unsigned int max_AC_size = 0;
   unsigned int max_ABCD_size = 0;
   unsigned int max_ABCD0_size = 0;
   unsigned int max_SPHER_size = 0;
   unsigned int max_plan_size = 0;
   unsigned int max_HRR_size = 0;
   unsigned int max_SPH_size = 0;
   unsigned int max_TRA_size = 0;
   for ( unsigned int L : encoded_moments ){
      max_AC_size    = max(max_AC_size,    AC_size[L]);
      max_ABCD_size  = max(max_ABCD_size,  ABCD_size[L]);
      max_ABCD0_size = max(max_ABCD0_size, ABCD0_size[L]);
      max_SPHER_size  = max(max_SPHER_size, SPHER_size[L]);

      int la,lb,lc,ld;
      decodeL(L,&la,&lb,&lc,&ld);
      std::vector<int> * plan = NULL ;
      unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
      plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );

      max_plan_size = max(max_plan_size, (unsigned int) plan->size());
      max_HRR_size  = max(max_HRR_size,  (unsigned int) HRR[L].size());
      max_SPH_size  = max(max_SPH_size,  (unsigned int) SPH[L].size());
      max_TRA_size  = max(max_TRA_size,  (unsigned int) TRA[L].size());
   }

   std::vector<double> AC(max_AC_size);
   std::vector<double> ABCD(max_ABCD_size);
   std::vector<double> ABCD0(max_ABCD0_size);
   std::vector<double> SPHER(max_SPHER_size);

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

      unsigned int Nqrtt = offset_Q[L];
      unsigned int Ncells = NG[L];
      unsigned int Nshell = TRA[L].size() / TRA_SIZE; // TODO capitulate and save Nshell[L]

      cout << "Computing " << Ncells << " cells" ;
      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | " << corr << " | " ;
      cout << " AC: " << AC_size[L] << " ABCD " << ABCD_size[L] << "/" << ABCD0_size[L] ;


      timer.start();
      compute_VRR_batched(
         Ncells, *plan, Pm_input_list, HRR[L], Fm, ua.internal_buffer,
         AC, ABCD, vrr_blocksize, hrr_blocksize, labcd, numV, numVC );

      compute_HRR_batched(
         Ncells, *plan, HRR[L], ua.internal_buffer, ABCD, ABCD0,
         hrr_blocksize, Nc, numVC, numVCH );

      compute_SPH_batched( Nqrtt, la, lb, lc, ld, ABCD0, SPHER, ABCD );

      for( unsigned int i=0; i < Nqrtt*Ns; i++ ){
         SPHER[i] *= corr;
      }

      compute_TRA_batched( Nshell, la, lb, lc, ld, TRA[L], SPHER, OUT );

      timer.stop();
      cout << " CPU KRNL :" << timer.elapsedMilliseconds() << " ms" << endl;
   }

   timer2.stop();
   cout << "DISPATCH CPU " << timer2.elapsedMilliseconds() << endl;

   timer.start();   
   timer2.start();
   unsigned int *Fm_input_list_dev, *Pm_input_list_dev;
   double *data_dev, *cell_h_dev, *ftable_dev, *OUT_dev;
   double * Fm_dev;

   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&OUT_dev, sizeof(double)*OUT.size() )); // TODO mvoe alloc to be concurr to compute_Fm
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&Fm_input_list_dev, sizeof(unsigned int)*(Fm_input_list.size()) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&Pm_input_list_dev, sizeof(unsigned int)*(Pm_input_list.size()) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&data_dev, sizeof(double)*(ua.internal_buffer.size()) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&Fm_dev, sizeof(double)*(Fm.size()) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&cell_h_dev, sizeof(double)*(9) ));
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&ftable_dev, sizeof(double)*(nelem) ));

   cout << "Copying If " << sizeof(unsigned int)*(Fm_input_list.size()) * 1.e-6 << " MBytes " << endl;
   CUDA_GPU_ERR_CHECK( cudaMemcpy( 
      Fm_input_list_dev, Fm_input_list.data(), sizeof(unsigned int)*(Fm_input_list.size()), cudaMemcpyHostToDevice ));

   cout << "Copying Ip " << sizeof(unsigned int)*(Pm_input_list.size()) * 1.e-6 << " MBytes " << endl;
   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      Pm_input_list_dev, Pm_input_list.data(), sizeof(unsigned int)*(Pm_input_list.size()), cudaMemcpyHostToDevice ));

   cout << "Copying Ua " << sizeof(unsigned int)*(ua.internal_buffer.size()) * 1.e-6 << " MBytes " << endl;
   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      data_dev, ua.internal_buffer.data(), sizeof(double)*(ua.internal_buffer.size()), cudaMemcpyHostToDevice ));

   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      cell_h_dev, cell_h, sizeof(double)*(9), cudaMemcpyHostToDevice ));

   CUDA_GPU_ERR_CHECK( cudaMemcpy(
      ftable_dev, ftable, sizeof(double)*(nelem), cudaMemcpyHostToDevice ));
   timer.stop();
   cout << "F COPY " << timer.elapsedMilliseconds() << " ms " << endl;


   timer.start();   
   int Fm_blocksize = 256;
   int Fm_numblocks = (prm+Fm_blocksize-1)/Fm_blocksize;
   compute_Fm_batched_low_gpu<<<Fm_numblocks,Fm_blocksize>>>(
      Fm_input_list_dev,
      Pm_input_list_dev,
      data_dev,
      Fm_dev, 
      prm, periodic,
      cell_h_dev, 
      ftable_dev,
      ftable_ld );
   CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
   CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );

   timer.stop();
   cout << "Fm GPU " << timer.elapsedMilliseconds() << " ms " << endl;

   double *AC_dev, *ABCD_dev, *ABCD0_dev, *SPHER_dev;
   unsigned int *HRR_dev, *SPH_dev, *TRA_dev;
   int *plan_dev;
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&AC_dev,    sizeof(double)*AC.size()    )); // TODO move alloc to be cor Fm
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&ABCD_dev,  sizeof(double)*ABCD.size()  )); // TODO move alloc to be
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&ABCD0_dev, sizeof(double)*ABCD0.size() )); // TODO move alloc to be
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&SPHER_dev, sizeof(double)*SPHER.size() )); // TODO move alloc to be
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&plan_dev, sizeof(int)*max_plan_size )); // TODO move alloc to
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&HRR_dev, sizeof(unsigned int)*max_HRR_size )); // TODO move alloc to
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&SPH_dev, sizeof(unsigned int)*max_SPH_size )); // TODO move alloc to
   CUDA_GPU_ERR_CHECK( cudaMalloc( (void**)&TRA_dev, sizeof(unsigned int)*max_TRA_size )); // TODO move alloc to


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

      unsigned int Nqrtt = offset_Q[L];
      unsigned int Ncells = NG[L];
      unsigned int Nshell = TRA[L].size() / TRA_SIZE; // TODO capitulate and save Nshell[L]

      cout << "Computing " << Ncells << " cells" ;
      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | " << corr << " | " ;
      cout << " AC: " << AC_size[L] << " ABCD " << ABCD_size[L] << "/" << ABCD0_size[L] ;

      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         plan_dev, plan->data(), sizeof(int)*(plan->size()), cudaMemcpyHostToDevice ));
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         HRR_dev, HRR[L].data(), sizeof(unsigned int)*(HRR[L].size()), cudaMemcpyHostToDevice ));
      CUDA_GPU_ERR_CHECK( cudaMemcpy(
         TRA_dev, TRA[L].data(), sizeof(unsigned int)*(TRA[L].size()), cudaMemcpyHostToDevice ));

                                            //    0  1  2  3  4  5  6  7  8  9,10
      const int vrr_team_size_vs_total_L[11] =  { 1, 4, 8,16,16,32,32,32,64,64,64 };
      int vrr_team_size = 64;
      if ( labcd < 11 ){ vrr_team_size = vrr_team_size_vs_total_L[labcd]; }

      timer.start();
      compute_VRR_batched_gpu_low<<<Ncells,128>>>(
         Ncells, plan_dev, Pm_input_list_dev, HRR_dev, Fm_dev, data_dev,
         AC_dev, ABCD_dev, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, vrr_team_size );

      compute_HRR_batched_gpu_low<<<Ncells,128>>>(
         Ncells, plan_dev, HRR_dev, data_dev, ABCD_dev, ABCD0_dev,
         hrr_blocksize, Nc, numVC, numVCH );

      compute_SPH_batched_gpu_low<<<Nqrtt,128>>>( Nqrtt, la, lb, lc, ld, ABCD0_dev, SPHER_dev, ABCD_dev );

      int corrBS = 64;
      int corrNB = (Nqrtt*Ns+corrBS-1)/corrBS;
      apply_correction<<<corrNB,corrBS>>>( Nqrtt*Ns, SPHER_dev, corr );

      compute_TRA_batched_gpu_low<<<Nshell,128>>>( Nshell, la, lb, lc, ld, TRA_dev, SPHER_dev, OUT_dev );

      CUDA_GPU_ERR_CHECK( cudaPeekAtLastError() );
      CUDA_GPU_ERR_CHECK( cudaDeviceSynchronize() );
      timer.stop();
      cout << " GPU KRNL :" << timer.elapsedMilliseconds() << " ms" << endl;

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
}



