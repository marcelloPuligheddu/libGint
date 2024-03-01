/*
Copyright (c) 2023 Science and Technology Facilities Council

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef AIS_H_COMPILE_GUARD
#define AIS_H_COMPILE_GUARD


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
#include "compute_KS.h"
#include "fgamma.h"

using std::max;

class AIS {
   public:
   AIS(){}
   void add_prm ( const int ipa, const int ipb, const int ipc, const int ipd, const int n1, const int n2, const int n3 ) ;
   void add_shell ();
   void add_cell();
   int add_qrt( int la, int lb, int lc, int ld, int nla, int nlb, int nlc, int nld );
   void add_qrtt(
         double symm_fac, int la, int lb, int lc, int ld, 
         int inla, int inlb, int inlc, int inld,
         int ld_ac, int ld_ad, int ld_bc, int ld_bd, 
         unsigned int offset_ac_L_set, unsigned int offset_ad_L_set, 
         unsigned int offset_bc_L_set, unsigned int offset_bd_L_set, 
         bool Tac, bool Tad, bool Tbc, bool Tbd );
   void add_set();
   void setA( double* A_,double*Za_,int npa_ );
   void setB( double* B_,double*Zb_,int npb_ );
   void setC( double* C_,double*Zc_,int npc_ );
   void setD( double* D_,double*Zd_,int npd_ );
   void clearAl();
   void clearBl();
   void clearCl();
   void clearDl();
   void setAl( int la_, int nla_, double* Ka_  );
   void setBl( int lb_, int nlb_, double* Kb_  );
   void setCl( int lc_, int nlc_, double* Kc_  );
   void setDl( int ld_, int nld_, double* Kd_  );
   void moveA( double * A );
   void moveB( double * B );
   void moveC( double * C );
   void moveD( double * D );
   void compute_max_vector_size();
   size_t memory_needed();

   void dispatch(bool skip_cpu);
   size_t out_size = 0;

   std::vector<double> OUT;
   bool periodic = false;
   void show_state();
   void report_througput(bool skip_gpu);
//   private:

   void reset_indices();

   double *A, *B, *C, *D;
   double *Za, *Zb, *Zc, *Zd;
   int npa, npb, npc, npd;
   int nnla=0, nnlb=0, nnlc=0, nnld=0;
   unsigned int idx_A, idx_B, idx_C, idx_D;
   unsigned int idx_Za, idx_Zb, idx_Zc, idx_Zd;
   std::vector<int> all_la,all_lb,all_lc,all_ld;
   std::vector<int> all_nla,all_nlb,all_nlc,all_nld;
   std::vector<unsigned int> all_idx_Ka, all_idx_Kb, all_idx_Kc, all_idx_Kd;

   double * K; // not owned by AIS
   double * P; // not owned by AIS
   double * K_dev; // owned and managed by AIS
   double * P_dev; // owned and managed by AIS
   size_t FP_size;
   std::vector<double> K_from_dev; // owned and managed by AIS
   void set_P( std::vector<double> & P );
   void set_K( std::vector<double> & P );
   std::vector<double> get_K( );
 
   size_t max_integral_scratch_size = 0;
   size_t max_plan_size = 0;
   size_t max_PMI_size = 0;
   size_t max_FVH_size = 0;
   size_t max_SPH_size = 0;
   size_t max_KS_size = 0;
   size_t max_TRA_size = 0;

   unsigned int offset_F[NL4] = {0};
   unsigned int offset_V[NL4] = {0};
   unsigned int offset_G[NL4] = {0};
   unsigned int offset_Q[NL4] = {0};
   unsigned int offset_T[NL4] = {0};
   int all_vrr_blocksize[NL4] = {0};
   int all_hrr_blocksize[NL4] = {0};

   std::vector<unsigned int> FVH[NL4];
   std::vector<unsigned int> PMI[NL4];
   std::vector<unsigned int> SPH[NL4];
   std::vector<unsigned int> TRA[NL4];
   std::vector<unsigned int> KS[NL4];

   unsigned int dest=0;
   std::vector<unsigned int> prm_tmp_list;
   UniqueArray ua;
//   bool is_gamma = true;
   unsigned int n_set = 0;
   unsigned int prm_in_set = 0;
   unsigned int prm = 0;
   unsigned int p0 = 0;
   unsigned int cell_in_set = 0;
   PlanCollection plans;

   unsigned int Fm_size[NL4] = {0};
   unsigned int AC_size[NL4] = {0};
   unsigned int ABCD_size[NL4] = {0};
   unsigned int ABCD0_size[NL4] = {0};
   unsigned int SPHER_size[NL4] = {0};
   unsigned int OUT_size[NL4] = {0};

   double cell_h[9] = {0};
   double cell_inv_h[9] = {0};

   std::unordered_set<unsigned int> encoded_moments ;
   bool first = true;
   std::unordered_set<unsigned int> all_moments ;
   std::vector<size_t> record_of_out_sizes[NL4];
   std::vector<double> record_of_times_cpu[NL4];
   std::vector<double> record_of_times_gpu[NL4];

};

#endif // #ifndef AIS_H_COMPILE_GUARD
