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
#include "fgamma.h"

using std::max;

class AIS {
   public:
   AIS(){}
   void add_prm ( const int ipa, const int ipb, const int ipc, const int ipd, const int n1, const int n2, const int n3 ) ;
   void add_shell ( 
      int la, int lb, int lc, int ld, 
      double* Ka, double* Kb, double* Kc, double* Kd,
      double* A, double* B, double* C, double* D, 
      double* Za, double* Zb, double* Zc, double* Zd,
      int npa, int npb, int npc, int npd, int nla, int nlb, int nlc, int nld );
   void add_cell();
   int add_qrt( int la, int lb, int lc, int ld, int nla, int nlb, int nlc, int nld );
   void add_set();

   void dispatch();
   int out_size = 0;
   std::vector<double> OUT;
   bool periodic = false;
   void show_state();
//   private:

   unsigned int offset_F[NL4] = {0};
   unsigned int offset_V[NL4] = {0};
   unsigned int offset_G[NL4] = {0};
   unsigned int offset_Q[NL4] = {0};
   unsigned int offset_T[NL4] = {0};

   std::vector<unsigned int> FVH[NL4];
   std::vector<unsigned int> PMI[NL4];
   std::vector<unsigned int> SPH[NL4];
   std::vector<unsigned int> TRA[NL4];

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
};

#endif // #ifndef AIS_H_COMPILE_GUARD
