#include <iostream>
#include "AIS.h"
#include "util.h"
#include "timer.h"
#include <cassert>
#include <unordered_map>
#include <tuple>
#include <cstdlib>
#include <ctime>

using std::cout;
using std::cerr;
using std::cout;
using std::cin;

// A hash function used to hash a tuple 
struct hash_tuple { 
    size_t operator()( const std::tuple<int,int,int,int>& x) const{ 
        return ((std::get<0>(x)*1024 + std::get<1>(x))*1024 + std::get<2>(x)) * 1024 + std::get<3>(x);
    } 
};



int main(int argc, char** argv){

bool skip_cpu = false;
if (argc > 1 and argv[1][0] == 'G' ){ skip_cpu = true; }


Timer timer;

timer.start();

int nbas, natom, env_size;
std::vector<int> bas;
std::vector<int> atm;
std::vector<double> env;
char mode;

cin >> mode ;
cout << "MODE: " << mode;
if ( skip_cpu ){ cout << " SKIP "; }
cout << endl;

cin >> nbas ;
for (int ibas=0; ibas < nbas; ibas++ ){
   int iatom, l, nz, nl, k, idx_z, idx_K, unused;
   cin >> iatom >> l >> nz >> nl >> k >> idx_z >> idx_K >> unused;

   bas.push_back(iatom);
   bas.push_back(l);
   bas.push_back(nz);
   bas.push_back(nl);
   bas.push_back(k);
   bas.push_back(idx_z);
   bas.push_back(idx_K);
   bas.push_back(unused);
}

cin >> natom ;
for (int iatom=0; iatom < natom; iatom++ ){
   int IZ, idx_R, Nmod, idx_Z, unused1, unused2;
   cin >> IZ >> idx_R >> Nmod >> idx_Z >> unused1 >> unused2 ;
   atm.push_back(IZ);
   atm.push_back(idx_R);
   atm.push_back(Nmod);
   atm.push_back(idx_Z);
   atm.push_back(unused1);
   atm.push_back(unused2);
}

cin >> env_size ;
for (int ienv=0; ienv < env_size; ienv++ ){
   double tmp;
   cin >> tmp;
   env.push_back(tmp);
}

typedef std::tuple<int,int,int,int> four_ints;
std::unordered_map< four_ints, unsigned int, hash_tuple > offset_set;
std::unordered_map< four_ints, unsigned int, hash_tuple > ld_set;

std::vector<double> SpDm;
std::vector<double> SpKS;
std::vector<double> my_F;


int len_offset;
cin >> len_offset;
for ( int uff=0; uff < len_offset; uff++ ){
   int set_i, set_j, atom_i,atom_j;
   unsigned int off,ld;
   cin >> set_i >> set_j >> atom_i >> atom_j >> off >> ld ;
   offset_set[four_ints(set_i,set_j,atom_i,atom_j)] = off;
   ld_set[four_ints(set_i,set_j,atom_i,atom_j)] = ld; 
}

int len_SpDm;
cin >> len_SpDm;

if ( mode == 'C' ){
   for( int idx_SpDm=0; idx_SpDm < len_SpDm; idx_SpDm++ ){
      double dm;
      cin >> dm;
      SpDm.push_back(dm);
   }

   int len_SpKS;
   cin >> len_SpKS;
   for( int idx_SpKS=0; idx_SpKS < len_SpKS; idx_SpKS++ ){
      double ks;
      cin >> ks;
      SpKS.push_back(ks);
   }
}

if ( mode == 'P' ){
   std::srand(std::time(nullptr));
   for( int idx_SpDm=0; idx_SpDm < len_SpDm; idx_SpDm++ ){
      SpDm.push_back( std::rand() / (RAND_MAX + 1u) );
      SpKS.push_back( std::rand() / (RAND_MAX + 1u) );
   }
}


int int_size;
cin >> int_size ;

double A1[3] = {10.,0.,0.};
double A2[3] = {0.,10.,0.};
double A3[3] = {0.,0.,10.};
double cell_h[9] = {A1[0],A1[1],A1[2], A2[0],A2[1],A2[2], A3[0],A3[1],A3[2]};
int n1_min = 0;
int n1_max = 0;
int n2_min = 0;
int n2_max = 0;
int n3_min = 0;
int n3_max = 0;
double PCells[3] = {0.,0.,0.};
bool periodic = false;

AIS ais;
ais.set_P ( SpDm );
// Since we all always adding to F, it is important that it starts from a value of zero
my_F.resize( SpKS.size(), 0.0 );
ais.set_K ( my_F );

ais.show_state();
ais.periodic = periodic;
for( int uff=0; uff < 9 ; uff++){
   ais.cell_h[uff] = cell_h[uff];
}

for ( int i = 0 ; i < nbas ; i ++ ){
for ( int j = 0 ; j < nbas ; j ++ ){

   double* RA = &env[ atm[ bas[i*8+0]*6+1 ]];
   double* RB = &env[ atm[ bas[j*8+0]*6+1 ]];
   int la_min = bas[i*8+1];
   int lb_min = bas[j*8+1];
   int la_max = bas[i*8+1];
   int lb_max = bas[j*8+1];
   int nza = bas[i*8+2];
   int nzb = bas[j*8+2];
   double* Za = &env[bas[i*8+5]];
   double* Zb = &env[bas[j*8+5]];
   ais.setA(RA,Za,nza);
   ais.setB(RB,Zb,nzb);
   ais.clearAl();
   ais.clearBl();
   for ( int la = la_min; la <= la_max; la++){
      int nla = bas[i*8+3];
      double* Ka = &env[bas[i*8+6]];
      ais.setAl( la, nla, Ka );
   }
   for ( int lb = lb_min; lb <= lb_max; lb++){
      int nlb = bas[j*8+3];
      double* Kb = &env[bas[j*8+6]];
      ais.setBl( lb, nlb, Kb );
   }

   for ( int k = 0 ; k < nbas ; k ++ ){
   for ( int l = 0 ; l < nbas ; l ++ ){

      double* RC = &env[ atm[ bas[k*8+0]*6+1 ]];
      int lc_min = bas[k*8+1];
      int lc_max = bas[k*8+1];
      int nzc = bas[k*8+2];
      double* Zc = &env[bas[k*8+5]];
      ais.setC(RC, Zc, nzc );
      ais.clearCl();
      for ( int lc = lc_min; lc <= lc_max; lc++){
         int nlc = bas[k*8+3];
         double* Kc = &env[bas[k*8+6]];
         ais.setCl( lc, nlc, Kc );
      }

      double* RD = &env[ atm[ bas[l*8+0]*6+1 ]];
      int ld_min = bas[l*8+1];
      int ld_max = bas[l*8+1];
      int nzd = bas[l*8+2];
      double* Zd = &env[bas[l*8+5]];

      ais.setD(RD, Zd, nzd );
      ais.clearDl();
      for ( int ld = ld_min; ld <= ld_max; ld++){
         int nld = bas[l*8+3];
         double* Kd = &env[bas[l*8+6]];
         ais.setDl( ld, nld, Kd );
      }

      int atom_i = bas[i*8+0];
      int atom_j = bas[j*8+0];
      int atom_k = bas[k*8+0];
      int atom_l = bas[l*8+0];
      bool screened_by_symm = false;
      if ( atom_i > atom_j ){ screened_by_symm = true; }
      if ( atom_k > atom_l ){ screened_by_symm = true; }
      if ( not ( (atom_k+atom_l) <= (atom_i+atom_j)) ){ screened_by_symm = true; }
      if ( ((atom_i + atom_j) == (atom_k + atom_l)) and (atom_k < atom_i) ){ screened_by_symm = true; }

      int offset_ac_atom = 0; // should be given by the dbscr system. We just use a big set,set,atom,atom matrix for simplicity
      int offset_ad_atom = 0;
      int offset_bc_atom = 0;
      int offset_bd_atom = 0;

      int ikind = atom_i; // no real kind in pyscf, use atom index as proxy
      int jkind = atom_j;
      int kkind = atom_k;
      int lkind = atom_l;

      double symm_fac = 0.5;

      if (atom_i == atom_j) { symm_fac *= 2.0; }
      if (atom_k == atom_l) { symm_fac *= 2.0; }
      if (atom_i == atom_k and atom_j == atom_l and atom_i != atom_j and atom_k != atom_l) { symm_fac *= 2.0; }
      if (atom_i == atom_k and atom_i == atom_j and atom_k == atom_l ) { symm_fac *= 2.0; }
      symm_fac = 1.0/symm_fac;
      if ( screened_by_symm ) { symm_fac = 0.0; }

      unsigned int offset_bd_L_set, offset_bc_L_set, offset_ad_L_set, offset_ac_L_set;
      int ld_bd_set, ld_bc_set, ld_ad_set, ld_ac_set;
      int Tbd, Tbc, Tad, Tac;
      // bd

      if ( atom_j >= atom_l ){
         four_ints set_atom(j,l,jkind,lkind);
         offset_bd_L_set = offset_bd_atom + offset_set[set_atom]; // ## no multi L set in pyscf
         ld_bd_set = ld_set[set_atom];
         Tbd = false;
      } else { 
         four_ints set_atom(l,j,lkind,jkind);
         offset_bd_L_set = offset_bd_atom + offset_set[set_atom];
         ld_bd_set = ld_set[set_atom];
         Tbd = true;
      }
      // bc
      if ( atom_j >= atom_k ){
         four_ints set_atom(j,k,jkind,kkind);
         offset_bc_L_set = offset_bc_atom + offset_set[set_atom];
         ld_bc_set = ld_set[set_atom];
         Tbc = false;
      } else {
         four_ints set_atom(k,j,kkind,jkind);
         offset_bc_L_set = offset_bc_atom + offset_set[set_atom];
         ld_bc_set = ld_set[set_atom];
         Tbc = true;
      }
      // ad
      if ( atom_i >= atom_l ){
         four_ints set_atom(i,l,ikind,lkind);
         offset_ad_L_set = offset_ad_atom + offset_set[set_atom];
         ld_ad_set = ld_set[set_atom];
         Tad = false;
      } else {
         four_ints set_atom(l,i,lkind,ikind);
         offset_ad_L_set = offset_ad_atom + offset_set[set_atom];
         ld_ad_set = ld_set[set_atom];
         Tad = true;
      }
      // ac
      if ( atom_i >= atom_k ){
         four_ints set_atom(i,k,ikind,kkind);
         offset_ac_L_set = offset_ac_atom + offset_set[set_atom];
         ld_ac_set = ld_set[set_atom];
         Tac = false;
      } else {
         four_ints set_atom(k,i,kkind,ikind);
         offset_ac_L_set = offset_ac_atom + offset_set[set_atom];
         ld_ac_set = ld_set[set_atom];
         Tac = true;
      }

      bool set_is_screened = true;
      for (int n1=n1_min; n1 <= n1_max ; n1++){

         double* R1 = &PCells[n1*3];
         // TODO real B in cp2k is shifted so that AB(0) = min(A-B) under cell
         double RBp[3] = {RB[0]+R1[0], RB[1]+R1[1], RB[2]+R1[2] };
         double AB[3] = {RA[0]-RBp[0], RA[1]-RBp[1], RA[2]-RBp[2]};


         ais.moveB(RBp);

         for (int n2=n2_min; n2 <= n2_max ; n2++){

            double* R2 = &PCells[n2*3];
            // TODO real D in cp2k is shifted so that CD(0) = min(C-D) under cell
            double RDp[3] = {RD[0]+R2[0], RD[1]+R2[1], RD[2]+R2[2] };
            double CD[3] = {RC[0]-RDp[0], RC[1]-RDp[1], RC[2]-RDp[2]};

            for (int n3=n3_min; n3 <= n3_max ; n3++){
               double RCp[3];
               double* R3 = &PCells[n3*3];
               if ( periodic ){
                  RCp[0] = RC[0]+R3[0];
                  RCp[1] = RC[1]+R3[1];
                  RCp[2] = RC[2]+R3[2];
                  RDp[0] = RD[0]+R3[0];
                  RDp[1] = RD[1]+R3[1];
                  RDp[2] = RD[2]+R3[2];
               } else {
                  RCp[0] = RC[0];
                  RCp[1] = RC[1];
                  RCp[2] = RC[2];
                  RDp[0] = RD[0];
                  RDp[1] = RD[1];
                  RDp[2] = RD[2];
               }
               bool cell_is_screened = true;
               ais.moveC(RCp);
               ais.moveD(RDp);

               for( int iza=0; iza < nza ; iza++){
               for( int izb=0; izb < nzb ; izb++){
//                  double za = env[bas[i*8+5]+iza];
//                  double zb = env[bas[j*8+5]+izb];
//                  double zab = za+zb;
//                  double P[3] = {0.,0.,0.};
//                  compute_weighted_distance(P,RA,RBp,za,zb,zab);

                  for( int izc=0; izc < nzc ; izc++){
                  for( int izd=0; izd < nzd ; izd++){
//                     double zc = env[bas[k*8+5]+izc];
//                     double zd = env[bas[l*8+5]+izd];
//                     double zcd = zc + zd;
//                     double Q[3] = {0.,0.,0.};
//                     compute_weighted_distance(Q,RCp,RDp,zc,zd,zcd);
   //                  double PQ0[3] = {P[0]-Q[0], P[1]-Q[1], P[2]-Q[2]};

//                     int n123[3] = {0,0,0};
//                     if (periodic){ }

//                     ais.add_prm(iza,izb,izc,izd,n123[0],n123[1],n123[2]);
                     ais.add_prm(iza,izb,izc,izd,0,0,0);

                     cell_is_screened = false;
                     set_is_screened = false;
                  }} // zc,zd
               }} // za,zb

               if ( not cell_is_screened ) {
                  ais.add_shell();
               }
               ais.add_cell();
            } // R3
         } // R2
      } // R1
      if ( not set_is_screened ){
         for ( int la = la_min; la <= la_max; la++){
         for ( int lb = lb_min; lb <= lb_max; lb++){
         for ( int lc = lc_min; lc <= lc_max; lc++){
         for ( int ld = ld_min; ld <= ld_max; ld++){
            int nla = bas[i*8+3];
            int nlb = bas[j*8+3];
            int nlc = bas[k*8+3];
            int nld = bas[l*8+3];
            ais.add_qrt(la,lb,lc,ld, nla,nlb,nlc,nld );
            for( int inla = 0 ; inla < nla; inla++ ){
            for( int inlb = 0 ; inlb < nlb; inlb++ ){
            for( int inlc = 0 ; inlc < nlc; inlc++ ){
            for( int inld = 0 ; inld < nld; inld++ ){
               ais.add_qrtt(
                  symm_fac, la,lb,lc,ld, 
                  inla,inlb,inlc,inld,
                  ld_ac_set,ld_ad_set,ld_bc_set,ld_bd_set, 
                  offset_ac_L_set,offset_ad_L_set,
                  offset_bc_L_set,offset_bd_L_set,
                  Tac,Tad,Tbc,Tbd );
            }}}}
         }}}}
         ais.add_set();
      }

      bool is_last_qrtt = (i==(nbas-1)) and (j==(nbas-1)) and (k==(nbas-1)) and (l==(nbas-1));
      if ( (k==0 and l==0 and ais.memory_needed() > 4.e9) or is_last_qrtt ){
   //      cout << " Prepare step: " << timer.elapsedMilliseconds() << endl;

         ais.dispatch(skip_cpu);

         if ( mode == 'C' ){
            int nerrors = 0;
            double diff_sum = 0.0;
            double adiff_sum = 0.0;
            int Nval = int(ais.OUT.size());

            for(int i=0; i < Nval; i++ ){
               double ref;
               cin >> ref;
               double val = ais.OUT[i];
               double diff = ref - val;
               double adiff = abs(diff);
               diff_sum += diff;
               adiff_sum += adiff;

               if ( adiff > 1.e-12 ){
                  nerrors++;
                  double ratio = 1.0;
                  if ( abs(ref) > 0. ){ ratio = val / ref ; }
                  cout << " I: CPU - REF: Error at " << i << " " << val << " " << ref 
                       << " " << diff << " " << ratio << " " << endl ;
                  if ( nerrors >= 100 ){
                     cout << " TOO MANY ERRORS ! EXITING NOW " << endl;
                     return EXIT_FAILURE ;
                  }
               }
            }
    
            cout << "I: E[ CPU-REF ] " << diff_sum / Nval << endl;
            cout << "I: E[|CPU-REF|] " << adiff_sum / Nval << endl;
            if ( nerrors > 0 ){ return EXIT_FAILURE ; }
         }
      }

   }} // bas cd
}} // bas ab

timer.stop();

// gets the KS
std::vector<double> my_final_F = ais.get_K( );
// stupid symmetry operator(s)
for ( int iset = 0 ; iset < nbas ; iset ++ ){
for ( int jset = 0 ; jset < nbas ; jset ++ ){
   int atom_i = bas[iset*8+0];
   int atom_j = bas[jset*8+0];
   if ( atom_i == atom_j ){
      int li = bas[iset*8+1];
      int nli = bas[iset*8+3];
      int di = (2*li+1)*nli;
            
      int lj = bas[jset*8+1];
      int nlj = bas[jset*8+3];
      int dj = (2*lj+1)*nlj;
            
      int i = offset_set[four_ints(iset,jset,atom_i,atom_j)];
      for ( int ma = 0; ma < di; ma++ ){
         int j = offset_set[four_ints(jset,iset,atom_j,atom_i)] + ma;
         for ( int mb = 0; mb < dj; mb++ ){
            if ( i > j ){
               my_final_F[i] = my_final_F[i] + my_final_F[j];
               my_final_F[j] = my_final_F[i];
            }
            i += 1;
            j += di;
         }
      }
   }
}}
// Applies a 0.5 factor to off-diagonal elements by multiplying by 0.5 and then doubling the diagonal
for ( int i=0; i < my_final_F.size() ; i++ ){ my_final_F[i] *= 0.5; }
for ( int iset = 0 ; iset < nbas ; iset ++ ){
   int atom_i = bas[iset*8+0];
   int li = bas[iset*8+1];
   int nli = bas[iset*8+3];
   int di = (2*li+1) * nli;
   int i0 = offset_set[four_ints(iset,iset,atom_i,atom_i)];
   for ( int ma = 0; ma < di; ma++ ){
      int i = i0 + ma * di + ma;
      my_final_F[i] *= 2.;
   }
}


if ( mode == 'C' ){
   cout << " Testing KS " << endl;
   cout.flush();
   int nerrors = 0;
   double diff_sum = 0.0;
   double adiff_sum = 0.0;
   int Nval = int( my_final_F.size());
   for(int i=0; i < Nval; i++ ){
      double ref = SpKS[i];
      double val = my_final_F[i];
      double diff = ref - val;
      double adiff = abs(diff);
      diff_sum += diff;
      adiff_sum += adiff;

      if ( adiff > 1.e-12 ){
         nerrors++;
         double ratio = 1.0;
         if ( abs(ref) > 0. ){ ratio = val / ref ; }
         cout << " F: CPU - REF: Error at " << i << " " << val << " " << ref
              << " " << diff << " " << ratio << " " << endl ;
         if ( nerrors >= 100 ){
            cout << " F: TOO MANY ERRORS ! EXITING NOW " << endl;
            return EXIT_FAILURE ;
         }
      }
   }

   cout << "F: E[ CPU-REF ] " << diff_sum / Nval << endl;
   cout << "F: E[|CPU-REF|] " << adiff_sum / Nval << endl;
   if ( nerrors > 0 ){ return EXIT_FAILURE ; }

}

// ais.show_state();
if ( mode == 'P' ){ ais.report_througput(skip_cpu); }

return EXIT_SUCCESS;
}









