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

// A my_hash function used to my_hash a v4
 

//int my_hash( int x0, int x1, int x2, int x3) { return ((x0*256 + x1)*256 + x2) * 256 + x3; }
int my_hash( int x0, int x1, int x2, int x3) { return x0*256 + x1; }



int main(int argc, char** argv){

bool skip_cpu = false;
if (argc > 1 and argv[1][0] == 'G' ){ skip_cpu = true; }


Timer timer;
//Timer timer_shell;
timer.start();

int nbas, natom, env_size;
std::vector<int> bas;
std::vector<int> atm;
std::vector<double> env;
char mode;
int nspin;


cin >> mode ;
cout << "MODE: " << mode;
cin >> nspin ;
cout << "NSPIN: " << nspin;

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


std::vector< unsigned int > offset_set( 256*256 );
std::vector< unsigned int > ld_set ( 256*256 );

std::vector<double> SpDm_a;
std::vector<double> SpKS_a;
std::vector<double> my_F_a;

std::vector<double> SpDm_b;
std::vector<double> SpKS_b;
std::vector<double> my_F_b;

// reads the structure of the density matrix
int len_offset;
cin >> len_offset;


int atom_set;
for ( int uff=0; uff < len_offset; uff++ ){
   int set_i, set_j, atom_i,atom_j;
   int off,ld;
   cin >> set_i >> set_j >> atom_i >> atom_j >> off >> ld ;
   atom_set = my_hash(set_i,set_j,atom_i,atom_j);
   offset_set[atom_set] = off;
   ld_set[atom_set] = ld; 
}


int len_SpDm;
cin >> len_SpDm;

if ( mode == 'C' ){
   // reads the actual sparse density matrix
   for( int idx_SpDm=0; idx_SpDm < len_SpDm; idx_SpDm++ ){
      double dm;
      cin >> dm;
      SpDm_a.push_back(dm);
   }
   if ( nspin == 2 ){
      for( int idx_SpDm=0; idx_SpDm < len_SpDm; idx_SpDm++ ){
         double dm;
         cin >> dm;
         SpDm_b.push_back(dm);
      } 
   }
   // reads the actual sparse fock matrix
   int len_SpKS;
   cin >> len_SpKS;
   for( int idx_SpKS=0; idx_SpKS < len_SpKS; idx_SpKS++ ){
      double ks;
      cin >> ks;
      SpKS_a.push_back(ks);
   }
   if ( nspin == 2 ){
      for( int idx_SpKS=0; idx_SpKS < len_SpKS; idx_SpKS++ ){
         double ks;
         cin >> ks;
         SpKS_b.push_back(ks);
      }
   }
}

if ( mode == 'P' ){
   std::srand(std::time(nullptr));
   for( int idx_SpDm=0; idx_SpDm < len_SpDm; idx_SpDm++ ){
      SpDm_a.push_back( std::rand() / (RAND_MAX + 1u) );
      SpKS_a.push_back( std::rand() / (RAND_MAX + 1u) );
   }
   if ( nspin == 2 ){
      SpDm_a.push_back( std::rand() / (RAND_MAX + 1u) );
      SpKS_a.push_back( std::rand() / (RAND_MAX + 1u) );  
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
// Since we all always adding to F, it is important that it starts from a value of zero
if ( nspin == 1 ){
   ais.set_P ( SpDm_a );
   my_F_a.resize( SpKS_a.size(), 0.0 );
   ais.set_K ( my_F_a );
} else {
   // polarized
   ais.set_P ( SpDm_a, SpDm_b );
   my_F_a.resize( SpKS_a.size(), 0.0 );
   my_F_b.resize( SpKS_b.size(), 0.0 );
   ais.set_K ( my_F_a, my_F_b );
}


ais.show_state();
ais.periodic = periodic;
for( int uff=0; uff < 9 ; uff++){
   ais.cell_h[uff] = cell_h[uff];
}

for ( int i = 0 ; i < nbas ; i ++ ){
   double* RA = &env[ atm[ bas[i*8+0]*6+1 ]];
   int la_min = bas[i*8+1];
   int la_max = bas[i*8+1];
   int nza = bas[i*8+2];
   double* Za = &env[bas[i*8+5]];
   ais.setA(i, RA, Za, nza);
   for ( int la = la_min; la <= la_max; la++){
      int nla = bas[i*8+3];
      double* Ka = &env[bas[i*8+6]];
      ais.setAl(i, la, nla, Ka );
   }
}

for ( int j = 0 ; j < nbas ; j ++ ){
   double* RB = &env[ atm[ bas[j*8+0]*6+1 ]];
   int lb_min = bas[j*8+1];
   int lb_max = bas[j*8+1];
   int nzb = bas[j*8+2];
   double* Zb = &env[bas[j*8+5]];
   ais.setB(j, RB, Zb, nzb);
   for ( int lb = lb_min; lb <= lb_max; lb++){
      int nlb = bas[j*8+3];
      double* Kb = &env[bas[j*8+6]];
      ais.setBl(j, lb, nlb, Kb );
   }
}

for ( int k = 0 ; k < nbas ; k ++ ){
   double* RC = &env[ atm[ bas[k*8+0]*6+1 ]];
   int lc_min = bas[k*8+1];
   int lc_max = bas[k*8+1];
   int nzc = bas[k*8+2];
   double* Zc = &env[bas[k*8+5]];
   ais.setC(k, RC, Zc, nzc );
   for ( int lc = lc_min; lc <= lc_max; lc++){
      int nlc = bas[k*8+3];
      double* Kc = &env[bas[k*8+6]];
      ais.setCl(k, lc, nlc, Kc );
   }
}

for ( int l = 0 ; l < nbas ; l ++ ){
   double* RD = &env[ atm[ bas[l*8+0]*6+1 ]];
   int ld_min = bas[l*8+1];
   int ld_max = bas[l*8+1];
   int nzd = bas[l*8+2];
   double* Zd = &env[bas[l*8+5]];
   ais.setD(l, RD, Zd, nzd );
   for ( int ld = ld_min; ld <= ld_max; ld++){
      int nld = bas[l*8+3];
      double* Kd = &env[bas[l*8+6]];
      ais.setDl(l, ld, nld, Kd );
   }
}

ais.set_max_n_prm( 1 );
ais.set_L();
for ( int i = 0 ; i < nbas ; i ++ ){
   double* RA = &env[ atm[ bas[i*8+0]*6+1 ]];
   int la_min = bas[i*8+1];
   int la_max = bas[i*8+1];
   int nza = bas[i*8+2];

for ( int j = 0 ; j < nbas ; j ++ ){
   double* RB = &env[ atm[ bas[j*8+0]*6+1 ]];
   int lb_min = bas[j*8+1];
   int lb_max = bas[j*8+1];
   int nzb = bas[j*8+2];

for ( int k = 0 ; k < nbas ; k ++ ){
   double* RC = &env[ atm[ bas[k*8+0]*6+1 ]];
   int lc_min = bas[k*8+1];
   int lc_max = bas[k*8+1];
   int nzc = bas[k*8+2];

for ( int l = 0 ; l < nbas ; l ++ ){
   double* RD = &env[ atm[ bas[l*8+0]*6+1 ]];
   int ld_min = bas[l*8+1];
   int ld_max = bas[l*8+1];
   int nzd = bas[l*8+2];


      // TODO: split the calculation of symm k, Tbd offset and ld to functions

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
         atom_set = my_hash(j,l,jkind,lkind);
         offset_bd_L_set = offset_bd_atom + offset_set[atom_set]; // ## no multi L set in pyscf
         ld_bd_set = ld_set[atom_set];
         Tbd = false;
      } else { 
         atom_set = my_hash(l,j,lkind,jkind);
         offset_bd_L_set = offset_bd_atom + offset_set[atom_set];
         ld_bd_set = ld_set[atom_set];
         Tbd = true;
      }
      // bc
      if ( atom_j >= atom_k ){
         atom_set = my_hash(j,k,jkind,kkind);
         offset_bc_L_set = offset_bc_atom + offset_set[atom_set];
         ld_bc_set = ld_set[atom_set];
         Tbc = false;
      } else {
         atom_set = my_hash(k,j,kkind,jkind);
         offset_bc_L_set = offset_bc_atom + offset_set[atom_set];
         ld_bc_set = ld_set[atom_set];
         Tbc = true;
      }
      // ad
      if ( atom_i >= atom_l ){
         atom_set = my_hash(i,l,ikind,lkind);
         offset_ad_L_set = offset_ad_atom + offset_set[atom_set];
         ld_ad_set = ld_set[atom_set];
         Tad = false;
      } else {
         atom_set = my_hash(l,i,lkind,ikind);
         offset_ad_L_set = offset_ad_atom + offset_set[atom_set];
         ld_ad_set = ld_set[atom_set];
         Tad = true;
      }
      // ac
      if ( atom_i >= atom_k ){
         atom_set = my_hash(i,k,ikind,kkind);
         offset_ac_L_set = offset_ac_atom + offset_set[atom_set];
         ld_ac_set = ld_set[atom_set];
         Tac = false;
      } else {
         atom_set = my_hash(k,i,kkind,ikind);
         offset_ac_L_set = offset_ac_atom + offset_set[atom_set];
         ld_ac_set = ld_set[atom_set];
         Tac = true;
      }

      bool set_is_screened = true;
      for (int n1=n1_min; n1 <= n1_max ; n1++){

         double* R1 = &PCells[n1*3];
         // TODO real B in cp2k is shifted so that AB(0) = min(A-B) under cell
         double RBp[3] = {RB[0]+R1[0], RB[1]+R1[1], RB[2]+R1[2] };
         double AB[3] = {RA[0]-RBp[0], RA[1]-RBp[1], RA[2]-RBp[2]};

//         ais.moveB(RBp);

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

//               ais.moveC(RCp);
//               ais.moveD(RDp);

               bool cell_is_screened = true;
               for( int iza=0; iza < nza ; iza++){
               for( int izb=0; izb < nzb ; izb++){
                  double za = env[bas[i*8+5]+iza];
                  double zb = env[bas[j*8+5]+izb];
                  double zab = za+zb;
                  double P[3] = {0.,0.,0.};
                  compute_weighted_distance(P,RA,RBp,za,zb,zab);

                  for( int izc=0; izc < nzc ; izc++){
                  for( int izd=0; izd < nzd ; izd++){
                     double zc = env[bas[k*8+5]+izc];
                     double zd = env[bas[l*8+5]+izd];
                     double zcd = zc + zd;
                     double Q[3] = {0.,0.,0.};
                     compute_weighted_distance(Q,RCp,RDp,zc,zd,zcd);
                     double PQ0[3] = {P[0]-Q[0], P[1]-Q[1], P[2]-Q[2]};

//                     int shift[3] = {0,0,0};
//                     if (periodic){ }

                     ais.add_prm(iza,izb,izc,izd,n1,n2,n3);

                     cell_is_screened = false;
                     set_is_screened = false;
                  }} // zc,zd
               }} // za,zb


               if ( not cell_is_screened ) {
                  ais.add_shell(i,j,k,l,n1,n2);
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
            ais.add_qrt(la,lb,lc,ld, nla,nlb,nlc,nld);
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
         cout << " Prepare step: " << timer.elapsedMilliseconds() << endl;
         cout.flush();
         timer.stop();

         timer.start();
         ais.dispatch(skip_cpu);
         timer.stop();
         cout << " Dispatch step: " << timer.elapsedMilliseconds() << endl;


         if ( mode == 'C' ){
            int nerrors = 0;
            double diff_sum = 0.0;
            double adiff_sum = 0.0;
            int Nval = int(ais.OUT.size());

            for(int ival=0; ival < Nval; ival++ ){
               double ref;
               cin >> ref;
               double val = ais.OUT[ival];
               double diff = ref - val;
               double adiff = abs(diff);
               diff_sum += diff;
               adiff_sum += adiff;

               if ( adiff > 1.e-12 ){
                  nerrors++;
                  double ratio = 1.0;
                  if ( abs(ref) > 0. ){ ratio = val / ref ; }
                  cout << " I: CPU - REF: Error at " << ival << " " << val << " " << ref 
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
         timer.start();
      }

   }} // bas cd
}} // bas ab

timer.stop();

// gets the KS

std::vector<double> my_final_F_a( my_F_a.size() );
std::vector<double> my_final_F_b( my_F_b.size() );

if ( nspin == 1 ){
   ais.get_K( my_final_F_a );
} else {
   ais.get_K( my_final_F_a,  my_final_F_b );
}



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

      int atom_set_ij = my_hash(iset,jset,atom_i,atom_j);
      int i = offset_set[atom_set_ij];
      for ( int ma = 0; ma < di; ma++ ){
         int atom_set_ji = my_hash(jset,iset,atom_j,atom_i);
         int j = offset_set[atom_set_ji] + ma;
         for ( int mb = 0; mb < dj; mb++ ){
            if ( i > j ){
               my_final_F_a[i] = my_final_F_a[i] + my_final_F_a[j];
               my_final_F_a[j] = my_final_F_a[i];
               if ( nspin == 2 ){
                  my_final_F_b[i] = my_final_F_b[i] + my_final_F_b[j];
                  my_final_F_b[j] = my_final_F_b[i];                
               }
            }
            i += 1;
            j += di;
         }
      }
   }
}}

// Applies a 0.5 factor to off-diagonal elements by multiplying everything by 0.5 and then doubling the diagonal
for ( int i=0; i < my_final_F_a.size() ; i++ ){ my_final_F_a[i] *= 0.5; }
if ( nspin == 2 ){
   for ( int i=0; i < my_final_F_b.size() ; i++ ){ my_final_F_b[i] *= 0.5; }
}

for ( int iset = 0 ; iset < nbas ; iset ++ ){
   int atom_i = bas[iset*8+0];
   int li = bas[iset*8+1];
   int nli = bas[iset*8+3];
   int di = (2*li+1) * nli;
   int atom_set_ii = my_hash(iset,iset,atom_i,atom_i);
   int i0 = offset_set[atom_set_ii];
   for ( int ma = 0; ma < di; ma++ ){
      int i = i0 + ma * di + ma;
      my_final_F_a[i] *= 2.;
      if ( nspin == 2 ){
         my_final_F_b[i] *= 2;
      }
   }
}


if ( mode == 'C' ){
   cout << " Testing KS " << endl;
   cout.flush();
   int nerrors = 0;
   double diff_sum = 0.0;
   double adiff_sum = 0.0;
   int Nval = int( my_final_F_a.size());
   for(int i=0; i < Nval; i++ ){

      double ref = SpKS_a[i];
      double val = my_final_F_a[i];
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
      if ( nspin == 2 ){
         ref = SpKS_b[i];
         val = my_final_F_b[i];
         diff = ref - val;
         adiff = abs(diff);
         diff_sum += diff;
         adiff_sum += adiff;

         if ( adiff > 1.e-12 ){
            nerrors++;
            double ratio = 1.0;
            if ( abs(ref) > 0. ){ ratio = val / ref ; }
            cout << " Fb: CPU - REF: Error at " << i << " " << val << " " << ref
                 << " " << diff << " " << ratio << " " << endl ;
            if ( nerrors >= 100 ){
               cout << " Fb: TOO MANY ERRORS ! EXITING NOW " << endl;
               return EXIT_FAILURE ;
            }
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









