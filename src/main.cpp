#include <iostream>
#include "AIS.h"
#include "util.h"
#include "timer.h"


using std::cout;
using std::cerr;
using std::cout;
using std::cin;

int main(int argc, char** argv){

Timer timer;

timer.start();
int nbas, natom, env_size;
std::vector<int> bas;
std::vector<int> atm;
std::vector<double> env;
char mode;

cin >> mode ;
cout << "MODE: " << mode << endl;

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
ais.show_state();
ais.periodic = periodic;
for( int uff=0; uff < 9 ; uff++){
   ais.cell_h[uff] = cell_h[uff];
}

for ( int i = 0 ; i < nbas ; i ++ ){
for ( int j = 0 ; j < nbas ; j ++ ){
for ( int k = 0 ; k < nbas ; k ++ ){
for ( int l = 0 ; l < nbas ; l ++ ){

   double* RA = &env[ atm[ bas[i*8+0]*6+1 ]];
   double* RB = &env[ atm[ bas[j*8+0]*6+1 ]];
   double* RC = &env[ atm[ bas[k*8+0]*6+1 ]];
   double* RD = &env[ atm[ bas[l*8+0]*6+1 ]];

   int la_min = bas[i*8+1];
   int lb_min = bas[j*8+1];
   int lc_min = bas[k*8+1];
   int ld_min = bas[l*8+1];

   int la_max = bas[i*8+1];
   int lb_max = bas[j*8+1];
   int lc_max = bas[k*8+1];
   int ld_max = bas[l*8+1];

   int nza = bas[i*8+2];
   int nzb = bas[j*8+2];
   int nzc = bas[k*8+2];
   int nzd = bas[l*8+2];

   bool set_is_screened = true;

   for (int n1=n1_min; n1 <= n1_max ; n1++){

      double* R1 = &PCells[n1*3];
      // TODO real B in cp2k is shifted so that AB(0) = min(A-B) under cell
      double RBp[3] = {RB[0]+R1[0], RB[1]+R1[1], RB[2]+R1[2] };
      double AB[3] = {RA[0]-RBp[0], RA[1]-RBp[1], RA[2]-RBp[2]};

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
//                  double PQ0[3] = {P[0]-Q[0], P[1]-Q[1], P[2]-Q[2]};

                  int n123[3] = {0,0,0};
                  if (periodic){
                     
                  }

                  ais.add_prm(iza,izb,izc,izd,n123[0],n123[1],n123[2]);

                  cell_is_screened = false;
                  set_is_screened = false;
               }} // zc,zd
            }} // za,zb

            double* Za = &env[bas[i*8+5]];
            double* Zb = &env[bas[j*8+5]];
            double* Zc = &env[bas[k*8+5]];
            double* Zd = &env[bas[l*8+5]];

            if ( not cell_is_screened ) {
               for ( int la = la_min; la <= la_max; la++){
               for ( int lb = lb_min; lb <= lb_max; lb++){
               for ( int lc = lc_min; lc <= lc_max; lc++){
               for ( int ld = ld_min; ld <= ld_max; ld++){
                  double* Ka = &env[bas[i*8+6]];
                  double* Kb = &env[bas[j*8+6]];
                  double* Kc = &env[bas[k*8+6]];
                  double* Kd = &env[bas[l*8+6]];
                  int nla = bas[i*8+3];
                  int nlb = bas[j*8+3];
                  int nlc = bas[k*8+3];
                  int nld = bas[l*8+3];
                  ais.add_shell(la,lb,lc,ld,Ka,Kb,Kc,Kd, RA,RBp,RCp,RDp, Za,Zb,Zc,Zd, nza,nzb,nzc,nzd, nla,nlb,nlc,nld);
               }}}} // la lb lc ld
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
      }}}}
      ais.add_set();
   }
}}}} // bas abcd

timer.stop();
cout << " Prepare step: " << timer.elapsedMilliseconds() << endl;

ais.dispatch();

// ais.show_state();


if ( mode == 'P' ){ return EXIT_SUCCESS; }

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
      cout << " CPU - REF: Error at " << i << " " << val << " " << ref << " " << diff << " " << ratio << " " << endl ;
      if ( nerrors >= 100 ){
         cout << " TOO MANY ERRORS ! EXITING NOW " << endl;
         return EXIT_FAILURE ;
      }
   }

}

cout << "E[ CPU-REF ] " << diff_sum / Nval << endl;
cout << "E[|CPU-REF|] " << adiff_sum / Nval << endl;
return EXIT_SUCCESS;
}









