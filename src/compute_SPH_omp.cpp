#include <vector>
//#include "util.h"
#include "define.h"
//#include "c2s.h"
#include "compute_SPH_omp.h"
#include <iostream>

using std::cout;
using std::endl;


constexpr double c2s[1*1+3*3+5*6+7*10+9*15] = {
1.0,

0.0, 1.0, 0.0, 
0.0, 0.0, 1.0, 
1.0, 0.0, 0.0, 

 0., 1., 0., 0., 0., 0.,
 0., 0., 0., 0., 1., 0.,
 -0.28867513459,0., 0., -0.28867513459, 0., 0.57735026919,
 0., 0., 1., 0., 0., 0.,
 0.5, 0., 0., -0.5, 0., 0.,

 0,0.612372436,0,0,0,0,-0.204124145,0,0,0,
 0,0,0,0,1,0,0,0,0,0,
 0,-0.158113883,0,0,0,0,-0.158113883,0,0.632455532,0,
 0,0,-0.387298335,0,0,0,0,-0.387298335,0,0.25819889,
-0.158113883,0,0,-0.158113883,0,0.632455532,0,0,0,0,
 0,0,0.5,0,0,0,0,-0.5,0,0,
 0.204124145,0,0,-0.612372436,0,0,0,0,0,0,

0,0.288675135,0,0,0,0,-0.288675135,0,0,0,0,0,0,0,0
,0,0,0,0,0.612372436,0,0,0,0,0,0,-0.204124145,0,0,0
,0,-0.109108945,0,0,0,0,-0.109108945,0,0.654653671,0,0,0,0,0,0
,0,0,0,0,-0.231455025,0,0,0,0,0,0,-0.231455025,0,0.3086067,0
,0.036596253,0,0,0.073192505,0,-0.292770022,0,0,0,0,0.036596253,0,-0.292770022,0,0.097590007
,0,0,-0.231455025,0,0,0,0,-0.231455025,0,0.3086067,0,0,0,0,0
,-0.054554473,0,0,0,0,0.327326835,0,0,0,0,0.054554473,0,-0.327326835,0,0
,0,0,0.204124145,0,0,0,0,-0.612372436,0,0,0,0,0,0,0
,0.072168784,0,0,-0.433012702,0,0,0,0,0,0,0.072168784,0,0,0,0

};
template< int l, int s >
constexpr int first_cart_contr_to_sph( ){
   if constexpr ( l == 0 ){ return 0; }

   if constexpr ( l == 1 ){ 
      if constexpr( s == 0 ){ return 1; }
      if constexpr( s == 1 ){ return 2; }
      if constexpr( s == 2 ){ return 0; }
   }
   if constexpr ( l == 2 ){ 
      if constexpr( s == 0 ){ return 1; }
      if constexpr( s == 1 ){ return 4; }
      if constexpr( s == 2 ){ return 0; }
      if constexpr( s == 3 ){ return 2; }
      if constexpr( s == 4 ){ return 0; }
   }
   if constexpr ( l == 3 ){ 
      if constexpr( s == 0 ){ return 1; }
      if constexpr( s == 1 ){ return 4; }
      if constexpr( s == 2 ){ return 1; }
      if constexpr( s == 3 ){ return 2; }
      if constexpr( s == 4 ){ return 0; }
      if constexpr( s == 5 ){ return 2; }
      if constexpr( s == 6 ){ return 0; }
   }
}

constexpr int c2s_ptr[5] = {0, 1*1, 1*1+3*3, 1*1+3*3+5*6, 1*1+3*3+5*6+7*10};

constexpr int NC( int la ){ return (la+1)*(la+2)/2; }
constexpr int NS( int la ){ return (2*la+1); }

template < int la, int ac, int as, int N, int BS >
void sph_term( 
      double * const __restrict__ I,
      double * const __restrict__ O ){
   constexpr int nca = NC(la);
   constexpr int nsa = NS(la);

//   const int tid = threadIdx.x;

   if constexpr ( ac < nca and as < nsa ){
      constexpr double C = c2s[ c2s_ptr[la] + as*nca + ac];
      if constexpr ( C != 0 ){
         #pragma omp parallel for
         for ( int n=0; n<N; n++ ){
            if constexpr ( first_cart_contr_to_sph<la,as>() == ac ){
               O[as+n*NS(la)]  = C * I[N*ac+n];
            } else {
               O[as+n*NS(la)] += C * I[N*ac+n];
            }
         }
      }
      if constexpr ( ac+1 < nca ){ sph_term<la,ac+1,as,N,BS>(I,O); }
      if constexpr ( ac+1 == nca and as+1 < nsa ){ sph_term<la,0,as+1,N,BS>(I,O); }
      if constexpr ( ac+1 == nca and as+1 == nsa ){  }
   }
}

typedef void (*sph_func_t)(double* __restrict__, double* __restrict__, double* __restrict__);

template< int la, int lb, int lc, int ld, int BS >
void sph( double * block_ABCD0, double * block_SPHER, double * block_tmp ){
   sph_term< la, 0,0, NC(lb)*NC(lc)*NC(ld), BS >( block_ABCD0, block_tmp );
   sph_term< lb, 0,0, NS(la)*NC(lc)*NC(ld), BS >( block_tmp, block_ABCD0 );
   sph_term< lc, 0,0, NS(la)*NS(lb)*NC(ld), BS >( block_ABCD0, block_tmp );
   sph_term< ld, 0,0, NS(la)*NS(lb)*NS(lc), BS >( block_tmp, block_SPHER ); 
}

#define INSTANTIATE_SPH(la, lb, lc, ld, BS) sph< la,lb,lc,ld, BS >

#define INSTANTIATE_SPH_LD(la, lb, lc, BS ) \
  INSTANTIATE_SPH(la, lb, lc, 0, BS), \
  INSTANTIATE_SPH(la, lb, lc, 1, BS), \
  INSTANTIATE_SPH(la, lb, lc, 2, BS), \
  INSTANTIATE_SPH(la, lb, lc, 3, BS)

#define INSTANTIATE_SPH_LC(la, lb, BS) \
  INSTANTIATE_SPH_LD(la, lb, 0, BS), \
  INSTANTIATE_SPH_LD(la, lb, 1, BS), \
  INSTANTIATE_SPH_LD(la, lb, 2, BS), \
  INSTANTIATE_SPH_LD(la, lb, 3, BS)

#define INSTANTIATE_SPH_LB(la, BS) \
  INSTANTIATE_SPH_LC(la, 0, BS), \
  INSTANTIATE_SPH_LC(la, 1, BS), \
  INSTANTIATE_SPH_LC(la, 2, BS), \
  INSTANTIATE_SPH_LC(la, 3, BS)

#define INSTANTIATE_ALL_SPH(BS) \
  INSTANTIATE_SPH_LB(0, BS), \
  INSTANTIATE_SPH_LB(1, BS), \
  INSTANTIATE_SPH_LB(2, BS), \
  INSTANTIATE_SPH_LB(3, BS)

sph_func_t get_sph( int idx ){
   sph_func_t sph_funcs[] = {
      INSTANTIATE_ALL_SPH( 64 )
   };
   return sph_funcs[idx];
}

void compute_SPH_omp(
      const int Nqrtt, int la, int lb, int lc, int ld,
      double* const __restrict__ ABCD0,
      double* const __restrict__ SPHER,
      double* const __restrict__ tmp_scratch ){

   #pragma omp target teams distribute is_device_ptr(ABCD0,SPHER,tmp_scratch) depend( in:ABCD0,tmp_scratch) depend(out:SPHER)
   for( int b=0; b < Nqrtt; b++ ){
      double * block_ABCD0 = ABCD0 + b * NC(la)*NC(lb)*NC(lc)*NC(ld);
      double * block_tmp   = tmp_scratch + b * NC(la)*NC(lb)*NC(lc)*NC(ld);
      double * block_SPHER = SPHER + b * NS(la)*NS(lb)*NS(lc)*NS(ld);

      int idx = la * 64 + lb * 16 + lc * 4 + ld;

      get_sph( idx )( block_ABCD0, block_SPHER, block_tmp );

  }
}

