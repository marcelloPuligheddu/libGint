#include <vector>
//#include "util.h"
#include "define.h"
//#include "c2s.h"
#include "compute_SPH_omp.h"
#include <iostream>

constexpr int inline NC( int la ){ return (la+1)*(la+2)/2; }
constexpr int inline NS( int la ){ return (2*la+1); }

template < int la, int N >
void inline sph_term(
      double * const __restrict__ I,
      double * const __restrict__ O )
{
   if constexpr ( la == 0 ){
      #pragma omp parallel for
      for( int n = 0; n < N; n++ ){
         O[0+n*1] = 1.0* I[N*0+n];
      }
   }
   if constexpr ( la == 1 ){
      #pragma omp parallel for
      for( int n = 0; n < N; n++ ){
         O[0+n*3] = 1.0* I[N*1+n];
         O[1+n*3] = 1.0* I[N*2+n];
         O[2+n*3] = 1.0* I[N*0+n];
      }
   }
   if constexpr ( la == 2 ){
      #pragma omp parallel for
      for( int n = 0; n < N; n++ ){
         O[0+n*5] = 1.0* I[N*1+n];
         O[1+n*5] = 1.0* I[N*4+n];
         O[2+n*5] = -0.28867513459* I[N*0+n]+-0.28867513459* I[N*3+n]; +0.57735026919* I[N*5+n];
         O[3+n*5] = 1.0* I[N*2+n];
         O[4+n*5] = 0.5* I[N*0+n]+-0.5* I[N*3+n]; 
      }
   }
   if constexpr ( la == 3 ){
      #pragma omp parallel for
      for( int n = 0; n < N; n++ ){
         O[0+n*7] = 0.612372436* I[N*1+n]+-0.204124145* I[N*6+n]; 
         O[1+n*7] = 1* I[N*4+n];
         O[2+n*7] = -0.158113883* I[N*1+n]+-0.158113883* I[N*6+n]; +0.632455532* I[N*8+n]; 
         O[3+n*7] = -0.387298335* I[N*2+n]+-0.387298335* I[N*7+n]; +0.25819889* I[N*9+n]; 
         O[4+n*7] = -0.158113883* I[N*0+n]+-0.158113883* I[N*3+n]; +0.632455532* I[N*5+n]; 
         O[5+n*7] = 0.5* I[N*2+n]+-0.5* I[N*7+n]; 
         O[6+n*7] = 0.204124145* I[N*0+n]+-0.612372436* I[N*3+n]; 
      }
   }
   if constexpr ( la == 4 ){
      #pragma omp parallel for
      for( int n = 0; n < N; n++ ){
         O[0+n*9] = 0.288675135* I[N*1+n]+-0.288675135* I[N*6+n]; 
         O[1+n*9] = 0.612372436* I[N*4+n]+-0.204124145* I[N*11+n]; 
         O[2+n*9] = -0.109108945* I[N*1+n]+-0.109108945* I[N*6+n]; +0.654653671* I[N*8+n]; 
         O[3+n*9] = -0.231455025* I[N*4+n]+-0.231455025* I[N*11+n]; +0.3086067* I[N*13+n]; 
         O[4+n*9] = 0.036596253* I[N*0+n]+0.073192505* I[N*3+n]; +-0.292770022* I[N*5+n]; +0.036596253* I[N*10+n]; +-0.292770022* I[N*12+n]; +0.097590007* I[N*14+n]; 
         O[5+n*9] = -0.231455025* I[N*2+n]+-0.231455025* I[N*7+n]; +0.3086067* I[N*9+n]; 
         O[6+n*9] = -0.054554473* I[N*0+n]+0.327326835* I[N*5+n]; +0.054554473* I[N*10+n]; +-0.327326835* I[N*12+n]; 
         O[7+n*9] = 0.204124145* I[N*2+n]+-0.612372436* I[N*7+n]; 
         O[8+n*9] = 0.072168784* I[N*0+n]+-0.433012702* I[N*3+n]; +0.072168784* I[N*10+n]; 
      }
   }
}

template < int la, int lb, int lc, int ld >
void sph_omp_templated( const int Nqrtt, double* const __restrict__ ABCD0,
      double* const __restrict__ SPHER,
      double* const __restrict__ tmp_scratch )
{
   #pragma omp target teams distribute is_device_ptr(ABCD0,SPHER,tmp_scratch) depend( in:ABCD0,tmp_scratch) depend(out:SPHER)
   for( int b=0; b < Nqrtt; b++ ){
      double * block_ABCD0 = ABCD0 + b * NC(la)*NC(lb)*NC(lc)*NC(ld);
      double * block_tmp   = tmp_scratch + b * NC(la)*NC(lb)*NC(lc)*NC(ld);
      double * block_SPHER = SPHER + b * NS(la)*NS(lb)*NS(lc)*NS(ld);
      sph_term< la, NC(lb)*NC(lc)*NC(ld) >( block_ABCD0, block_tmp );
      sph_term< lb, NS(la)*NC(lc)*NC(ld) >( block_tmp, block_ABCD0 );
      sph_term< lc, NS(la)*NS(lb)*NC(ld) >( block_ABCD0, block_tmp );
      sph_term< ld, NS(la)*NS(lb)*NS(lc) >( block_tmp, block_SPHER );
  }
}


typedef void (*sph_func_t)(const int Nqrtt, double* __restrict__, double* __restrict__, double* __restrict__);


#define INSTANTIATE_SPH(la, lb, lc, ld ) template void sph_omp_templated<la, lb, lc, ld >(const int, double*, double*, double*);
#define SPT_ENTRY(la, lb, lc, ld) &sph_omp_templated<la, lb, lc, ld>
#define SPH_MAX_L 4

#define INSTANTIATE_SPH_LD(la, lb, lc ) \
  INSTANTIATE_SPH(la, lb, lc, 0); \
  INSTANTIATE_SPH(la, lb, lc, 1); \
  INSTANTIATE_SPH(la, lb, lc, 2); \
  INSTANTIATE_SPH(la, lb, lc, 3);

#define INSTANTIATE_SPH_LC(la, lb) \
  INSTANTIATE_SPH_LD(la, lb, 0); \
  INSTANTIATE_SPH_LD(la, lb, 1); \
  INSTANTIATE_SPH_LD(la, lb, 2); \
  INSTANTIATE_SPH_LD(la, lb, 3);

#define INSTANTIATE_SPH_LB(la) \
  INSTANTIATE_SPH_LC(la, 0); \
  INSTANTIATE_SPH_LC(la, 1); \
  INSTANTIATE_SPH_LC(la, 2); \
  INSTANTIATE_SPH_LC(la, 3);

#define INSTANTIATE_ALL_SPH \
  INSTANTIATE_SPH_LB(0); \
  INSTANTIATE_SPH_LB(1); \
  INSTANTIATE_SPH_LB(2); \
  INSTANTIATE_SPH_LB(3);

INSTANTIATE_ALL_SPH

#define FOR_EACH_LD_PTR(la, lb, lc) \
    SPT_ENTRY(la, lb, lc, 0), \
    SPT_ENTRY(la, lb, lc, 1), \
    SPT_ENTRY(la, lb, lc, 2), \
    SPT_ENTRY(la, lb, lc, 3)

#define FOR_EACH_LC_PTR(la, lb) \
    FOR_EACH_LD_PTR(la, lb, 0), \
    FOR_EACH_LD_PTR(la, lb, 1), \
    FOR_EACH_LD_PTR(la, lb, 2), \
    FOR_EACH_LD_PTR(la, lb, 3)

#define FOR_EACH_LB_PTR(la) \
    FOR_EACH_LC_PTR(la, 0), \
    FOR_EACH_LC_PTR(la, 1), \
    FOR_EACH_LC_PTR(la, 2), \
    FOR_EACH_LC_PTR(la, 3)

#define SPT_TABLE_ENTRIES \
    FOR_EACH_LB_PTR(0), \
    FOR_EACH_LB_PTR(1), \
    FOR_EACH_LB_PTR(2), \
    FOR_EACH_LB_PTR(3)

sph_func_t sph_table[] = {
    SPT_TABLE_ENTRIES
};

void compute_SPH_omp(
      const int Nqrtt, int la, int lb, int lc, int ld,
      double* const __restrict__ ABCD0,
      double* const __restrict__ SPHER,
      double* const __restrict__ tmp_scratch ){
   int idx = (( la * SPH_MAX_L + lb ) * SPH_MAX_L + lc) * SPH_MAX_L + ld;
   sph_table[idx]( Nqrtt, ABCD0, SPHER, tmp_scratch );
}

