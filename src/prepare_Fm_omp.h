#ifndef PREPARE_FM_OMP_H_COMPILE_GUARD
#define PREPARE_FM_OMP_H_COMPILE_GUARD

// Fills Fm with intermediate vectors
void prepare_Fm_omp( 
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const int Ng );
#endif
