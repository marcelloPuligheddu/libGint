//#include <cmath>
//#include <omp.h>
//#include <vector>
#include "define.h"
//#include "util.h"
//#include "fgamma.h"
//#include "t_c_g0_n.h"
#include "prepare_Fm.h"
//#include <iostream>

//#ifdef __LIBGINT_OMP_OFFLOAD
    #include "prepare_Fm_omp.h"
//#else
//    #include "hip/hip_runtime.h"
//    #include "hipblas/hipblas.h"
//    #include "prepare_Fm_dev.h"
//#endif

#define SQRT2 1.4142135623730950488016887242096980785696718753
//#define BLKIDX 1

//using std::cout;
//using std::endl;

//#ifdef __LIBGINT_OMP_OFFLOAD
void prepare_Fm( 
      const unsigned int* const __restrict__ FVH_dev,
      const unsigned int* const __restrict__ OF_dev,
      const unsigned int* const __restrict__ PMX_dev,
      const double* const __restrict__ data_dev,
      double* const __restrict__ Fm_dev,
      int Nprm, int labcd, bool periodic,
      const double* const __restrict__ cell_h_dev,
      const double* const __restrict__ neighs_dev,
      const int max_ncells  ){
    prepare_Fm_omp_blocking( FVH_dev, OF_dev, PMX_dev, data_dev, Fm_dev, Nprm, labcd,
         periodic, cell_h_dev, neighs_dev, max_ncells );
}
//#else
/*void prepare_Fm(
      const unsigned int* const __restrict__ FVH_dev,
      const unsigned int* const __restrict__ OF_dev,
      const unsigned int* const __restrict__ PMX_dev,
      const double* const __restrict__ data_dev,
      double* const __restrict__ Fm_dev,
      int Nprm, int labcd, bool periodic,
      const double* const __restrict__ cell_h_dev,
      const double* const __restrict__ neighs_dev,
      const int max_ncells, hipStream_t hip_stream  ){
    int prep_Fm_blocksize = 128;
    int prep_Fm_numblocks = (Nprm+prep_Fm_blocksize-1)/prep_Fm_blocksize;
    prepare_Fm_batched_gpu_low_private<<<prep_Fm_numblocks,prep_Fm_blocksize,0,hip_stream>>>(
        FVH_dev, OF_dev, PMX_dev, data_dev, Fm_dev, Nprm, labcd,
        periodic, cell_h_dev, neighs_dev, max_ncells );
}*/
//#endif



