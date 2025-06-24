#include "define.h"
#include "prepare_Fm.h"

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

