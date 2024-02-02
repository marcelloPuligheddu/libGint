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


#ifndef DEFINE_COMPILE_GUARD
#define DEFINE_COMPILE_GUARD

// 
#define PAL_SLOTS 3

#define BAS_SLOTS 8
#define BAS_OFFSET_L 1
#define BAS_OFFSET_NPGF 2
#define BAS_OFFSET_Z 5
#define BAS_OFFSET_C 6
#define BAS_OFFSET_R 7

#define Q_PARA_SLOTS 22

#define Q_IDXS_SLOTS 4
#define Q_IDXS_OFFSET_KA 0
#define Q_IDXS_OFFSET_KB 1
#define Q_IDXS_OFFSET_KC 2
#define Q_IDXS_OFFSET_KD 3

#define QRT_SIZE 14
#define QRT_OFFSET_A 0
#define QRT_OFFSET_B 1
#define QRT_OFFSET_C 2
#define QRT_OFFSET_D 3
#define QRT_OFFSET_PP 4
#define QRT_OFFSET_PS 5
#define QRT_OFFSET_NLA 6
#define QRT_OFFSET_NLB 7
#define QRT_OFFSET_NLC 8
#define QRT_OFFSET_NLD 9
#define QRT_OFFSET_NGA 10
#define QRT_OFFSET_NGB 11
#define QRT_OFFSET_NGC 12
#define QRT_OFFSET_NGD 13

// code for an operation
#define VRR1 1
#define VRR2 2
#define VRR3 3
#define VRR4 4
#define VRR5 5
#define VRR6 6
#define CP2S 7
#define HRR1 8
#define HRR2 9
#define SYTM 10
#define SYBL 11 
// T is the type of an operation. LA,LB,LC,LD are the moments of the results of the operation. 
// M and H are used for the m index of the operand. H is also used for the priority of the operation
// M1 to M6 are memory location. M1 is always the result.
#define OP_SIZE 13
#define T__OFFSET 0
#define LA_OFFSET 1
#define LB_OFFSET 2
#define LC_OFFSET 3
#define LD_OFFSET 4
#define M__OFFSET 5
#define H__OFFSET 6
#define M1_OFFSET 7
#define M2_OFFSET 8
#define M3_OFFSET 9
#define M4_OFFSET 10
#define M5_OFFSET 11
#define M6_OFFSET 12

#define MAX_NUM_VRR_TEAMS 16
#define MAX_NUM_QUARTETS 100000
#define MAX_NUM_BLOCKS 100000
#define MEGABYTE (1000*1000)
#define MAX_MEMORY_PER_RUN (1000*MEGABYTE)


#define L_BASE (32)
#define SERIAL_L4(a,b,c,d) ((((a)*L_BASE+(b))*L_BASE+(c))*L_BASE+(d))


#define FM_SIZE  11
#define PM_SIZE  2
#define VRR_SIZE 3
#define HRR_SIZE 16
#define SPH_SIZE 2
#define TRA_SIZE 6

#define FMA_OFFSET_LMAX    0
#define FMA_OFFSET_IDX_A   1
#define FMA_OFFSET_IDX_B   2
#define FMA_OFFSET_IDX_C   3
#define FMA_OFFSET_IDX_D   4
#define FMA_OFFSET_IDX_ZA  5
#define FMA_OFFSET_IDX_ZB  6
#define FMA_OFFSET_IDX_ZC  7
#define FMA_OFFSET_IDX_ZD  8
#define FMA_OFFSET_OF0     9
#define FMA_OFFSET_P0      10

#define PMA_OFFSET_IF      0
#define PMA_OFFSET_IPZN    1

#define VRR_OFFSET_OF      0
#define VRR_OFFSET_LMAX    1
#define VRR_OFFSET_IPZN    2

#define HRR_OFFSET_OV      0    
#define HRR_OFFSET_NPRM    1
#define HRR_OFFSET_OG      2
#define HRR_OFFSET_OQ      3
#define HRR_OFFSET_IDX_AB  4
#define HRR_OFFSET_IDX_CD  5 
#define HRR_OFFSET_NLABCD  6
#define HRR_OFFSET_PLABCD  7
#define HRR_OFFSET_PPABCD  8
#define HRR_OFFSET_IDX_KA  9
#define HRR_OFFSET_IDX_KB  10
#define HRR_OFFSET_IDX_KC  11
#define HRR_OFFSET_IDX_KD  12
#define HRR_OFFSET_LMAX    13
#define HRR_OFFSET_OF0     14
#define HRR_OFFSET_OP      15

#define SPH_OFFSET_Q 0
#define SPH_OFFSET_NLABCD 1

#define TRA_OFFSET_Q 0
#define TRA_OFFSET_NLA 1
#define TRA_OFFSET_NLB 2
#define TRA_OFFSET_NLC 3
#define TRA_OFFSET_NLD 4
#define TRA_OFFSET_DEST 5

#define MAX_L 7
#define NL ((MAX_L + 1))
#define NL2 ((NL*NL))
#define NL3 ((NL2*NL))
#define NL4 ((NL2*NL2))


#include<stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#define CUDA_GPU_ERR_CHECK(ans) { gpuAssertCUDA((ans), __FILE__, __LINE__); }
inline void gpuAssertCUDA( cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include "cublas_v2.h"
#define CUBLAS_GPU_ERR_CHECK(ans) { gpuAssertCublas((ans), __FILE__, __LINE__); }
static const char* my_cublasGetStatusString(cublasStatus_t error){
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

inline void gpuAssertCublas( cublasStatus_t code, const char *file, int line, bool abort=true){
   if( code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS error: %s %s %d\n", my_cublasGetStatusString(code), file, line);
      if (abort) exit(code);
   }
}


#endif
