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

// TODO remove magic numbers
// Due to accumulation of arith errors, it is possible for the QP wrapping
// to become unstable and incompatible with the rounding used on the CPU side
// for the bra and ket primitive screening.
#define EPS_ROUNDING 1.e-9
// Batches that compute less than this number of
// \"integrals between spherical harmonic basis functions\"
// \"after contraction and after sum over lattice vectors\"
// will be deferred for calculations at a later time
#define MIN_INT_BATCH_SIZE 100




// max number of periodic cell, 2**8
#define MAX_N_CELL 256 
//#define MAX_N_CELL 100 // Useful for debug, so the encoding is human readable

// max number of prm per STO, 2**6. Way larger than sanity should allow
#define MAX_N_PRM   16
//#define MAX_N_PRM   10 // Useful for debug, so the encoding is human readable
// max number of linear combinations of 
// gassian in the same set and with the same ang. moment, 2**4.
// E.g. this cp2k set with 23 linear combinations of 32 gaussians is not valid
// 1
// 1 0 0 32 23 
// [exp and gcc omitted]

#define MAX_N_L     4
//#define MAX_N_L     10 // Useful for debug, so the encoding is human readable

// Should be 8 + 4*6 = 32 bits
static_assert( (unsigned long int) MAX_N_CELL * MAX_N_PRM * MAX_N_PRM * MAX_N_PRM * MAX_N_PRM <= 4294967296,
      "Error, MAX_N_PRM or MAX_N_CELL are too large" );
// Should be 8 + 8 + 4*4 = 32 bits
static_assert( (unsigned long int) MAX_N_CELL * MAX_N_CELL * MAX_N_L * MAX_N_L * MAX_N_L * MAX_N_L <= 4294967296,
      "Error, MAX_N_L or MAX_N_CELL are too large");

// TODO remove outdated 
#define PAL_SLOTS 3

#define BAS_SLOTS 8
#define BAS_OFFSET_L 1
#define BAS_OFFSET_NPGF 2
#define BAS_OFFSET_Z 5
#define BAS_OFFSET_C 6
#define BAS_OFFSET_R 7

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
#define OP_SIZE 19
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

#define L1_OFFSET 13
#define L2_OFFSET 14
#define L3_OFFSET 15
#define L4_OFFSET 16
#define L5_OFFSET 17
#define L6_OFFSET 18

#define MAX_NUM_VRR_TEAMS 16
#define MAX_NUM_QUARTETS 100000
#define MAX_NUM_BLOCKS 100000
#define MEGABYTE (1000*1000)
#define MAX_MEMORY_PER_RUN (1000*MEGABYTE)

#define COULOMB   0
#define TRUNCATED 1

#define L_BASE (32)
#define SERIAL_L4(a,b,c,d) ((((a)*L_BASE+(b))*L_BASE+(c))*L_BASE+(d))

//#define PRM_TMP_SIZE 1 
#define FVH_SIZE 18
//#define PMI_SIZE 2
#define SPH_SIZE 2
#define TRA_SIZE 6
#define KS_SIZE  8

//#define PRM_TMP_OFFSET_IPZN 0
//#define PMI_OFFSET_OF     0
//#define PMI_OFFSET_IPZN   1

#define FVH_OFFSET_OV      0
#define FVH_OFFSET_OG      1
#define FVH_OFFSET_OQ      2
#define FVH_OFFSET_NPRM    3
#define FVH_OFFSET_IDX_A   4
#define FVH_OFFSET_IDX_B   5
#define FVH_OFFSET_IDX_C   6
#define FVH_OFFSET_IDX_D   7
#define FVH_OFFSET_IDX_ZA  8
#define FVH_OFFSET_IDX_ZB  9
#define FVH_OFFSET_IDX_ZC  10
#define FVH_OFFSET_IDX_ZD  11
#define FVH_OFFSET_IDX_KA  12
#define FVH_OFFSET_IDX_KB  13
#define FVH_OFFSET_IDX_KC  14
#define FVH_OFFSET_IDX_KD  15
#define FVH_OFFSET_NLABCD  16
#define FVH_OFFSET_NPABCD  17

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

#define KS_OFFSET_IDXFAC  0
#define KS_OFFSET_INLABCD 1
#define KS_OFFSET_LDABCD  2
#define KS_OFFSET_OFFAC   3
#define KS_OFFSET_OFFAD   4
#define KS_OFFSET_OFFBC   5
#define KS_OFFSET_OFFBD   6
#define KS_OFFSET_TALL    7 

#define CELL_HMAT_OFF 0
#define CELL_HINV_OFF 9


#define POT_TRUNC_N1 128
#define POT_TRUNC_N2 128

#define CDPA_COSx  0
#define CDPA_COSy  1
#define CDPA_COSz  2
#define CDPA_DOSx  3
#define CDPA_DOSy  4
#define CDPA_DOSz  5
#define CDPA_Px    6
#define CDPA_Py    7
#define CDPA_Pz    8
#define CDPA_PmAx  9
#define CDPA_PmAy 10
#define CDPA_PmAz 11

#define CDPA_Kfac 12
#define CDPA_invz 13
#define CDPA_zc   14
#define CDPA_zd   15
#define CDPA_zab  16

#define CDPA_zcd  17
#define CDPA_rho  18
#define CDPA_R    19
#define CDPA_izab 20
#define CDPA_i2ab 21
#define CDPA_mrab 22
#define CDPA_izcd 23
#define CDPA_i2cd 24
#define CDPA_mrcd 25
#define CDPA_i2z  26

#define CDPA_SIZE 27



#include<stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_WARPSIZE 32
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

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

inline void gpuAssertCublas( cublasStatus_t code, const char *file, int line, bool abort=true){
   if( code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS error: %s %s %d\n", my_cublasGetStatusString(code), file, line);
      if (abort) exit(code);
   }
}


#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif // USE_NVTX

#endif // compile guard
