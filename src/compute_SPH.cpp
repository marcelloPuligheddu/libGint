#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"

#include <vector>
#include "util.h"
#include "define.h"
#include "c2s.h"
#include "compute_SPH.h"
#include <hip/hip_runtime.h>
#include "hipblas.h"
#include <iostream>

using std::cout;
using std::endl;


void compute_SPH_batched_gpu_alt(
      const int Nqrtt, const int la, const int lb, const int lc, const int ld,
      double* const __restrict__ ABCD0,
      double* const __restrict__ SPHER,
      double* const __restrict__ tmp_scratch, double* const __restrict__ C2S, hipblasHandle_t cublas_handle ){

   int ncoa = compute_Nc(la);
   int ncob = compute_Nc(lb);
   int ncoc = compute_Nc(lc);
   int ncod = compute_Nc(ld);
    
   int nsoa = compute_Ns(la);
   int nsob = compute_Ns(lb);
   int nsoc = compute_Ns(lc);
   int nsod = compute_Ns(ld);
    
   int Ns = compute_Ns(la,lb,lc,ld);
   int Nc = compute_Nc(la,lb,lc,ld);
    
   double* c2sA = &C2S[ c2s_ptr[la] ];
   double* c2sB = &C2S[ c2s_ptr[lb] ];
   double* c2sC = &C2S[ c2s_ptr[lc] ];
   double* c2sD = &C2S[ c2s_ptr[ld] ];

   int m,n,k,lda,ldb,ldc,strA,strB,strC;
   const double zero = 0.0;
   const double one = 1.0;
   double *A, *B, *C;



/*
With :
int m = ncob*ncoc*ncod;
int n = nsoa;
int k = ncoa;
int lda = ncob*ncoc*ncod;
int ldb = ncoa;
int ldc = nsoa;

cblas_dgemm(
   CblasRowMajor, CblasTrans, CblasTrans, m,n,k, 
      1.0,  A=cart,  lda=m, 
            B=c2sA,  ldb=k,
      0.0,  C=tmp,   ldc=n);
            ==
cblas_dgemm (
   CblasColMajor, CblasTrans, CblasTrans, n,m,k,
      1.0,  A=c2sA,  lda=k,
            B=cart,  ldb=m,
      0.0,  C=tmp,   ldc=n)

So that:
m__ = n = nsoa
n__ = m = ncob*ncoc*ncod
k__ = k = ncoa
lda__ = ldb = ncoa = k__ = k
ldb__ = lad = ncob*ncoc*ncod = m = n__
ldc__ = ldc = nsoa = n == m__

cblas_dgemm (
   CblasColMajor, CblasTrans, CblasTrans, m__,n__,k__,
      1.0,  A=c2sA,  lda=k__,
            B=cart,  ldb=n__,
      0.0,  C=tmp,   ldc=m__)

*/


//      tmp = ( cart.reshape( (ncoa,ncob*ncoc*ncod) ).T @ c2sA.T )
      m = nsoa;
      n = ncob*ncoc*ncod;
      k = ncoa;
      lda = k;
      ldb = n;
      ldc = m;
      A = c2sA;
      B = ABCD0;
      C = tmp_scratch;
      strA = 0;
      strB = k*n;
      strC = m*n;

//      cout << endl;

//      cout << " [H T T " << m << " " << n << " " << k << "]" ;
//      cout <<  " C " << " " << C << " " << ldc << " " << strC << " = " ;
//      cout <<  " A " << " " << A << " " << lda << " " << strA << " * " ;
//      cout <<  " B " << " " << B << " " << ldb << " " << strB << " X " ;
//      cout << Nqrtt << endl;

//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

      CUBLAS_GPU_ERR_CHECK(
         hipblasDgemmStridedBatched( cublas_handle, HIPBLAS_OP_T, HIPBLAS_OP_T, m,n,k, 
         &one, A, lda, strA, B, ldb, strB, &zero, C, ldc, strC, Nqrtt ));

//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      (cart as tmp2) = ( tmp.reshape( (ncob,ncoc*ncod*nsoa) ).T @ c2sB.T )
      m = nsob;
      n = ncoc*ncod*nsoa;
      k = ncob;
      lda = k;
      ldb = n;
      ldc = m;
      A = c2sB;
      B = tmp_scratch;
      C = ABCD0;
      strA = 0;
      strB = k*n;
      strC = m*n;

//      cout << " [H T T " << m << " " << n << " " << k << "]" ;
//      cout <<  " C " << " " << C << " " << ldc << " " << strC << " = " ;
//      cout <<  " A " << " " << A << " " << lda << " " << strA << " * " ;
//      cout <<  " B " << " " << B << " " << ldb << " " << strB << " X " ;
//      cout << Nqrtt << endl;

      CUBLAS_GPU_ERR_CHECK(
         hipblasDgemmStridedBatched( cublas_handle, HIPBLAS_OP_T, HIPBLAS_OP_T, m,n,k, 
         &one, A, lda, strA, B, ldb, strB, &zero, C, ldc, strC,  Nqrtt ));
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );


//      tmp = ( (cart as tmp2).reshape( (ncoc,ncod*nsoa*nsob) ).T @ c2sC.T )
      m = nsoc;
      n = ncod*nsoa*nsob;
      k = ncoc;
      lda = k;
      ldb = n;
      ldc = m;
      A = c2sC;
      B = ABCD0;
      C = tmp_scratch;
      strA = 0;
      strB = k*n;
      strC = m*n;

//      cout << " [H T T " << m << " " << n << " " << k << "]" ;
//      cout <<  " C " << " " << C << " " << ldc << " " << strC << " = " ;
//      cout <<  " A " << " " << A << " " << lda << " " << strA << " * " ;
//      cout <<  " B " << " " << B << " " << ldb << " " << strB << " X " ;
//      cout << Nqrtt << endl;

      CUBLAS_GPU_ERR_CHECK(
         hipblasDgemmStridedBatched( cublas_handle, HIPBLAS_OP_T, HIPBLAS_OP_T, m,n,k, 
         &one, A, lda, strA, B, ldb, strB, &zero, C, ldc, strC,  Nqrtt ));
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      spher = ( tmp.reshape( (ncod,nsoa*nsob*nsoc) ).T @ c2sD.T )
      m = nsod;
      n = nsoa*nsob*nsoc;
      k = ncod;
      lda = k;
      ldb = n;
      ldc = m;
      A = c2sD;
      B = tmp_scratch;
      C = SPHER;
      strA = 0;
      strB = k*n;
      strC = m*n;

//      cout << " [H T T " << m << " " << n << " " << k << "]" ;
//      cout <<  " C " << " " << C << " " << ldc << " " << strC << " = " ;
//      cout <<  " A " << " " << A << " " << lda << " " << strA << " * " ;
//      cout <<  " B " << " " << B << " " << ldb << " " << strB << " X " ;
//      cout << Nqrtt << endl;

      CUBLAS_GPU_ERR_CHECK(
         hipblasDgemmStridedBatched( cublas_handle, HIPBLAS_OP_T, HIPBLAS_OP_T, m,n,k,
         &one, A, lda, strA, B, ldb, strB, &zero, C, ldc, strC,  Nqrtt ));
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );   
}


