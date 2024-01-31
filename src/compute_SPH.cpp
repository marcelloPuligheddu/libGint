#include <cblas.h>
#include <vector>
#include "util.h"
#include "c2s.h"
#include "compute_SPH.h"

#include<iostream>

using std::cout;
using std::endl;

__device__ void matmulG(
      CBLAS_ORDER order, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k,
      const double* const __restrict__ A, int lda,
      const double* const __restrict__ B, int ldb,
      double* const __restrict__ C, int ldc ){

   for ( int rowcolC = threadIdx.x; rowcolC < m*n; rowcolC++ ){
      int rowC = rowcolC / n;
      int colC = rowcolC % n;
      double sum = 0.0;
      for (int ik = 0; ik < k; ik++) {
         int rowA = rowC;
         int colA = ik;
         int rowB = ik;
         int colB = colC;
         // transpose A
         int tmp = rowA;
         rowA = colA;
         colA = tmp;
         // transpose B
         tmp = rowB;
         rowB = colB;
         colB = tmp;

         sum += A[rowA*lda+colA] * B[rowB*ldb+colB];
      }
      C[rowC*ldc+colC] = sum;
   }
}


void matmul(
      CBLAS_ORDER order, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k,
      const double* const __restrict__ A, int lda,
      const double* const __restrict__ B, int ldb,
      double* const __restrict__ C, int ldc ){
   cblas_dgemm( order,ta,tb,m,n,k, 1.0, A,lda,
                                        B,ldb,
                                   0.0, C,ldc);
}

__global__ void compute_SPH_batched_gpu_low(
      const int Nqrtt, const int la, const int lb, const int lc, const int ld,
      double* const __restrict__ ABCD0,
      double* const __restrict__ SPHER,
      double* const __restrict__ tmp_scratch ){

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
    
   const double* c2sA = &c2s[ c2s_ptr[la] ];
   const double* c2sB = &c2s[ c2s_ptr[lb] ];
   const double* c2sC = &c2s[ c2s_ptr[lc] ];
   const double* c2sD = &c2s[ c2s_ptr[ld] ];
    
   for ( int block=blockIdx.x; block < Nqrtt; block+=gridDim.x ){
      double* __restrict__ cart = &ABCD0[block*Nc];
      double* __restrict__ tmp = &tmp_scratch[block*Nc];
      double* __restrict__ spher = &SPHER[block*Ns];
      int m,n,k,lda,ldb,ldc;
//      tmp = ( cart.reshape( (ncoa,ncob*ncoc*ncod) ).T @ c2sA.T )
      m = ncob*ncoc*ncod;
      n = nsoa;
      k = ncoa;
      lda = ncob*ncoc*ncod;
      ldb = ncoa;
      ldc = nsoa;
      matmulG( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, cart, lda, c2sA, ldb, tmp, ldc );
      __syncthreads();

//      (cart as tmp2) = ( tmp.reshape( (ncob,ncoc*ncod*nsoa) ).T @ c2sB.T )
      m = ncoc*ncod*nsoa;
      n = nsob;
      k = ncob;
      lda = ncoc*ncod*nsoa;
      ldb = ncob;
      ldc = nsob;
      matmulG( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, tmp, lda, c2sB, ldb, cart, ldc );
      __syncthreads();

//      tmp = ( (cart as tmp2).reshape( (ncoc,ncod*nsoa*nsob) ).T @ c2sC.T )
      m = ncod*nsoa*nsob;
      n = nsoc;
      k = ncoc;
      lda = ncod*nsoa*nsob;
      ldb = ncoc;
      ldc = nsoc;
      matmulG( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, cart, lda, c2sC, ldb, tmp, ldc );
      __syncthreads();

//      spher = ( tmp.reshape( (ncod,nsoa*nsob*nsoc) ).T @ c2sD.T )
      // need actual copy
      m = nsoa*nsob*nsoc;
      n = nsod;
      k = ncod;
      lda = nsoa*nsob*nsoc;
      ldb = ncod;
      ldc = nsod;
      matmulG( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, tmp, lda, c2sD, ldb, spher, ldc );
      __syncthreads();
   
   }
}



void compute_SPH_batched_low(
      const int Nqrtt, const int la, const int lb, const int lc, const int ld,
      double* const __restrict__ ABCD0,
      double* const __restrict__ SPHER,
      double* const __restrict__ tmp_scratch ){

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
    
   const double* c2sA = &c2s[ c2s_ptr[la] ];
   const double* c2sB = &c2s[ c2s_ptr[lb] ];
   const double* c2sC = &c2s[ c2s_ptr[lc] ];
   const double* c2sD = &c2s[ c2s_ptr[ld] ];
    
   for ( int block=0; block < Nqrtt; block++ ){
      double* __restrict__ cart = &ABCD0[block*Nc];
      double* __restrict__ tmp = &tmp_scratch[block*Nc];
      double* __restrict__ spher = &SPHER[block*Ns];

//      tmp = ( cart.reshape( (ncoa,ncob*ncoc*ncod) ).T @ c2sA.T )
      if ( la == -10 ){
         tmp = cart;
         cart = &tmp_scratch[block*Nc];
      } else {
         int m = ncob*ncoc*ncod;
         int n = nsoa;
         int k = ncoa;
         int lda = ncob*ncoc*ncod;
         int ldb = ncoa;
         int ldc = nsoa;
         matmul( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, cart, lda, c2sA, ldb, tmp, ldc );
      }

//      cart = ( tmp.reshape( (ncob,ncoc*ncod*nsoa) ).T @ c2sB.T )
      if ( lb == -10 ){
         cart = tmp;
         tmp = &tmp_scratch[block*Nc];
      } else {
         int m = ncoc*ncod*nsoa;
         int n = nsob;
         int k = ncob;
         int lda = ncoc*ncod*nsoa;
         int ldb = ncob;
         int ldc = nsob;
         matmul( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, tmp, lda, c2sB, ldb, cart, ldc );
      }

//      tmp = ( spher.reshape( (ncoc,ncod*nsoa*nsob) ).T @ c2sC.T )
      if ( lc == -10 ){
         tmp = cart;
         cart = &ABCD0[block*Nc];
      } else {
         int m = ncod*nsoa*nsob;
         int n = nsoc;
         int k = ncoc;
         int lda = ncod*nsoa*nsob;
         int ldb = ncoc;
         int ldc = nsoc;
         matmul( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, cart, lda, c2sC, ldb, tmp, ldc );
      }
//      spher = ( tmp.reshape( (ncod,nsoa*nsob*nsoc) ).T @ c2sD.T )
      // need actual copy
      spher = &SPHER[block*Ns];
      if ( ld == -10 ){
         for( int iabcd=0; iabcd < Ns; iabcd++ ){
            spher[iabcd] = tmp[iabcd];
         }
      } else {
         int m = nsoa*nsob*nsoc;
         int n = nsod;
         int k = ncod;
         int lda = nsoa*nsob*nsoc;
         int ldb = ncod;
         int ldc = nsod;
         matmul( CblasRowMajor, CblasTrans, CblasTrans, m,n,k, tmp, lda, c2sD, ldb, spher, ldc );
      }
   }
}


void compute_SPH_batched(
      const int Nqrtt, const int la, const int lb, const int lc, const int ld,
      std::vector<double>& ABCD0, std::vector<double>& SPHER, std::vector<double>& tmp_scratch ){
   compute_SPH_batched_low(Nqrtt, la, lb, lc, ld, ABCD0.data(), SPHER.data(), tmp_scratch.data() );
}







