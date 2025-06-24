#include "prepare_Fm_omp.h"
//#include "util.cpp"
#include "define.h"
#include <omp.h>
#include <cmath>
#include <cstdint>

#define BASE4 256
void decode4_omp( unsigned int abcd, uint8_t a[4] ){
   a[3] = abcd % BASE4;
   a[2] = abcd / BASE4 % BASE4 ;
   a[1] = abcd / (BASE4*BASE4) % BASE4 ;
   a[0] = abcd / (BASE4*BASE4*BASE4) ;
}
#undef BASE4

void decode_shell_omp(
      const unsigned int shell,
      uint8_t nl[4], 
      uint8_t np[2] ){

   static_assert( MAX_N_CELL <= 256 );
   static_assert( MAX_N_L    <= 256 );
   static_assert( (long int)MAX_N_L*MAX_N_L*MAX_N_L*MAX_N_L*MAX_N_CELL*MAX_N_CELL <= ((long int )2<<31) );

   nl[0] = (shell / (MAX_N_CELL * MAX_N_CELL * MAX_N_L * MAX_N_L * MAX_N_L));
   nl[1] = (shell / (MAX_N_CELL * MAX_N_CELL * MAX_N_L * MAX_N_L)) % MAX_N_L;
   nl[2] = (shell / (MAX_N_CELL * MAX_N_CELL * MAX_N_L)) % MAX_N_L;
   nl[3] = (shell / (MAX_N_CELL * MAX_N_CELL)) % MAX_N_L;
   np[0]  = (shell / MAX_N_CELL) % MAX_N_CELL;
   np[1]  = (shell) % MAX_N_CELL;
}

#define FM_N_VEC 4
#define FM_N_SCA 5
int Fsize_omp( int L ){ return L + 1 + FM_N_VEC * 3 + FM_N_SCA; }
#undef FM_N_VEC
#undef FM_N_SCA

//template< bool ortho >
void compute_pbc_omp( const double A[3], const double B[3], const double * const cell, double * AB ){
   // modifies AB = B - A + R such that:
   // AB is inside cell
   // R  is a lattice vector
   const double * const h_mat = &cell[CELL_HMAT_OFF];
   const double * const h_inv = &cell[CELL_HINV_OFF];

   AB[0] = B[0]-A[0];
   AB[1] = B[1]-A[1];
   AB[2] = B[2]-A[2];

//   if constexpr ( ortho ) {
//      AB[0] = AB[0] - h_mat[0*3+0]*round(h_inv[0*3+0]*AB[0]);
//      AB[1] = AB[1] - h_mat[1*3+1]*round(h_inv[1*3+1]*AB[1]);
//      AB[2] = AB[2] - h_mat[2*3+2]*round(h_inv[2*3+2]*AB[2]);
//   } else {
      // note it is a 3x3 by 3 matrix vector product
      double s0 = h_inv[0*3+0] * AB[0] + h_inv[1*3+0] * AB[1] + h_inv[2*3+0] * AB[2] ;
      double s1 = h_inv[0*3+1] * AB[0] + h_inv[1*3+1] * AB[1] + h_inv[2*3+1] * AB[2] ;
      double s2 = h_inv[0*3+2] * AB[0] + h_inv[1*3+2] * AB[1] + h_inv[2*3+2] * AB[2] ;

      s0 -= round(s0);
      s1 -= round(s1);
      s2 -= round(s2);

      // note it is a 3x3 by 3 matrix vector product
      AB[0] = h_mat[0*3+0] * s0 + h_mat[1*3+0] * s1 + h_mat[2*3+0] * s2;
      AB[1] = h_mat[0*3+1] * s0 + h_mat[1*3+1] * s1 + h_mat[2*3+1] * s2;
      AB[2] = h_mat[0*3+2] * s0 + h_mat[1*3+2] * s1 + h_mat[2*3+2] * s2;
//   }
} 

template< typename T >
inline void compute_weighted_distance_omp(
      T X12[3], const T X1[3], const T X2[3],
      const T c1, const T c2, const T c12 ){
   X12[0] = ( c1*X1[0] + c2*X2[0] ) * c12;
   X12[1] = ( c1*X1[1] + c2*X2[1] ) * c12;
   X12[2] = ( c1*X1[2] + c2*X2[2] ) * c12;
}

#include <iostream>
using std::cout;
using std::endl;


// Fills Fm with intermediate vectors
void prepare_Fm_omp_blocking(
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const int Ng ){

   #pragma omp target parallel for is_device_ptr( FVH,OF,PMX,data,Fm,cell,neighs ) depend( in:FVH,OF,PMX,data,cell,neighs ) depend( out:Fm )
   for( int p = 0 ; p < NFm ; p ++ ){

   unsigned int i    =  OF[p];
   unsigned int ipzn = PMX[p*PMX_SIZE+PMX_OFFSET_IPZN];
   uint8_t ipabcd[4];

   decode4_omp( ipzn, ipabcd );

   unsigned int idx_A  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_A];
   unsigned int idx_B  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_B];
   unsigned int idx_C  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_C];
   unsigned int idx_D  = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_D];
   unsigned int idx_za = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZA] + ipabcd[0];
   unsigned int idx_zb = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZB] + ipabcd[1];
   unsigned int idx_zc = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZC] + ipabcd[2];
   unsigned int idx_zd = FVH[i*FVH_SIZE+FVH_OFFSET_IDX_ZD] + ipabcd[3];
   unsigned int encoded_elabcd_12 = FVH[i*FVH_SIZE+FVH_OFFSET_ELABCD];

   uint8_t n12[2];
   decode_shell_omp( encoded_elabcd_12, ipabcd, n12);
   uint8_t n1 = n12[0];
   uint8_t n2 = n12[1];

   // original position of the atoms before *any* pbc is applied
   const double* Ao = &data[idx_A];
   const double* Bo = &data[idx_B];
   const double* Co = &data[idx_C];
   const double* Do = &data[idx_D];

   const double za = data[idx_za];
   const double zb = data[idx_zb];
   const double zc = data[idx_zc];
   const double zd = data[idx_zd];


   const double zab = za+zb;
   const double inv_zab = 1. / zab;
   const double zcd = zc+zd;
   const double inv_zcd = 1. / zcd;
   const double z = zab + zcd;
   const double inv_z = 1./z;

   int F_size = Fsize_omp(L);

   double A[3], B[3], C[3], D[3], Cs[3], Ds[3];
   double ABs[3], CDs[3], PQs[3];
   double P[3], Q[3], shift[3];

   compute_pbc_omp( Ao, Bo, cell, ABs );
   A[0] = Ao[0];
   A[1] = Ao[1];
   A[2] = Ao[2];
   B[0] = Ao[0] + ABs[0] + neighs[n1*3+0];
   B[1] = Ao[1] + ABs[1] + neighs[n1*3+1];
   B[2] = Ao[2] + ABs[2] + neighs[n1*3+2];
   compute_weighted_distance_omp( P, A,B,za,zb,inv_zab );

   compute_pbc_omp( Co, Do, cell, CDs );  
   C[0] = Co[0];
   C[1] = Co[1];
   C[2] = Co[2];
   D[0] = Co[0] + CDs[0] + neighs[n2*3+0];
   D[1] = Co[1] + CDs[1] + neighs[n2*3+1];
   D[2] = Co[2] + CDs[2] + neighs[n2*3+2];
   compute_weighted_distance_omp( Q, C,D,zc,zd,inv_zcd );

   compute_pbc_omp( Q, P, cell, PQs );
   shift[0] = P[0] - Q[0] - PQs[0]; 
   shift[1] = P[1] - Q[1] - PQs[1]; 
   shift[2] = P[2] - Q[2] - PQs[2];

   double normf = 2. * pow(M_PI,5./2.) * inv_zab * inv_zcd ;

   double dx = A[0]-B[0];
   double dy = A[1]-B[1];
   double dz = A[2]-B[2];
   double RAB2 = dx*dx + dy*dy + dz*dz ;
   double expon_ab = - (za*zb)/(za+zb) * RAB2;

   dx = C[0]-D[0];
   dy = C[1]-D[1];
   dz = C[2]-D[2];
   double RCD2 = dx*dx + dy*dy + dz*dz ;
   double expon_cd = - (zc*zd)/(zc+zd) * RCD2;

//   double R1 = R_pgf[ipa*npb*2+ipb*2+0] * RAB2 + R_pgf[ipa*npb*2+ipb*2+1];
//   double R2 = R_pgf[ipc*npd*2+ipd*2+0] * RCD2 + R_pgf[ipc*npd*2+ipd*2+1];
//   double R12 = R1 + R2;
//   double Zn = 1./sqrt(z)/16./M_PI/M_PI; // libcint norm
   double Zn = 1./sqrt(z); // cp2k uses the correct norm so we can use OS86 eq 44

   double Kfactor = Zn * normf * exp(expon_ab+expon_cd);

   int Of = (p * Ng + 0 ) * F_size ;

   Cs[0] = Co[0]                           + shift[0];
   Cs[1] = Co[1]                           + shift[1];
   Cs[2] = Co[2]                           + shift[2];


   Ds[0] = Co[0] + CDs[0] + neighs[n2*3+0] + shift[0];
   Ds[1] = Co[1] + CDs[1] + neighs[n2*3+1] + shift[1];
   Ds[2] = Co[2] + CDs[2] + neighs[n2*3+2] + shift[2];

   compute_weighted_distance_omp( Q, Cs,Ds,zc ,zd ,inv_zcd );

   Fm[Of+L+ 1] = P[0]-A[0];
   Fm[Of+L+ 2] = P[1]-A[1];
   Fm[Of+L+ 3] = P[2]-A[2];

   Fm[Of+L+ 4] = P[0];
   Fm[Of+L+ 5] = P[1];
   Fm[Of+L+ 6] = P[2];

   Fm[Of+L+ 7] = Q[0]-Cs[0];
   Fm[Of+L+ 8] = Q[1]-Cs[1];
   Fm[Of+L+ 9] = Q[2]-Cs[2];

   Fm[Of+L+10] = Q[0];
   Fm[Of+L+11] = Q[1];
   Fm[Of+L+12] = Q[2];

   Fm[Of+L+13] = Kfactor;
   Fm[Of+L+14] = inv_z;
//   Fm[Of+L+14] = R12;
   Fm[Of+L+15] = zc;
   Fm[Of+L+16] = zd;
   Fm[Of+L+17] = zab;

//   printf(" p %d | Ao: %lg %lg %lg    Bo: %lg %lg %lg || Co %lg %lg %lg     Do %lg %lg %lg    | n12: %d %d | \n",
//            p,     Ao[0],Ao[1],Ao[2], Bo[0],Bo[1],Bo[2], Co[0],Co[1],Co[2], Do[0],Do[1],Do[2], (int)n1, (int)n2);
//   printf(" p %d | B: %lg %lg %lg  C: %lg %lg %lg || S: %lg %lg %lg | Cs %lg %lg %lg | Kfactor %lg \n", 
//            p,     B[0],B[1],B[2], C[0],C[1],C[2],   shift[0],shift[1],shift[2], Fm[Of+L+1],Fm[Of+L+2],Fm[Of+L+3], Kfactor );
   }
}


