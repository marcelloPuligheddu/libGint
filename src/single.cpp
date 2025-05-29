#include <vector>
#include <unordered_map>
#include <iostream>
#include "UniqueArray.h"
#include <functional>

#include "hipblas/hipblas.h"

using std::cout;
using std::endl;

bool linear_search( const std::vector<double>& data, double* values, int size, int* sp ){
   // I know it can be faster. If data is (indirectly) sorted it will be even faster to rewrite it
//   cout << " Looking for " << values[0] << " of size " << size << " among " << data.size() << endl;
   for( unsigned int i=0; i < data.size(); i++ ){
      bool found = false;
      bool is_possible = true;
      for( int s=0; s<size and is_possible ; s++ ){
         if ( values[s] != data[i+s] ){
            is_possible = false;
         }
      }
      if ( is_possible ){
//         cout << "Found " << i << endl;
         found = true;
         (*sp) = i;
         return found;
      }
   }
//   cout << " not found " << endl;
   return false;
}


__inline__ size_t hash_combine( size_t lhs, size_t rhs ) {
  return lhs ^ ( rhs + 0x517cc1b727220a95 + (lhs << 6) + (lhs >> 2) );
}

size_t hash_3doubles( const double* const x ){
   size_t hash = std::hash<double>{}(x[0]);
   hash = hash_combine( hash, std::hash<double>{}(x[1]) );
   hash = hash_combine( hash, std::hash<double>{}(x[2]) );
   return hash;
}

size_t hash_doubles( const double* const x, size_t l ){
   size_t hash_x = 0;
   for( size_t i=0; i < l; i++ ){
      size_t hash_i = std::hash<double>{}(x[i]);
      hash_x = hash_combine( hash_x, hash_i );
   }
   return hash_x;
}

UniqueArray::UniqueArray(){
   internal_buffer = std::vector<double>() ;
   hm = std::unordered_map<size_t,size_t>();
   count = 0;
}

// First horrible inefficient implementation
unsigned int UniqueArray::add( const double* const __restrict__ values, const int & size ){
/*
      if ( count % 1 == 0 ){
         cout << " Add iter # " << count ;
         for( int i=0; i < size; i++ ){ cout << " " << values[i] ; }
         cout << endl;
      }
      count++;
*/
      size_t h;
      if ( size == 3 ){
         h = hash_3doubles(values);
      } else {
         h = hash_doubles(values, size);
      }
      auto search = hm.find(h);
      if ( search != hm.end() ) { return search->second; }

      unsigned int prev_size = (unsigned int) internal_buffer.size();
      hm.insert({ h, prev_size });
      internal_buffer.insert(internal_buffer.end(), values, values+size );

      return prev_size;
/*
      unsigned int prev_size = (unsigned int) internal_buffer.size();
      int sp=0;
      bool found = linear_search( internal_buffer, values, size, &sp );
      if ( found ){ return sp ; }
      
      internal_buffer.insert(internal_buffer.end(), values, values+size );
      return prev_size;
*/
}

void UniqueArray::show_ia(){
   for ( unsigned int i=0; i < internal_buffer.size(); i++ ){
      std::cout << i << " " << internal_buffer[i] << std::endl ;
   }
}



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


#ifndef BW_BY_PATCH_H_COMPILE_GUARD
#define BW_BY_PATCH_H_COMPILE_GUARD

// Table of bias and weight used in the t_c_g0.F
// in the transformations 
// TG1 = (2*X1 - BW[4p+0])*BW[4p+1]
// TG2 = (2*X2 - BW[4p+2])*BW[4p+3]
// 
double BW_by_patch[207*4] = {
0.03125000000000, 32.00000000000000, 0.01562500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.01562500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.04687500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.04687500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.01562500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.04687500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.07812500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.07812500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.10937500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.10937500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.07812500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.10937500000000, 64.00000000000000,
0.31250000000000, 16.00000000000000, 0.03125000000000, 32.00000000000000,
0.43750000000000, 16.00000000000000, 0.03125000000000, 32.00000000000000,
0.31250000000000, 16.00000000000000, 0.09375000000000, 32.00000000000000,
0.43750000000000, 16.00000000000000, 0.09375000000000, 32.00000000000000,
0.03125000000000, 32.00000000000000, 0.14062500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.14062500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.17187500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.17187500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.14062500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.17187500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.20312500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.20312500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.23437500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.23437500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.20312500000000, 64.00000000000000,
0.18750000000000, 16.00000000000000, 0.23437500000000, 64.00000000000000,
0.31250000000000, 16.00000000000000, 0.15625000000000, 32.00000000000000,
0.31250000000000, 16.00000000000000, 0.21875000000000, 32.00000000000000,
0.43750000000000, 16.00000000000000, 0.18750000000000, 16.00000000000000,
0.62500000000000, 8.00000000000000, 0.12500000000000, 8.00000000000000,
0.87500000000000, 8.00000000000000, 0.12500000000000, 8.00000000000000,
0.03125000000000, 32.00000000000000, 0.26562500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.29687500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.28125000000000, 32.00000000000000,
0.15625000000000, 32.00000000000000, 0.28125000000000, 32.00000000000000,
0.21875000000000, 32.00000000000000, 0.28125000000000, 32.00000000000000,
0.03125000000000, 32.00000000000000, 0.32812500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.35937500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.34375000000000, 32.00000000000000,
0.18750000000000, 16.00000000000000, 0.34375000000000, 32.00000000000000,
0.31250000000000, 16.00000000000000, 0.31250000000000, 16.00000000000000,
0.43750000000000, 16.00000000000000, 0.31250000000000, 16.00000000000000,
0.03125000000000, 32.00000000000000, 0.39062500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.42187500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.40625000000000, 32.00000000000000,
0.03125000000000, 32.00000000000000, 0.45312500000000, 64.00000000000000,
0.03125000000000, 32.00000000000000, 0.48437500000000, 64.00000000000000,
0.09375000000000, 32.00000000000000, 0.46875000000000, 32.00000000000000,
0.18750000000000, 16.00000000000000, 0.40625000000000, 32.00000000000000,
0.18750000000000, 16.00000000000000, 0.46875000000000, 32.00000000000000,
0.31250000000000, 16.00000000000000, 0.43750000000000, 16.00000000000000,
0.43750000000000, 16.00000000000000, 0.43750000000000, 16.00000000000000,
0.56250000000000, 16.00000000000000, 0.37500000000000, 8.00000000000000,
0.68750000000000, 16.00000000000000, 0.37500000000000, 8.00000000000000,
0.87500000000000, 8.00000000000000, 0.37500000000000, 8.00000000000000,
0.03125000000000, 32.00000000000000, 0.53125000000000, 32.00000000000000,
0.03125000000000, 32.00000000000000, 0.59375000000000, 32.00000000000000,
0.09375000000000, 32.00000000000000, 0.53125000000000, 32.00000000000000,
0.09375000000000, 32.00000000000000, 0.59375000000000, 32.00000000000000,
0.03125000000000, 32.00000000000000, 0.65625000000000, 32.00000000000000,
0.03125000000000, 32.00000000000000, 0.71875000000000, 32.00000000000000,
0.09375000000000, 32.00000000000000, 0.68750000000000, 16.00000000000000,
0.01562500000000, 64.00000000000000, 0.81250000000000, 16.00000000000000,
0.04687500000000, 64.00000000000000, 0.81250000000000, 16.00000000000000,
0.01562500000000, 64.00000000000000, 0.93750000000000, 16.00000000000000,
0.04687500000000, 64.00000000000000, 0.93750000000000, 16.00000000000000,
0.09375000000000, 32.00000000000000, 0.81250000000000, 16.00000000000000,
0.09375000000000, 32.00000000000000, 0.93750000000000, 16.00000000000000,
0.15625000000000, 32.00000000000000, 0.56250000000000, 16.00000000000000,
0.21875000000000, 32.00000000000000, 0.56250000000000, 16.00000000000000,
0.15625000000000, 32.00000000000000, 0.68750000000000, 16.00000000000000,
0.21875000000000, 32.00000000000000, 0.68750000000000, 16.00000000000000,
0.15625000000000, 32.00000000000000, 0.81250000000000, 16.00000000000000,
0.15625000000000, 32.00000000000000, 0.93750000000000, 16.00000000000000,
0.21875000000000, 32.00000000000000, 0.81250000000000, 16.00000000000000,
0.21875000000000, 32.00000000000000, 0.93750000000000, 16.00000000000000,
0.31250000000000, 16.00000000000000, 0.56250000000000, 16.00000000000000,
0.28125000000000, 32.00000000000000, 0.68750000000000, 16.00000000000000,
0.34375000000000, 32.00000000000000, 0.68750000000000, 16.00000000000000,
0.43750000000000, 16.00000000000000, 0.56250000000000, 16.00000000000000,
0.43750000000000, 16.00000000000000, 0.68750000000000, 16.00000000000000,
0.31250000000000, 16.00000000000000, 0.81250000000000, 16.00000000000000,
0.28125000000000, 32.00000000000000, 0.93750000000000, 16.00000000000000,
0.34375000000000, 32.00000000000000, 0.93750000000000, 16.00000000000000,
0.40625000000000, 32.00000000000000, 0.87500000000000, 8.00000000000000,
0.46875000000000, 32.00000000000000, 0.87500000000000, 8.00000000000000,
0.56250000000000, 16.00000000000000, 0.62500000000000, 8.00000000000000,
0.68750000000000, 16.00000000000000, 0.62500000000000, 8.00000000000000,
0.53125000000000, 32.00000000000000, 0.87500000000000, 8.00000000000000,
0.59375000000000, 32.00000000000000, 0.87500000000000, 8.00000000000000,
0.68750000000000, 16.00000000000000, 0.87500000000000, 8.00000000000000,
0.81250000000000, 16.00000000000000, 0.75000000000000, 4.00000000000000,
0.93750000000000, 16.00000000000000, 0.75000000000000, 4.00000000000000,
0.00781250000000, 128.00000000000000, 1.25000000000000, 4.00000000000000,
0.00781250000000, 128.00000000000000, 1.75000000000000, 4.00000000000000,
0.02343750000000, 128.00000000000000, 1.25000000000000, 4.00000000000000,
0.02343750000000, 128.00000000000000, 1.75000000000000, 4.00000000000000,
0.04687500000000, 64.00000000000000, 1.25000000000000, 4.00000000000000,
0.04687500000000, 64.00000000000000, 1.75000000000000, 4.00000000000000,
0.09375000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.09375000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.07812500000000, 64.00000000000000, 1.75000000000000, 4.00000000000000,
0.10937500000000, 64.00000000000000, 1.75000000000000, 4.00000000000000,
0.15625000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.21875000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.15625000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.21875000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.15625000000000, 32.00000000000000, 1.75000000000000, 4.00000000000000,
0.21875000000000, 32.00000000000000, 1.75000000000000, 4.00000000000000,
0.28125000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.34375000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.40625000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.46875000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.28125000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.34375000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.40625000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.46875000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.31250000000000, 16.00000000000000, 1.62500000000000, 8.00000000000000,
0.31250000000000, 16.00000000000000, 1.87500000000000, 8.00000000000000,
0.40625000000000, 32.00000000000000, 1.62500000000000, 8.00000000000000,
0.46875000000000, 32.00000000000000, 1.62500000000000, 8.00000000000000,
0.43750000000000, 16.00000000000000, 1.87500000000000, 8.00000000000000,
0.53125000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.59375000000000, 32.00000000000000, 1.12500000000000, 8.00000000000000,
0.53125000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.59375000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.68750000000000, 16.00000000000000, 1.12500000000000, 8.00000000000000,
0.65625000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.71875000000000, 32.00000000000000, 1.37500000000000, 8.00000000000000,
0.53125000000000, 32.00000000000000, 1.62500000000000, 8.00000000000000,
0.59375000000000, 32.00000000000000, 1.62500000000000, 8.00000000000000,
0.53125000000000, 32.00000000000000, 1.87500000000000, 8.00000000000000,
0.59375000000000, 32.00000000000000, 1.87500000000000, 8.00000000000000,
0.65625000000000, 32.00000000000000, 1.62500000000000, 8.00000000000000,
0.71875000000000, 32.00000000000000, 1.62500000000000, 8.00000000000000,
0.68750000000000, 16.00000000000000, 1.87500000000000, 8.00000000000000,
0.81250000000000, 16.00000000000000, 1.12500000000000, 8.00000000000000,
0.81250000000000, 16.00000000000000, 1.37500000000000, 8.00000000000000,
0.93750000000000, 16.00000000000000, 1.25000000000000, 4.00000000000000,
0.81250000000000, 16.00000000000000, 1.62500000000000, 8.00000000000000,
0.81250000000000, 16.00000000000000, 1.87500000000000, 8.00000000000000,
0.93750000000000, 16.00000000000000, 1.62500000000000, 8.00000000000000,
0.93750000000000, 16.00000000000000, 1.87500000000000, 8.00000000000000,
1.12500000000000, 8.00000000000000, 0.25000000000000, 4.00000000000000,
1.12500000000000, 8.00000000000000, 0.75000000000000, 4.00000000000000,
1.37500000000000, 8.00000000000000, 0.50000000000000, 2.00000000000000,
1.06250000000000, 16.00000000000000, 1.25000000000000, 4.00000000000000,
1.18750000000000, 16.00000000000000, 1.25000000000000, 4.00000000000000,
1.06250000000000, 16.00000000000000, 1.75000000000000, 4.00000000000000,
1.18750000000000, 16.00000000000000, 1.75000000000000, 4.00000000000000,
1.31250000000000, 16.00000000000000, 1.50000000000000, 2.00000000000000,
1.43750000000000, 16.00000000000000, 1.50000000000000, 2.00000000000000,
1.75000000000000, 4.00000000000000, 1.00000000000000, 1.00000000000000,
0.12500000000000, 8.00000000000000, 0.25000000000000, 4.00000000000000,
0.12500000000000, 8.00000000000000, 0.75000000000000, 4.00000000000000,
0.37500000000000, 8.00000000000000, 0.50000000000000, 2.00000000000000,
0.12500000000000, 8.00000000000000, 1.12500000000000, 8.00000000000000,
0.06250000000000, 16.00000000000000, 1.37500000000000, 8.00000000000000,
0.18750000000000, 16.00000000000000, 1.37500000000000, 8.00000000000000,
0.03125000000000, 32.00000000000000, 1.56250000000000, 16.00000000000000,
0.03125000000000, 32.00000000000000, 1.68750000000000, 16.00000000000000,
0.09375000000000, 32.00000000000000, 1.62500000000000, 8.00000000000000,
0.01562500000000, 64.00000000000000, 1.78125000000000, 32.00000000000000,
0.01562500000000, 64.00000000000000, 1.84375000000000, 32.00000000000000,
0.04687500000000, 64.00000000000000, 1.81250000000000, 16.00000000000000,
0.00781250000000, 128.00000000000000, 1.89062500000000, 64.00000000000000,
0.00781250000000, 128.00000000000000, 1.92187500000000, 64.00000000000000,
0.02343750000000, 128.00000000000000, 1.90625000000000, 32.00000000000000,
0.00781250000000, 128.00000000000000, 1.95312500000000, 64.00000000000000,
0.00781250000000, 128.00000000000000, 1.98437500000000, 64.00000000000000,
0.02343750000000, 128.00000000000000, 1.95312500000000, 64.00000000000000,
0.02343750000000, 128.00000000000000, 1.98437500000000, 64.00000000000000,
0.04687500000000, 64.00000000000000, 1.90625000000000, 32.00000000000000,
0.03906250000000, 128.00000000000000, 1.96875000000000, 32.00000000000000,
0.05468750000000, 128.00000000000000, 1.96875000000000, 32.00000000000000,
0.09375000000000, 32.00000000000000, 1.81250000000000, 16.00000000000000,
0.07812500000000, 64.00000000000000, 1.90625000000000, 32.00000000000000,
0.07812500000000, 64.00000000000000, 1.96875000000000, 32.00000000000000,
0.10937500000000, 64.00000000000000, 1.93750000000000, 16.00000000000000,
0.18750000000000, 16.00000000000000, 1.62500000000000, 8.00000000000000,
0.15625000000000, 32.00000000000000, 1.81250000000000, 16.00000000000000,
0.15625000000000, 32.00000000000000, 1.93750000000000, 16.00000000000000,
0.21875000000000, 32.00000000000000, 1.87500000000000, 8.00000000000000,
0.37500000000000, 8.00000000000000, 1.25000000000000, 4.00000000000000,
0.31250000000000, 16.00000000000000, 1.62500000000000, 8.00000000000000,
0.31250000000000, 16.00000000000000, 1.87500000000000, 8.00000000000000,
0.43750000000000, 16.00000000000000, 1.75000000000000, 4.00000000000000,
0.53125000000000, 32.00000000000000, 1.00000000000000, 1.00000000000000,
0.59375000000000, 32.00000000000000, 1.00000000000000, 1.00000000000000,
0.68750000000000, 16.00000000000000, 0.50000000000000, 2.00000000000000,
0.68750000000000, 16.00000000000000, 1.50000000000000, 2.00000000000000,
0.81250000000000, 16.00000000000000, 0.50000000000000, 2.00000000000000,
0.78125000000000, 32.00000000000000, 1.50000000000000, 2.00000000000000,
0.84375000000000, 32.00000000000000, 1.50000000000000, 2.00000000000000,
0.93750000000000, 16.00000000000000, 0.50000000000000, 2.00000000000000,
0.93750000000000, 16.00000000000000, 1.50000000000000, 2.00000000000000,
1.06250000000000, 16.00000000000000, 0.50000000000000, 2.00000000000000,
1.18750000000000, 16.00000000000000, 0.50000000000000, 2.00000000000000,
1.06250000000000, 16.00000000000000, 1.50000000000000, 2.00000000000000,
1.18750000000000, 16.00000000000000, 1.50000000000000, 2.00000000000000,
1.31250000000000, 16.00000000000000, 0.50000000000000, 2.00000000000000,
1.43750000000000, 16.00000000000000, 0.50000000000000, 2.00000000000000,
1.37500000000000, 8.00000000000000, 1.50000000000000, 2.00000000000000,
1.62500000000000, 8.00000000000000, 1.00000000000000, 1.00000000000000,
1.87500000000000, 8.00000000000000, 1.00000000000000, 1.00000000000000
};

#endif
#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
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

#ifndef C2S_H_COMPILE_GUARD
#define C2S_H_COMPILE_GUARD

#include <math.h>

#define c2s_2_2_0 -0.5/sqrt(3.0)
#define c2s_2_2_2 -0.5/sqrt(3.0)
#define c2s_2_2_5  1.0/sqrt(3.0)

// matrices for the cartesian to spherical harmonic transformations
// good for up to max angular moment of 4

static double c2s[1*1+3*3+5*6+7*10+9*15] = {
1.0, 

0.0, 1.0, 0.0, // 1.0, 0.0, 0.0, // 0. 1. 0.
0.0, 0.0, 1.0, // 0.0, 1.0, 0.0, // 0. 0. 1.
1.0, 0.0, 0.0, // 0.0, 0.0, 1.0, // 1. 0. 0.

// libcint c2s matrix for l=2
// 0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 0.0, 
// 0.0, 0.0, 0.0, 0.0, 1.0925484305920792, 0.0, 
// -0.31539156525252005, 0.0, 0.0, -0.31539156525252005, 0.0, 0.6307831305050401, 
// 0.0, 0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 
// 0.5462742152960396, 0.0, 0.0, -0.5462742152960396, 0.0, 0.0, 

// cp2k c2s matrix for l=2
 0., 1., 0., 0., 0., 0.,
 0., 0., 0., 0., 1., 0.,
 c2s_2_2_0,0., 0., c2s_2_2_2, 0., c2s_2_2_5,
 0., 0., 1., 0., 0., 0.,
 0.5, 0., 0., -0.5, 0., 0.,

// libcint c2s matrix for l=3
// 0.0, 1.7701307697799304, 0.0, 0.0, 0.0, 0.0, -0.5900435899266435, 0.0, 0.0, 0.0, 
// 0.0, 0.0, 0.0, 0.0, 2.8906114426405543, 0.0, 0.0, 0.0, 0.0, 0.0, 
// 0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0, 
// 0.0, 0.0, -1.1195289977703462, 0.0, 0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 0.7463526651802308, 
// -0.4570457994644657, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0, 0.0, 0.0, 0.0, 
// 0.0, 0.0, 1.4453057213202771, 0.0, 0.0, 0.0, 0.0, -1.4453057213202771, 0.0, 0.0, 
// 0.5900435899266435, 0.0, 0.0, -1.7701307697799304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

// cp2k c2s matrix for l=3
// xxx         xxy         xxz         xyy         xyz         xzz      yyy            yyz      yzz      zzz
 0,0.612372436,0,0,0,0,-0.204124145,0,0,0,
 0,0,0,0,1,0,0,0,0,0,
 0,-0.158113883,0,0,0,0,-0.158113883,0,0.632455532,0,
 0,0,-0.387298335,0,0,0,0,-0.387298335,0,0.25819889,
-0.158113883,0,0,-0.158113883,0,0.632455532,0,0,0,0,
 0,0,0.5,0,0,0,0,-0.5,0,0,
 0.204124145,0,0,-0.612372436,0,0,0,0,0,0,

// xxxx xxxy xxxz xxyy xxyz xxzz xyyy xyyz xyzz xzzz  yyyy yyyz yyzz yzzz zzzz
0,0.288675135,0,0,0,0,-0.288675135,0,0,0,0,0,0,0,0
,0,0,0,0,0.612372436,0,0,0,0,0,0,-0.204124145,0,0,0
,0,-0.109108945,0,0,0,0,-0.109108945,0,0.654653671,0,0,0,0,0,0
,0,0,0,0,-0.231455025,0,0,0,0,0,0,-0.231455025,0,0.3086067,0
,0.036596253,0,0,0.073192505,0,-0.292770022,0,0,0,0,0.036596253,0,-0.292770022,0,0.097590007
,0,0,-0.231455025,0,0,0,0,-0.231455025,0,0.3086067,0,0,0,0,0
,-0.054554473,0,0,0,0,0.327326835,0,0,0,0,0.054554473,0,-0.327326835,0,0
,0,0,0.204124145,0,0,0,0,-0.612372436,0,0,0,0,0,0,0
,0.072168784,0,0,-0.433012702,0,0,0,0,0,0,0.072168784,0,0,0,0

// 0.0, 2.5033429417967046, 0.0, 0.0, 0.0, 0.0, -2.5033429417967046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
// 0.0, 0.0, 0.0, 0.0, 5.310392309339791, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7701307697799304, 0.0, 0.0, 0.0, 
// 0.0, -0.94617469575756, 0.0, 0.0, 0.0, 0.0, -0.94617469575756, 0.0, 5.6770481745453605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
// 0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0, 2.676186174229157, 0.0, 
// 0.31735664074561293, 0.0, 0.0, 0.6347132814912259, 0.0, -2.5388531259649034, 0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, -2.5388531259649034, 0.0, 0.8462843753216345, 
// 0.0, 0.0, -2.0071396306718676, 0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0, 2.676186174229157, 0.0, 0.0, 0.0, 0.0, 0.0, 
// -0.47308734787878, 0.0, 0.0, 0.0, 0.0, 2.8385240872726802, 0.0, 0.0, 0.0, 0.0, 0.47308734787878, 0.0, -2.8385240872726802, 0.0, 0.0, 
// 0.0, 0.0, 1.7701307697799304, 0.0, 0.0, 0.0, 0.0, -5.310392309339791, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
// 0.6258357354491761, 0.0, 0.0, -3.755014412695057, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6258357354491761, 0.0, 0.0, 0.0, 0.0 

};


static int c2s_ptr[5] = {0, 1*1, 1*1+3*3, 1*1+3*3+5*6, 1*1+3*3+5*6+7*10};

#endif // #ifndef C2S_H_COMPILE_GUARD
#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include <iostream>
#include "compute_VRR.h"
#include "util.h"
#include "define.h"
#include <cassert>


using std::cout;
using std::endl;


// Sums the primitive along the N3 cell vector, goes from AC to AC
__global__ void compute_SFT_batched_gpu_low(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm, // unused
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD, // unused
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){

   unsigned int Nop = numVC - numV + 1;


   for( int block=blockIdx.x; block < Ncells*Nop ; block += gridDim.x ){

      unsigned int p      =  block / (Nop);
      int op              =  block % Nop + numV ;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];
      // Find the contraction we are doing
      const int t  = plan[ op*OP_SIZE + T__OFFSET ];
      if ( t != CP2S){ continue; }
      const int la = plan[ op*OP_SIZE + LA_OFFSET ];
      const int lc = plan[ op*OP_SIZE + LC_OFFSET ];
      const int off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
      const int NcoA = NLco_dev(la);
      const int NcoC = NLco_dev(lc);
      const int NcoAC = NcoA*NcoC;
      const int VBS = vrr_blocksize;

      // arguable
      constexpr int ETS = 16;
      constexpr int NET =  8;
//      const int best_eco_team_size = NcoAC ;
//      int eco_team_size = blockDim.x;
//      while ( eco_team_size > best_eco_team_size ){ eco_team_size /= 2; }

//      int num_eco_teams = blockDim.x / eco_team_size;
      int my_eco_team = threadIdx.x / ETS;
      int my_eco_rank = threadIdx.x % ETS;

      double * const pr_0 = &AC[ Ov*Ng*VBS + off_m1];
      const int pr_ld_i = Ng*VBS;
      // PR[i,0,j] = sum( PR[i,n,j] )
      // pr_ld_i = Ng*VBS
      // pr_ld_n = VBS
      for ( int idx_prm = my_eco_team ; idx_prm < n_prm ; idx_prm += NET ){
         double * pr = pr_0 + pr_ld_i*idx_prm;
         for( int j = my_eco_rank; j < NcoAC; j+= ETS ){
            double s = 0.0;
            for( int n3 = 1 ; n3 < Ng; n3++ ){ s += pr[ VBS*n3 + j ]; }
            pr[j] += s;
         }
      }
   }
}


// Transforms the NCELLS*NOPS*N1*N2*NGAUSS4 AC into the contracted NCELLS*NOP*N1*N2*NNL4 AC0
// The results are saved into ABCD, since they will be the starting points of the HHR
// Computes the 4 matrix products CC1 @ CC2 @ CC3 @ CC4 @ AC

__global__ void compute_ECO_batched_gpu_low(
      const int Ncells, const int* __restrict__ plan,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm, // unused
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){

   unsigned int Nop = numVC - numV + 1;

   for( int block=blockIdx.x; block < Ncells*Nop ; block += gridDim.x ){

      unsigned int ibk, op,Ov,Og,n_prm,nlabcd,npabcd,idx_K;
      int t,la,lc,off_m1,off_m2, NcoA,NcoC,NcoAC;

      ibk    = block / (Nop); 
      op     = block % (Nop) + numV ;
      Ov     = FVH[ibk*FVH_SIZE+FVH_OFFSET_OV];
      Og     = FVH[ibk*FVH_SIZE+FVH_OFFSET_OG];
      n_prm  = FVH[ibk*FVH_SIZE+FVH_OFFSET_NPRM];
      nlabcd = FVH[ibk*FVH_SIZE+FVH_OFFSET_NLABCD];
      npabcd = FVH[ibk*FVH_SIZE+FVH_OFFSET_NPABCD];
      idx_K  = FVH[ibk*FVH_SIZE+FVH_OFFSET_IDX_K];
      t  = plan[ op*OP_SIZE + T__OFFSET ];
      la = plan[ op*OP_SIZE + LA_OFFSET ];
      lc = plan[ op*OP_SIZE + LC_OFFSET ];
      off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
      off_m2 = plan[ op*OP_SIZE + M2_OFFSET ];
      NcoA = NLco_dev(la);
      NcoC = NLco_dev(lc);
      NcoAC = NcoA*NcoC;

      if ( t != CP2S ){ continue; }

      double * out = &ABCD[ Og*hrr_blocksize + off_m2];
      const double * const inp = &AC[Ov*Ng*vrr_blocksize + off_m1];
      const double * const Kabcd = &data[idx_K];

      constexpr int TS_l = 1;
      constexpr int TS_j = 1;
      constexpr int F1 = 8;
      constexpr int F2 = 16;
      constexpr int BS_p = 16;
//      constexpr int dim = F1*F2;
      constexpr int BS_l = F1 * TS_l;
      constexpr int BS_j = F2 * TS_j;
      constexpr int totResBlkT = BS_l * BS_j;
      constexpr int numThrBlkT = totResBlkT / (TS_l*TS_j);
      constexpr int strideK = numThrBlkT / BS_p;
      constexpr int strideI = numThrBlkT / BS_j;

      __shared__ double sK[BS_l*BS_p];
      __shared__ double sI[BS_p*BS_j];

      const int tRow = threadIdx.x / ( BS_j/TS_j ); // thr / F2
      const int tCol = threadIdx.x % ( BS_j/TS_j );
      const int iRoK = threadIdx.x / ( BS_p );
      const int iCoK = threadIdx.x % ( BS_p );
      const int iRoI = threadIdx.x / ( BS_j );
      const int iCoI = threadIdx.x % ( BS_j );

      double regL[TS_l];
      double regJ[TS_j];
      double tRes[TS_l*TS_j];

      unsigned int iB_p = iCoK;
      unsigned int og_p = 0;

//      if ( threadIdx.x == 0 and blockIdx.x % 100 == 0 ){ printf("ECO < %d %d %d > \n", nlabcd, n_prm, NcoAC ); }

      // Loop over nlabcd, then over primitives then over cartesian components
      for( unsigned int IB_l = 0 ; IB_l  < (nlabcd+BS_l-1) ; IB_l += BS_l ){
         for( unsigned int IB_p = 0 ; IB_p  < (n_prm+BS_p-1) ; IB_p += BS_p ){
            unsigned int p = iB_p + IB_p;
            if ( p < n_prm ){ og_p = PMX[(Ov+p)*PMX_SIZE+PMX_OFFSET_OGP ]; } 
            // collaborate to compute and load a BS_l * BS_p block of K in shared memory
            for ( int offK = 0 ; offK < BS_l; offK+=strideK ){
               unsigned int iB_l = iRoK + offK;
               unsigned int l = iB_l + IB_l;
               if ( p < n_prm and l < nlabcd ){
                  sK[iB_l*BS_p+iB_p ] = Kabcd[l*npabcd + og_p];
               } else {sK[iB_l*BS_p+iB_p] = 0.0;}
            }

            __syncthreads(); // __sync after writing sK

            // Uses this sK to run through the full [BS_l,ALL_J] input vector
            for ( unsigned int IB_j = 0 ; IB_j  < (NcoAC+BS_j-1) ; IB_j += BS_j ){

               // Loads a BS_p * BS_j block of the input matrix
               for ( int offI = 0 ; offI < BS_p; offI+=strideI ){
                  unsigned int iB_p = iRoI + offI;
                  unsigned int iB_j = iCoI;
                  unsigned int p = iB_p + IB_p;
                  unsigned int j = iB_j + IB_j;
                  
                  if ( p < n_prm and j < NcoAC ){
                     sI[iB_p*BS_j+iB_j] = inp[p*Ng*vrr_blocksize+j];
                  } else { sI[iB_p*BS_j+iB_j] = 0.0; }
               }

               __syncthreads(); // __sync after writing sI

               // Zeroes the register local results.
               for( int iT_lj = 0 ; iT_lj < TS_l*TS_j; iT_lj ++ ){ tRes[iT_lj] = 0.0; }
               // Sums over the block of primitives for all TS_l and TS_j
               for ( unsigned int iB_p = 0; iB_p < BS_p; iB_p++ ){
                  // sets local register caches
                  for( int iT_l = 0 ; iT_l < TS_l; iT_l ++ ){ regL[iT_l] = sK[ (tRow*TS_l+iT_l)*BS_p + iB_p             ]; }
                  for( int iT_j = 0 ; iT_j < TS_j; iT_j ++ ){ regJ[iT_j] = sI[ (iB_p          )*BS_j + (tCol*TS_j+iT_j) ]; }

                  for ( unsigned int res_l = 0 ; res_l < TS_l ; res_l++ ){
                     for ( unsigned int res_j = 0 ; res_j < TS_j ; res_j++ ){
                        tRes[res_l*TS_j+res_j] += regL[res_l] * regJ[res_j];
                     }
                  }
               }

               // Writes the results to output. Each thread is writing to a different location
               // No block can write to the same braket (for better or worse)
               // So no atomic is necessary
               for ( unsigned int res_l = 0 ; res_l < TS_l ; res_l++ ){
                  for ( unsigned int res_j = 0 ; res_j < TS_j ; res_j++ ){
                     unsigned int l = tRow * TS_l + res_l + IB_l;
                     unsigned int j = tCol * TS_j + res_j + IB_j;
                     if ( l < nlabcd and j < NcoAC ){
                        out[l*hrr_blocksize+j] += tRes[res_l*TS_j+res_j];
                     }
                  }
               }
               __syncthreads(); // __sync after using sI
            }
            __syncthreads(); // sync after using sK
         }
      } // end of strange gemm
   }
}


#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include <cmath>
#include <omp.h>
#include <vector>
#include "define.h"
#include "util.h"
#include "fgamma.h"
#include "t_c_g0_n.h"
#include "compute_Fm.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <iostream>

#define SQRT2 1.4142135623730950488016887242096980785696718753
#define BLKIDX 1

using std::cout;
using std::endl;

// Fills Fm with intermediate vectors
__global__ void prepare_Fm_batched_gpu_low_private(
      const unsigned int* const __restrict__ FVH,
      const unsigned int* const __restrict__ OF,
      const unsigned int* const __restrict__ PMX,
      const double* const __restrict__ data,
      double* const __restrict__ Fm,
      int NFm, int L, bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      const int Ng ){

   for( int p = threadIdx.x + blockIdx.x*blockDim.x ; p < NFm ; p += blockDim.x*gridDim.x ){

   unsigned int i    =  OF[p];
   unsigned int ipzn = PMX[p*PMX_SIZE+PMX_OFFSET_IPZN];
   uint8_t ipabcd[4];

   decode4( ipzn, ipabcd );

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
   decode_shell( encoded_elabcd_12, ipabcd, n12);
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

   int F_size = Fsize(L);

   double A[3], B[3], C[3], D[3], Cs[3], Ds[3];
   double ABs[3], CDs[3], PQs[3];
   double P[3], Q[3], shift[3];

   compute_pbc( Ao, Bo, cell, ABs );
   A[0] = Ao[0];
   A[1] = Ao[1];
   A[2] = Ao[2];
   B[0] = Ao[0] + ABs[0] + neighs[n1*3+0];
   B[1] = Ao[1] + ABs[1] + neighs[n1*3+1];
   B[2] = Ao[2] + ABs[2] + neighs[n1*3+2];
   compute_weighted_distance( P, A,B,za,zb,inv_zab );

   compute_pbc( Co, Do, cell, CDs );  
   C[0] = Co[0];
   C[1] = Co[1];
   C[2] = Co[2];
   D[0] = Co[0] + CDs[0] + neighs[n2*3+0];
   D[1] = Co[1] + CDs[1] + neighs[n2*3+1];
   D[2] = Co[2] + CDs[2] + neighs[n2*3+2];
   compute_weighted_distance( Q, C,D,zc,zd,inv_zcd );

   compute_pbc( Q, P, cell, PQs );
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

   compute_weighted_distance( Q, Cs,Ds,zc ,zd ,inv_zcd );

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

// Computes T and the foundamental ssss(m) integrals
__global__ void compute_Fm_batched_gpu_low_private( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ bias_and_weight_by_patch,   
      int potential_type, const int Ng ){

   int F_size = Fsize(L);

   int p = blockIdx.x;

   const int Of0 = (p * Ng + 0 ) * F_size ;

   double P[3], Q[3], PQ[3];
   double rpq2, T;
   double Kfac,invz,zc,zd,zab,zcd,rho,R;

//   PA[0] = Fm[Of0+L+1];
//   PA[1] = Fm[Of0+L+2];
//   PA[2] = Fm[Of0+L+3];
   P[0] = Fm[Of0+L+4];
   P[1] = Fm[Of0+L+5];
   P[2] = Fm[Of0+L+6];
//   QC[0] = Fm[Of0+L+7];
//   QC[1] = Fm[Of0+L+8];
//   QC[2] = Fm[Of0+L+9];
   Q[0] =  Fm[Of0+L+10];
   Q[1] =  Fm[Of0+L+11];
   Q[2] =  Fm[Of0+L+12];

   Kfac = Fm[Of0+L+13];
   invz = Fm[Of0+L+14];
   zc   = Fm[Of0+L+15];
   zd   = Fm[Of0+L+16];
   zab  = Fm[Of0+L+17];

   zcd  = zc + zd;
   rho  = zab * zcd * invz;
   R    = R_cut * sqrt(rho);

   for ( int n3 = threadIdx.x; n3 < Ng ; n3 += blockDim.x ){

      int Of = Of0 + n3 * F_size ;
      PQ[0] = P[0]-Q[0]-neighs[n3*3+0];
      PQ[1] = P[1]-Q[1]-neighs[n3*3+1];
      PQ[2] = P[2]-Q[2]-neighs[n3*3+2];
      rpq2 = (PQ[0]*PQ[0] + PQ[1]*PQ[1] + PQ[2]*PQ[2]);
      T = rho * rpq2 ;
//      printf("%d.%d %d.%d TR: %lg %lg %lg \n", blockIdx.x, threadIdx.x, p, n3, T, R, Kfac );
//      for( int m=0; m<L;m++){ Fm[Of+m] = T * Kfac + R; }
//         switch ( potential_type ){
//            case COULOMB :
//               fgamma0( L, T, &Fm[Of], ftable, ftable_ld );
//            break;
//            case TRUNCATED :

               bool use_gamma = t_c_g0_n_v2(
                  &Fm[Of], R, T, L, C0, ld_C0,
                  POT_TRUNC_N1, POT_TRUNC_N2,
                  x12_to_patch_low_R, x12_to_patch_high_R, bias_and_weight_by_patch, 0, Kfac );
               if (use_gamma) { fgamma0( L, T, &Fm[Of], ftable, ftable_ld, Kfac ); }

//            break;
//         } // end switch potential_type
   }
   __syncthreads();
}

// Fills Fm with the vectors and values required by VVR
__global__ void compute_Vm_batched_gpu_low_private( 
      double* __restrict__ Fm,
      int NFm, int L, bool periodic,
      double* __restrict__ neighs,
      double* __restrict__ ftable, int ftable_ld,
      const double R_cut, const double * const __restrict__ C0, const int ld_C0,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ bias_and_weight_by_patch,   
      int potential_type, const int Ng ){

   int F_size = Fsize(L);
   int p = blockIdx.x;

   const int Of0 = (p * Ng + 0 ) * F_size ;

   double P[3], QC[3], Q[3], W[3], PA[3], Qp[3];
   double zc,zd,zcd,zab,invz,rho,izab,i2ab,mrab,izcd,i2cd,mrcd,i2z;

   PA[0] = Fm[Of0+L+1];
   PA[1] = Fm[Of0+L+2];
   PA[2] = Fm[Of0+L+3];
   P[0] = Fm[Of0+L+4];
   P[1] = Fm[Of0+L+5];
   P[2] = Fm[Of0+L+6];
   QC[0] = Fm[Of0+L+7];
   QC[1] = Fm[Of0+L+8];
   QC[2] = Fm[Of0+L+9];
   Q[0] =  Fm[Of0+L+10];
   Q[1] =  Fm[Of0+L+11];
   Q[2] =  Fm[Of0+L+12];

//////////
   invz = Fm[Of0+L+14];
   zc   = Fm[Of0+L+15];
   zd   = Fm[Of0+L+16];
   zab  = Fm[Of0+L+17];

   __syncthreads(); // sync before Fm[Of0] is wrote over by tid 0

   zcd  = zc + zd;
   rho  = zab * zcd * invz;

   izab = 1. / zab;
   i2ab = 0.5 * izab;
   mrab =  -1. * rho * izab * i2ab;
   izcd = 1. / zcd;
   i2cd = 0.5 * izcd;
   mrcd =  -1. * rho * izcd * i2cd;
   i2z  = 0.5 * invz;

   for ( int n3 = threadIdx.x; n3 < Ng ; n3 += blockDim.x ){

      int Of = Of0 + n3 * F_size ;

      Qp[0] = Q[0] + neighs[n3*3+0];
      Qp[1] = Q[1] + neighs[n3*3+1];
      Qp[2] = Q[2] + neighs[n3*3+2];

      compute_weighted_distance( W, P, Qp, zab,zcd,invz );
      if ( Fm[Of+0] > 0.0 ){

         Fm[Of+L+ 1] = PA[0];
         Fm[Of+L+ 2] = PA[1];
         Fm[Of+L+ 3] = PA[2];
         Fm[Of+L+ 4] = W[0]-P[0];
         Fm[Of+L+ 5] = W[1]-P[1];
         Fm[Of+L+ 6] = W[2]-P[2];
         Fm[Of+L+ 7] = QC[0];
         Fm[Of+L+ 8] = QC[1];
         Fm[Of+L+ 9] = QC[2];
         Fm[Of+L+10] = W[0]-Qp[0];
         Fm[Of+L+11] = W[1]-Qp[1];
         Fm[Of+L+12] = W[2]-Qp[2];
         Fm[Of+L+13] = i2ab;
         Fm[Of+L+14] = mrab;
         Fm[Of+L+15] = i2cd;
         Fm[Of+L+16] = mrcd;
         Fm[Of+L+17] = i2z ;
      }
//      if ( L == 1 and blockIdx.x == 1024){
//         printf("%d.%d %d.%d TR: %lg %lg %lg \n", blockIdx.x, threadIdx.x, p, n3, i2ab, mrab, i2z );
//      }

   }
}

#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include <cassert>
#include <vector>
#include "util.h"
#include "define.h"


#define BASE4 256
unsigned int encode4( int a, int b, int c, int d ){
   return a*BASE4*BASE4*BASE4 + b*BASE4*BASE4 + c*BASE4 + d;
}
__device__ __host__ void decode4(
      unsigned int abcd, unsigned int* a, unsigned int* b,
      unsigned int* c, unsigned int * d ){
   (*d) = abcd % BASE4;
   (*c) = abcd / BASE4 % BASE4 ;
   (*b) = abcd / (BASE4*BASE4) % BASE4 ;
   (*a) = abcd / (BASE4*BASE4*BASE4) ;
}
__device__ __host__ void decode4( unsigned int abcd, uint8_t a[4] ){
   a[3] = abcd % BASE4;
   a[2] = abcd / BASE4 % BASE4 ;
   a[1] = abcd / (BASE4*BASE4) % BASE4 ;
   a[0] = abcd / (BASE4*BASE4*BASE4) ;
}
#undef BASE4

#define FM_N_VEC 4
#define FM_N_SCA 5
__device__ __host__ int Fsize( int L ){ return L + 1 + FM_N_VEC * 3 + FM_N_SCA; }
#undef FM_N_VEC
#undef FM_N_SCA


unsigned int encodeL( int la, int lb, int lc, int ld ){
   return la * NL3 + lb * NL2 + lc * NL + ld;
}

__device__ __host__ void decodeL( unsigned int L, int* la, int* lb, int* lc, int* ld ){
   (*ld) = L % NL;
   (*lc) = L / NL % NL ;
   (*lb) = L / NL2 % NL ;
   (*la) = L / NL3 ;
}

unsigned int encode_shell( const int nla, const int nlb, const int nlc, const int nld, const int n1, const int n2 ){
    assert(nla >= 0);
    assert(nla < MAX_N_L);
    assert(nlb >= 0);
    assert(nlb < MAX_N_L);
    assert(nlc >= 0);
    assert(nlc < MAX_N_L);
    assert(nld >= 0);
    assert(nld < MAX_N_L);
    assert(n1 >= 0);
    assert(n1 < MAX_N_CELL);
    assert(n2 >= 0);
    assert(n2 < MAX_N_CELL);

    unsigned int ret = 0;
    ret +=  n2;
    ret +=  n1 * MAX_N_CELL;
    ret += nld * MAX_N_CELL * MAX_N_CELL;
    ret += nlc * MAX_N_CELL * MAX_N_CELL * MAX_N_L;
    ret += nlb * MAX_N_CELL * MAX_N_CELL * MAX_N_L * MAX_N_L;
    ret += nla * MAX_N_CELL * MAX_N_CELL * MAX_N_L * MAX_N_L * MAX_N_L;
    return ret;
}

__host__ __device__ void decode_shell(
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


__host__ __device__ void decode_shell(
      const unsigned int shell,
      unsigned int* __restrict__ nla, unsigned int* __restrict__ nlb,
      unsigned int* __restrict__ nlc, unsigned int* __restrict__ nld,
      unsigned int* __restrict__ n1,  unsigned int* __restrict__ n2 ){
   (*nla) = (shell / (MAX_N_CELL * MAX_N_CELL * MAX_N_L * MAX_N_L * MAX_N_L));
   (*nlb) = (shell / (MAX_N_CELL * MAX_N_CELL * MAX_N_L * MAX_N_L)) % MAX_N_L;
   (*nlc) = (shell / (MAX_N_CELL * MAX_N_CELL * MAX_N_L)) % MAX_N_L;
   (*nld) = (shell / (MAX_N_CELL * MAX_N_CELL)) % MAX_N_L;
   (*n1)  = (shell / MAX_N_CELL) % MAX_N_CELL;
   (*n2)  = (shell) % MAX_N_CELL;
}

int max( std::vector<int> x ){
   if ( x.size() == 0 ){ return 0; };
   int ret = x[0];
   for( unsigned int idx=1; idx<x.size(); idx++ ){ ret = max(ret, x[idx]); }
   return ret;
}


__device__ __host__ double anint( double x ){

   if ( x == 0.5 ) { return 1.0; }
   if ( x == -0.5 ){ return -1.0;}
   return round(x);

}

__device__ __host__ double my_round( double x ){
   return x - floor(x) == 0.5 ? trunc(x) : round(x);
}



__device__ __host__ double my_wrap( double s ){
//   if ( s >  0.5 - EPS_ROUNDING and s < 0.5 + EPS_ROUNDING ){ return s; }
//   if ( s < -0.5 + EPS_ROUNDING and s >-0.5 - EPS_ROUNDING ){ return s; }
   return s - round(s);
}

//template< bool ortho >
__device__ __host__ void compute_pbc( const double A[3], const double B[3], const double * const cell, double * AB ){
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


// #### device L ####

__constant__ int _NLco_lut_dev[35] = { 0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561 };
__device__ int NLco_dev( int L ){ return _NLco_lut_dev[L+2]; }

// essentialy is using the pattern:
// s = 0 0 0
// p = 1 0 0, 0 1 0, 0 0 1
// d = 2 0 0, 1 1 0, 1 0 1, 0 2 0, 0 1 1, 0 0 2
// and noting that L-lx does not really depend on L
__constant__ short int lx_lut_dev[45] = { 0, 1,1, 2,2,2, 3,3,3,3, 4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8,8 };

// compute (cartesian) moment on x axis for a given total moment.
__device__ int lx_dev( const int i, const int L ){
   return L - lx_lut_dev[i];
}

// 
__device__ int lz_dev( const int i, const int L ){
   int i0 = NLco_dev(lx_lut_dev[i]-1);
   int lz_ = i - i0;
   return lz_;
}

// computes ly as L-lx-lz
__device__ int ly_dev( const int i, const int L ){
   int lx_ = lx_dev(i,L);
   int i0 = NLco_dev(lx_lut_dev[i]-1);
   int lz_ = i - i0;
   int ly_ = L-lx_-lz_;
   return ly_;
}


// #### host L ####

int NLco( int L ){
   const int _NLco_lut[35] = { 0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561 };
   return _NLco_lut[L+2];
}

int L_lx(const int i ){
   const short int lx_lut[45] = { 0, 1,1, 2,2,2, 3,3,3,3, 4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8,8 };
   return lx_lut[i];
}

int lx( const int i, const int L ){
   return L - L_lx(i);
}

int lz( const int i, const int L ){
   int i0 = NLco(L_lx(i)-1);
   int lz_ = i - i0;
   return lz_;
}

// computes ly as L-lx-lz
int ly( const int i, const int L ){
   int i0 = NLco(L_lx(i)-1);
   int lz_ = i - i0;
   int ly_ = L_lx(i)-lz_;
   return ly_;
}

// #### so far, both __device__ and __host__ are ok ####

__device__ __host__ int compute_Nc( int la, int lb, int lc, int ld ){
   return (la+1)*(la+2) * (lb+1)*(lb+2) * (lc+1)*(lc+2) * (ld+1)*(ld+2) / 16 ;
}


__device__ __host__ int compute_Ns( int la, int lb, int lc, int ld ){
   return (2*la+1) * (2*lb+1) * (2*lc+1) * (2*ld+1) ;
}


//__host__ __device__ void compute_weighted_distance(
//      double X12[3], const double X1[3], const double X2[3],
//      const double c1, const double c2, const double c12 ){
//}

#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>

#include "t_c_g0_n.h"

//#include "BW_by_patch.h"

#include "locate_patch_low_R.cpp"
#include "locate_patch_high_R.cpp"

#define SQRT2 1.4142135623730950488016887242096980785696718753
#define DEGREE 13
#define target_error 0.100000E-08
#define NDERIV_MAX 21

/*
double* read_c0( int Nder, FILE* stream, int* ldc0_ptr );
__device__ __host__ void PD2VAL( double* res, int Nder, double TG1, double TG2, const double* C0_row );
__device__ __host__ bool t_c_g0_n( double* res, double R, double T, int Nder, const double* C0, int ldc0 );
__host__ void fill_x12_to_patch( const int N1, const int N2, int * x12_to_patch_low_R, int * x12_to_patch_high_R );
__device__ __host__ bool t_c_g0_n_v2(
      double* res, double R, double T, int Nder, const double* C0, int ld_C0, int N1, int N2, 
      const   int*  const __restrict__ x12_to_patch_low_R, const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ BW_by_patch );
*/

double* read_c0( int Nder, FILE* stream, int* ldc0_ptr ){
   const int patches = 207;
   assert( Nder <= NDERIV_MAX );
//   int nderiv_init = Nder;
   int size1 = ((Nder       + 1)*(DEGREE + 1)*(DEGREE + 2))/2;
   int size2 = ((NDERIV_MAX + 1)*(DEGREE + 1)*(DEGREE + 2))/2;
   int ldc0 = size1;
   double* C0_ptr = (double*)malloc( patches*size1*sizeof(double) );
   for( int i=0; i < patches; i++ ){
      for ( int j=0; j < size2 ; j++ ){
         double temp;
         int nscanned = fscanf( stream , "%lE", &temp);
         assert ( nscanned == 1 );
         if ( j < ldc0 ){
            C0_ptr[i*ldc0+j] = temp;
         }
//         printf("%d %d %d %d %lE \n", i, j, i*size2+j, i*ldc0+j, C0_ptr[i*ldc0+j] );
      }
   }
//   printf("Done reading\n");
   (*ldc0_ptr) = ldc0;
   return C0_ptr;
}

__host__ void fill_x12_to_patch(
      const int N1, const int N2, 
      int * x12_to_patch_low_R, int * x12_to_patch_high_R ){
   for ( int i1 = 0 ; i1 < N1 ; i1 ++ ){
      double X1 = ((double)i1)/double(N1-1.0);
      for ( int i2 = 0 ; i2 < N2 ; i2 ++ ){
         double X2 = ((double)i2)/double(N2-1.0);
         x12_to_patch_low_R[ i1*N2+i2] = locate_patch_low_R ( X1,X2 );
         x12_to_patch_high_R[i1*N2+i2] = locate_patch_high_R( X1,X2 );
      }
   }
}


__device__ __host__ bool t_c_g0_n_v2( 
      double* res, double R, double T, int Nder, const double* C0, int ld_C0, 
      int N1, int N2,
      const   int*  const __restrict__ x12_to_patch_low_R, 
      const   int*  const __restrict__ x12_to_patch_high_R, 
      const double* const __restrict__ BW_by_patch,
      int iw, const double Kfac ){

//   printf(" computing tcg R: %lg T: %lg ", R, T );
//   bool use_gamma = false;

   double upper = R*R + 11.0*R + 50.0;
   double lower = R*R - 11.0*R +  0.0;
   double X1, X2;
   int patch = 255;

   if (T > upper) { 
      for ( int n = 0; n <= Nder ; n++ ){
         res[n] = 0.0;
      }
      return false;
   }

   if (R <= 11.0) { 
      X2 = R/11.0;
      upper = R*R + 11.0*R + 50.0;
      lower = 0.0;
      X1 = (T - lower)/(upper - lower);

      int i1 = (X1 * N1);
      int i2 = (X2 * N2);
      
      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

//      assert( i1 >= 0 );
//      assert( i1 < N1 );
//      assert( i2 >= 0 );
//      assert( i2 < N2 );

      patch = x12_to_patch_low_R[ i1*N2+i2 ];
   } else {
      if ( T < lower ) {
         // why even zero? Res is going to get overwritten by gamma
         for ( int n = 0; n <= Nder ; n++ ){
            res[n] = 0.0;
         }
         return true;
      }
      X2 = 11.0/R;
      X1 = (T-lower)/(upper-lower);
      
      int i1 = (X1 * N1);
      int i2 = (X2 * N2);

      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

//      assert( i1 >= 0 );
//      assert( i1 < N1 );
//      assert( i2 >= 0 );
//      assert( i2 < N2 );

      patch = x12_to_patch_high_R[ i1*N2+i2 ];
   }

   const double B1 = BW_by_patch[ patch*4 + 0 ];
   const double W1 = BW_by_patch[ patch*4 + 1 ];
   const double B2 = BW_by_patch[ patch*4 + 2 ];
   const double W2 = BW_by_patch[ patch*4 + 3 ];

//   if ( iw > 0 ){
//      printf(" Computing tcg @ %lg %lg -> X12p = %lg %lg %d | BW = %lg %lg %lg %lg \n", R,T,X1,X2,patch,B1,W1,B2,W2 );
//   }
   
   const double * const C0_row = &C0[ld_C0*patch];

   double TG1 = (2.*X1-B1)*W1;
   double TG2 = (2.*X2-B2)*W2;

   double T1[14];
   double T2[14];
   T1[0] = 1.0;
   T2[0] = 1.0;
   T1[1] = SQRT2*TG1;
   T2[1] = SQRT2*TG2;
   T1[2] = 2.*TG1*T1[1] - SQRT2;
   T2[2] = 2.*TG2*T2[1] - SQRT2;
   for ( int i=3; i < 14; i++ ) {
      // NOTE: this is the recurrence relation for Chebishev polynomial of the first kind
      T1[i] = 2.*TG1*T1[i-1] - T1[i-2];
      T2[i] = 2.*TG2*T2[i-1] - T2[i-2];
   }

   // NOTE: this horror has the structure v1(k) @ L(k) @ v2(k).T[::-1]
   // where v1 and v2 are vector and L is a (flattened) Triangular matrix
   for ( int k=0; k <= Nder; k++ ){
      double tmp = 0.0;
      int jl = 0; // unlike l, jl does not get reset after the l loop
      for ( int j=0; j < 14; j++ ){
         double dot = 0.0;
         for ( int l=0; l < 14-j; l++){
            dot += T1[l] * C0_row[k*105+jl];
//            printf(" T1[l]: %lg C0_kl : %lg || k j l jl kjl %d %d %d %d %d || \n", T1[l], C0_row[k*105+jl], k, j, l, jl, k*105+jl );
            jl++;
         }
//         printf(" T2[j]: %lg dot : %lg || j k %d %d || \n", T2[j], dot, j, k );
         tmp += dot * T2[j];

      }
      res[k] = tmp*Kfac;
   }
   return false;
}


/*
__device__ bool t_c_g0_n_v3(
      double* res, double R, double T, int Nder, const double* C0, int ld_C0,
      int N1, int N2,
      const   int*  const __restrict__ x12_to_patch_low_R,
      const   int*  const __restrict__ x12_to_patch_high_R,
      const double* const __restrict__ BW_by_patch,
      int iw=0 ){

   constexpr int NFT =   2;
   constexpr int SFT =  32;
   constexpr int NPT =   1;

   assert( NFT*SFT/NPT == blockDim.x );

   int my_fm_rank = threadIdx.x % SFT;
   int my_fm_team = threadIdx.x / SFT;
 
   double upper = R*R + 11.0*R + 50.0;
   double lower = R*R - 11.0*R +  0.0;
   double X1, X2;
   int patch = 255;

//   printf(" %d.%d v3 %lg %lg \n", blockIdx.x, threadIdx.x, T,R );

   if (T > upper) {
      for ( int n = 0; n <= Nder ; n++ ){
         res[n] = 0.0;
      }
      return false;
   }

   if (R <= 11.0) {
      X2 = R/11.0;
      upper = R*R + 11.0*R + 50.0;
      lower = 0.0;
      X1 = (T - lower)/(upper - lower);

      int i1 = (X1 * N1);
      int i2 = (X2 * N2);

      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

      patch = x12_to_patch_low_R[ i1*N2+i2 ];
   } else {
      if ( T < lower ) { // if R > 11 and T < R2 - 11 R use gamma
         // why even zero? Res is going to get overwritten by gamma
         for ( int n = 0; n <= Nder ; n++ ){
            res[n] = 0.0;
         }
         return true;
      }
      X2 = 11.0/R;
      X1 = (T-lower)/(upper-lower);

      int i1 = (X1 * N1);
      int i2 = (X2 * N2);

      if ( i1 == N1 ){ i1 = N1-1; }
      if ( i2 == N2 ){ i2 = N2-1; }

      patch = x12_to_patch_high_R[ i1*N2+i2 ];
   }

   const double B1 = BW_by_patch[ patch*4 + 0 ];
   const double W1 = BW_by_patch[ patch*4 + 1 ];
   const double B2 = BW_by_patch[ patch*4 + 2 ];
   const double W2 = BW_by_patch[ patch*4 + 3 ];

   double TG1 = (2.*X1-B1)*W1;
   double TG2 = (2.*X2-B2)*W2;

//   printf(" Computing tcg @ %lg %lg -> X12p = %lg %lg %d | BW = %lg %lg %lg %lg \n", T,R, X1,X2,patch, B1,W1,B2,W2 );

   double T1[16];
   double T2[16];

   constexpr int s_ld = SFT+8;

   __shared__ double s_tmp[NFT*s_ld];
   __shared__ double s_dot_jt[NFT*s_ld]; // TODO reuse s_tmp (?)
   __shared__ double dot[NFT*SFT]; // TODO not shared (?)

//   unsigned int tid = threadIdx.x;

   T1[0] = 1.0;
   T2[0] = 1.0;
   T1[1] = SQRT2*TG1;
   T2[1] = SQRT2*TG2;
   T1[2] = 2.*TG1*T1[1] - SQRT2;
   T2[2] = 2.*TG2*T2[1] - SQRT2;
   for ( int i=3; i < 14; i++ ) {
      // NOTE: this is the recurrence relation for Chebishev polynomial of the first kind
      T1[i] = 2.*TG1*T1[i-1] - T1[i-2];
      T2[i] = 2.*TG2*T2[i-1] - T2[i-2];
   }
   T1[14] = 0.0;
   T2[14] = 0.0;
   T1[15] = 0.0;
   T2[15] = 0.0;


   // NOTE: this horror has the structure v1(k) @ L(k) @ v2(k).T[::-1]
   // where v1 and v2 are vector and L is a (flattened) Triangular matrix

   // Zero the extra 8 doubles at the end of the shared memory assigned to this team
   s_dot_jt[my_fm_team*s_ld+my_fm_rank+8] = 0.0;
   s_tmp[my_fm_team*s_ld+my_fm_rank+8] = 0.0;

   for ( int k=0; k <= Nder; k++ ){
      int jl = 0;
      for ( int j=0; j < 14; j++ ){
         // Step 1: load C and multiply by T1 into shared memory
         // TODO: reshape C0 into 16*16 square matrix
         // NOTE: each thread will only ever use T1[my_fm_rank]
         // TODO: compute multiple T1 and T2 for different n3 in the same team and share
         int l = my_fm_rank;
         if ( l < 14-j ) {
            s_dot_jt[my_fm_team*s_ld+l] = C0[ld_C0*patch + k*105 + jl + l] * T1[l];
//            printf(" %d.%d.%d | C0.T1 = %lg %lg %lg \n", blockIdx.x, threadIdx.x, j, C0[ld_C0*patch + k*105 + jl + l] * T1[l], C0[ld_C0*patch + k*105 + jl + l], T1[l] );
         } else {
            s_dot_jt[my_fm_team*s_ld+l] = 0.0;
         }
         __syncwarp();

         // Step 2: sum over l to compute dot[j]
         const unsigned int sid = my_fm_team*s_ld+l;
         s_dot_jt[sid] += s_dot_jt[sid + 8];__syncwarp();
         s_dot_jt[sid] += s_dot_jt[sid + 4];__syncwarp();
         s_dot_jt[sid] += s_dot_jt[sid + 2];__syncwarp();
         s_dot_jt[sid] += s_dot_jt[sid + 1];__syncwarp();

         // s_dot_jt[0] now contains the sum of C0 * T1 for this j and this idx_t
         dot[my_fm_team*16+j] = s_dot_jt[my_fm_team*s_ld+0];
//         printf(" %d.%d.%d | C0@T1 = %lg \n", blockIdx.x, threadIdx.x, j, dot[my_fm_team*16+j] );
         __syncwarp();
         jl += 14-j;
      }
      // Zero extra term in dot. Dot is the size 14 vector product of C(14x14,Triangular) and T1(14)
      // TODO: Recast as a (16x16,Square) @ 16 product
      dot[my_fm_team*16+14] = 0.0;
      dot[my_fm_team*16+15] = 0.0;
      // Now we have all the C0 @ T1 for all j for this given idx_t and k
      int j = my_fm_rank;
      const unsigned int sid = my_fm_team*s_ld+j;
      // Step 3: multiply by T2 and sum over j
      s_tmp[my_fm_team*s_ld+j] = dot[my_fm_team*16+j] * T2[j];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 8];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 4];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 2];__syncwarp();
      s_tmp[sid] += s_tmp[sid + 1];__syncwarp();

      if( my_fm_rank == 0 ) {
         res[k] = s_tmp[my_fm_team*s_ld+0];
//         printf(" %d.%d | R %lg \n", blockIdx.x, threadIdx.x, res[k] );
      }
      __syncwarp();
   }
   return false;
}
*/


#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
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


#include "fgamma.h"

#define Teps 1.e-13
#define NDERIV_MAX 21
#define DEGREE 13

/*
double* read_c0( int Nder, FILE* stream, int* ldc0_ptr ){
   const int patches = 207;
   assert( Nder <= NDERIV_MAX );
   int size1 = ((Nder       + 1)*(DEGREE + 1)*(DEGREE + 2))/2;
   int size2 = ((NDERIV_MAX + 1)*(DEGREE + 1)*(DEGREE + 2))/2;
   int ldc0 = size1;
   double* C0_ptr = (double*)malloc( patches*size1*sizeof(double) );
   for( int i=0; i < patches; i++ ){
      for ( int j=0; j < size2 ; j++ ){
         double temp;
         int nscanned = fscanf( stream , "%lE", &temp);
         assert ( nscanned == 1 );
         if ( j < ldc0 ){
            C0_ptr[i*ldc0+j] = temp;
         }
      }
   }
   (*ldc0_ptr) = ldc0;
   return C0_ptr;
}
*/

void fgamma_ref( int nmax , double T, double* f){
   const int kmax = 50;
   double r[kmax+10];

   if (T < Teps) {
      for( int n = 0 ; n <= nmax ; n++ ){
         f[n] = 1.0/(2.*n + 1.);
      }
   } else if (T <= 50.0) {
      // Initialize ratios of Bessel functions
      r[kmax + 9] = 0.0;
      for ( int j = kmax+9; j >= 1; j--){
         r[j-1] = -T/((4.*j + 2.) - T*r[j+1-1]);
      }
      double factor = 2.0*sinh(0.5*T)*exp(-0.5*T)/T;
      for( int n = 0; n <= nmax ; n++ ){
      // Initialize iteration
         double sumtot = factor/(2.*n + 1.);
         double term = 1.0;

         // Begin the summation and recursion
         for ( int k = 1; k <= kmax ; k++ ){
            term = term*(2.*n - 2.*k + 1.)/(2.*n + 2.*k + 1);

            // Product of Bessel function quotients
            double p = 1.0;
            for ( int j = 1 ; j <= k ; j++ ){
               p = p*r[j-1];
            }
            double sumterm = factor*term*p*(2.*k + 1.)/(2.*n + 1.);
            //
            if (abs(sumterm) < DBL_EPSILON) {
               // Iteration converged
               break;
            } else if (k == kmax) {
               // No convergence with kmax iterations
               assert(k<kmax);
            } else {
               // Add the current term to the sum and continue the iteration
               sumtot = sumtot + sumterm;
            }
         }
         f[n] = sumtot;
      }
   } else {
      //Use asymptotic formula for T > 50
      f[0] = 0.5*sqrt(M_PI/T);
      // Use the upward recursion relation to
      // generate the remaining F_n(t) values
      double expt = exp(-T);
      for( int n = 1 ; n <= nmax ; n++ ){
         f[n] = 0.5*((2.*n-1.)*f[n-1] - expt)/T;
      }
   }
}

double* create_md_ftable( int nmax, double tmin, double tmax, double tdelta, int* ld ){
   int n = nmax + 6;
   int itabmin = int(floor(tmin/tdelta));
   int itabmax = int((tmax - tmin)/tdelta);
   int nelem = (itabmax - itabmin + 1 ) * (n+1);
   double * ftable = (double*) malloc(nelem*sizeof(double) );
   // Fill table
   for ( int itab = itabmin; itab <= itabmax; itab++ ){
      double T = double(itab)*tdelta;
      // fill ftable from itab*(n+1) to (itab+1)*(n+1) with f values
      fgamma_ref(n, T, &ftable[itab*(n+1)]);
   }
   *ld = n+1;
   return ftable;
}

void fgamma0_ref( int nmax, double T, double* f, const double* ftable, int ftable_ld ){
   if ( T < Teps ){
      // eps < T -> T=0
      for( int n = 0 ; n <= nmax ; n++ ){
         f[n] = 1.0/(2.*n+1.);
      }
   } else if ( T < 12.  ){
      // eps < T < 12 -> Taylor expansion + downward
      double tdelta = 0.1;
      int itab = int(round(T/tdelta));
      double ttab = double(itab)*tdelta;
      double tmp = 1.0;
      double inv_fac = 1.0; // 1/i!
      f[nmax] = ftable[itab*ftable_ld+nmax];
      // 6th order Taylor expansion
      tmp *= (ttab-T); inv_fac /= 1.0 ; f[nmax] += ftable[itab*ftable_ld+nmax+1]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac /= 2.0 ; f[nmax] += ftable[itab*ftable_ld+nmax+2]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac /= 3.0 ; f[nmax] += ftable[itab*ftable_ld+nmax+3]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac /= 4.0 ; f[nmax] += ftable[itab*ftable_ld+nmax+4]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac /= 5.0 ; f[nmax] += ftable[itab*ftable_ld+nmax+5]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac /= 6.0 ; f[nmax] += ftable[itab*ftable_ld+nmax+6]*tmp*inv_fac;
//      for ( int k=1; k <= 6; k++ ){
//         tmp *= (ttab-T);
//         f[nmax] += ftable[itab*ftable_ld+nmax+k]*tmp*ifac[k];
//      }
      // Use the downward recursion relation to 
      // generate the remaining F_n(t) values   
      double expt = exp(-T);
      for( int n = nmax-1 ; n >= 0 ; n--){
         f[n] = (2.0*T*f[n + 1] + expt)/(2.*n + 1.);
      }
   } else {
   // T > 12 -> segmented Taylor + upward
      double tmp = 1.0/T; // reciprocal value
      if (T <= 15.0) {
         // 12 < T <= 15 -> Four term polynom expansion
         double tmp2 = tmp*tmp;
         double tmp3 = tmp*tmp2;
         double g = 0.4999489092 - 0.2473631686*tmp + 0.321180909*tmp2 - 0.3811559346*tmp3;
         f[0] = 0.5*sqrt(M_PI*tmp) - g*exp(-T)*tmp;
      } else if (T <= 18.0) {
         // 15 < T <= 18 -> Three term polynom expansion
         double g = 0.4998436875 - 0.24249438*tmp + 0.24642845*tmp*tmp;
         f[0] = 0.5*sqrt(M_PI*tmp) - g*exp(-T)*tmp;
      } else if (T <= 24.0) {
         // 18 < T <= 24 -> Two term polynom expansion
         double g = 0.499093162 - 0.2152832*tmp;
         f[0] = 0.5*sqrt(M_PI*tmp) - g*exp(-T)*tmp;
      } else if (T <= 30.0) {
         // 24 < T <= 30 -> One term polynom expansion
         double g = 0.49;
         f[0] = 0.5*sqrt(M_PI*tmp) - g*exp(-T)*tmp;
      } else {
         // T > 30 -> Asymptotic formula
         f[0] = 0.5*sqrt(M_PI*tmp);
      }
      // Use the upward recursion relation to
      // generate the remaining F_n(t) values
      double expt = 0.0;
      if ( T > 2.*nmax + 36.){
         expt = 0.0;
      } else {
         expt = exp(-T);
      }
      for ( int n=1 ; n <= nmax ; n++ ){
         f[n] = (0.5*tmp)*( (2.*n - 1.)*f[n - 1] - expt );
      }
   }
}


__host__ __device__ void fgamma0( int nmax, double T, double* f, const double* ftable, int ftable_ld___, double fac ){
   if ( T < Teps ){
      // eps < T -> T=0
      for( int n = 0 ; n <= nmax ; n++ ){
         f[n] = fac/(2.*n+1.);
      }
      return;
   }
   if ( T < 12.  ){
      // eps < T < 12 -> Taylor expansion + downward
      constexpr double tdelta = 0.1;
      constexpr int ftable_ld = 28;
      int itab = int(round(T/tdelta));
      double ttab = double(itab)*tdelta;
      double tmp = 1.0;
      double inv_fac = 1.0; // 1/i!
      double fL = ftable[itab*ftable_ld+nmax+0];
      // 6th order Taylor expansion
      tmp *= (ttab-T); inv_fac = 1.0    ; fL += ftable[itab*ftable_ld+nmax+1]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac = 0.5    ; fL += ftable[itab*ftable_ld+nmax+2]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac = 1./6.  ; fL += ftable[itab*ftable_ld+nmax+3]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac = 1./24. ; fL += ftable[itab*ftable_ld+nmax+4]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac = 1./120.; fL += ftable[itab*ftable_ld+nmax+5]*tmp*inv_fac;
      tmp *= (ttab-T); inv_fac = 1./720.; fL += ftable[itab*ftable_ld+nmax+6]*tmp*inv_fac;
      // Use the downward recursion relation to 
      // generate the remaining F_n(t) values   
      double expt = exp(-T);
      f[nmax] = fL;
      for( int n = nmax-1 ; n >= 0 ; n--){
         f[n] = (2.0*T*f[n + 1] + expt)/(2.*n + 1.);
      }
   } else {
   // T > 12 -> 4 Taylor + upward
      double tmp = 1.0/T; // reciprocal value
      double tmp2 = tmp*tmp;
      double tmp3 = tmp*tmp2;
      double g = 0.4999489092 - 0.2473631686*tmp + 0.321180909*tmp2 - 0.3811559346*tmp3;
      // Use the upward recursion relation to
      // generate the remaining F_n(t) values
      double expt = exp(-T);
      f[0] = 0.5*sqrt(M_PI*tmp) - g*expt*tmp;
      for ( int n=1 ; n <= nmax ; n++ ){
         f[n] = (0.5*tmp)*( (2.*n - 1.)*f[n - 1] - expt );
      }
   }
   for( int n = 0 ; n <= nmax ; n++ ){ f[n] *= fac; }
   return;
}

#undef Teps
#undef NDERIV_MAX
#undef DEGREE
#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include "compute_KS.h"
#include "define.h"
#include "util.h"


__constant__ double symm_factors[5] = {2.0, 1.0, 0.5, 0.25, 0.0 };


// Add P @@ S[I] to K 
// where P is the density, K the Fock matrix, 
// S is ugual to minus the hf fraction times the symm factor for this combination of atoms
// The @@ symbol is the exchanged tensor contraction, where we 
// make use of the fact that the abcd ERI can be used with Pac Pad Pbc Pbd

__global__ void compute_KS_gpu(
      const int Nqrtt,
      const unsigned int * const __restrict__ KS,
      const int la, const int lb, const int lc, const int ld,
      const double * const __restrict__ P,
      const double * const __restrict__ I,
      double * const __restrict__ K,
      const double * const __restrict__ data,
      double hf_fac ){

   const int nsa = 2*la+1;
   const int nsb = 2*lb+1;
   const int nsc = 2*lc+1;
   const int nsd = 2*ld+1;
   const int nscd = nsc*nsd;
   const int nsbcd = nsb*nscd;
   const int nsabcd = nsa*nsbcd;

   for ( int block = blockIdx.x ; block < Nqrtt ; block += gridDim.x ){
      const unsigned int idx_fac = KS[block * KS_SIZE + KS_OFFSET_IDXFAC ];
      const unsigned int inlabcd = KS[block * KS_SIZE + KS_OFFSET_INLABCD];
      const unsigned int ld_abcd = KS[block * KS_SIZE + KS_OFFSET_LDABCD ];
      const unsigned int offset_L_set_atom_ac = KS[block * KS_SIZE + KS_OFFSET_OFFAC  ];
      const unsigned int offset_L_set_atom_ad = KS[block * KS_SIZE + KS_OFFSET_OFFAD  ];
      const unsigned int offset_L_set_atom_bc = KS[block * KS_SIZE + KS_OFFSET_OFFBC  ];
      const unsigned int offset_L_set_atom_bd = KS[block * KS_SIZE + KS_OFFSET_OFFBD  ];
      const unsigned int Tall    = KS[block * KS_SIZE + KS_OFFSET_TALL   ];

      unsigned int inla,inlb,inlc,inld;
      unsigned int ldac,ldad,ldbc,ldbd;
      unsigned int  Tac, Tad, Tbc, Tbd;

      decode4( inlabcd, &inla, &inlb, &inlc, &inld );
      decode4( ld_abcd, &ldac, &ldad, &ldbc, &ldbd );
      decode4( Tall   , &Tac,  &Tad,  &Tbc,  &Tbd  );

      const int Oq = block * nsabcd ;
      // MAYBE move the hf_fac multiplication outside this kernel, since K initial value is zero, only multiply once at the end
      // Cost would go from N(integral computeted) / n_gpu to SIZEOF( K ) for every gpu, so it may not be always useful
      // OR move the hf_fac multiplication to the GCC factors
      const double fac = symm_factors[idx_fac] * hf_fac;
      
      
      for ( int t = threadIdx.x ; t < nsabcd; t += blockDim.x ){


//         if ( block + t == 0 ){ printf( " ! P ! %p \n", P ); }
//         if ( block + t == 0 ){ printf( " ! P ! %p %lg         \n", P , P[0] ); }
//         if ( block + t == 0 ){ printf( " ! P ! %p %lg %lg     \n", P , P[0], P[1] ); }
//         if ( block + t == 0 ){ printf( " ! P ! %p %lg %lg %lg \n", P , P[0], P[1], P[2] ); }

         const int isa =   t / nsbcd;
         const int isb = ( t / nscd ) % nsb;
         const int isc = ( t / nsd  ) % nsc;
         const int isd =   t          % nsd;


//         const int idx_a = inla*nsa + isa;
//         const int idx_b = inlb*nsb + isb;
//         const int idx_c = inlc*nsc + isc;
//         const int idx_d = inld*nsd + isd;

         const int idx_a = isa;
         const int idx_b = isb;
         const int idx_c = isc;
         const int idx_d = isd;


//         const int idx_ac = Tac ?  idx_c * ldac + idx_a :  idx_a * ldac + idx_c;
//         const int idx_ad = Tad ?  idx_d * ldad + idx_a :  idx_a * ldad + idx_d;
//         const int idx_bc = Tbc ?  idx_c * ldbc + idx_b :  idx_b * ldbc + idx_c;
//         const int idx_bd = Tbd ?  idx_d * ldbd + idx_b :  idx_b * ldbd + idx_d;

         const int idx_ac_T = not Tac ?  idx_c * ldac + idx_a :  idx_a * ldac + idx_c;
         const int idx_ad_T = not Tad ?  idx_d * ldad + idx_a :  idx_a * ldad + idx_d;
         const int idx_bc_T = not Tbc ?  idx_c * ldbc + idx_b :  idx_b * ldbc + idx_c;
         const int idx_bd_T = not Tbd ?  idx_d * ldbd + idx_b :  idx_b * ldbd + idx_d;

//         const int Iac = offset_L_set_atom_ac + idx_ac;
//         const int Iad = offset_L_set_atom_ad + idx_ad;
//         const int Ibc = offset_L_set_atom_bc + idx_bc;
//         const int Ibd = offset_L_set_atom_bd + idx_bd;

         const int Iac_T = offset_L_set_atom_ac + idx_ac_T;
         const int Iad_T = offset_L_set_atom_ad + idx_ad_T;
         const int Ibc_T = offset_L_set_atom_bc + idx_bc_T;
         const int Ibd_T = offset_L_set_atom_bd + idx_bd_T;

         const double iabcd = - fac * I[ Oq + t ];
         const double kbd = iabcd * P [Iac_T];
         const double kbc = iabcd * P [Iad_T];
         const double kad = iabcd * P [Ibc_T];
         const double kac = iabcd * P [Ibd_T];


//         printf( " ---------- KS GPU %lg %lg %lg %lg %lg \n", iabcd, kbd,kbc,kad,kac );
//         printf("KS GPU %d.%d.0: Adding %4.10lg ( - %lg * %lg * %lg ) to %lg from P %d [%p] @ K %d [%p] \n ", block, t, kbd, fac, I[ Oq + t ], P[Iac_T], K[Ibd_T], Iac_T, &P[Iac_T], Ibd_T, &K[Ibd_T] );
//         printf("KS GPU %d.0: Adding %4.10lg ( - %lg * %lg * %lg ) to %lg from P %d @ K %d \n ", block, kbd, fac, I[ Oq + t ], P[Iac_T], K[Ibd_T], Iac_T, Ibd_T );
//         printf("KS GPU %d.1: Adding %4.10lg ( - %lg * %lg * %lg ) to %lg from P %d @ K %d \n ", block, kbc, fac, I[ Oq + t ], P[Iad_T], K[Ibc_T], Iad_T, Ibc_T );
//         printf("KS GPU %d.2: Adding %4.10lg ( - %lg * %lg * %lg ) to %lg from P %d @ K %d \n ", block, kad, fac, I[ Oq + t ], P[Ibc_T], K[Iad_T], Ibc_T, Iad_T );
//         printf("KS GPU %d.3: Adding %4.10lg ( - %lg * %lg * %lg ) to %lg from P %d @ K %d \n ", block, kac, fac, I[ Oq + t ], P[Ibd_T], K[Iac_T], Ibd_T, Iac_T );

         // MUST be atomics on device, or however K is distributed
         atomicAdd( &K[ Ibd_T ] , kbd);
         atomicAdd( &K[ Ibc_T ] , kbc);
         atomicAdd( &K[ Iad_T ] , kad);
         atomicAdd( &K[ Iac_T ] , kac);

      }
   }
}

/////////////// OUTDATED ////////////////////////

void compute_KS(
      const int Nqrtt,
      const unsigned int * const __restrict__ KS,
      const int la, const int lb, const int lc, const int ld,
      const double * const __restrict__ P,
      const double * const __restrict__ I,
      double * const __restrict__ K,
      const double * const __restrict__ data,
      double hf_fac ){

   const int nsa = 2*la+1;
   const int nsb = 2*lb+1;
   const int nsc = 2*lc+1;
   const int nsd = 2*ld+1;
   const int nscd = nsc*nsd;
   const int nsbcd = nsb*nscd;
   const int nsabcd = nsa*nsbcd;

   for ( int block = 0 ; block < Nqrtt ; block++ ){
      const unsigned int idx_fac = KS[block * KS_SIZE + KS_OFFSET_IDXFAC ];
      const unsigned int inlabcd = KS[block * KS_SIZE + KS_OFFSET_INLABCD];
      const unsigned int ld_abcd = KS[block * KS_SIZE + KS_OFFSET_LDABCD ];
      const unsigned int offset_L_set_atom_ac = KS[block * KS_SIZE + KS_OFFSET_OFFAC  ];
      const unsigned int offset_L_set_atom_ad = KS[block * KS_SIZE + KS_OFFSET_OFFAD  ];
      const unsigned int offset_L_set_atom_bc = KS[block * KS_SIZE + KS_OFFSET_OFFBC  ];
      const unsigned int offset_L_set_atom_bd = KS[block * KS_SIZE + KS_OFFSET_OFFBD  ];
      const unsigned int Tall    = KS[block * KS_SIZE + KS_OFFSET_TALL   ];

      unsigned int inla,inlb,inlc,inld;
      unsigned int ldac,ldad,ldbc,ldbd;
      unsigned int  Tac, Tad, Tbc, Tbd;

      decode4( inlabcd, &inla, &inlb, &inlc, &inld );
      decode4( ld_abcd, &ldac, &ldad, &ldbc, &ldbd );
      decode4( Tall   , &Tac,  &Tad,  &Tbc,  &Tbd  );

      const unsigned int Oq = block * nsabcd ;
      const double fac = data[idx_fac];
      
      for ( int t = 0 ; t < nsabcd; t++ ){
         const int isa =   t / nsbcd;
         const int isb = ( t / nscd ) % nsb;
         const int isc = ( t / nsd  ) % nsc;
         const int isd =   t          % nsd;

         const int idx_a = inla*nsa + isa;
         const int idx_b = inlb*nsb + isb;
         const int idx_c = inlc*nsc + isc;
         const int idx_d = inld*nsd + isd;

         const int idx_ac = Tac ?  idx_c * ldac + idx_a :  idx_a * ldac + idx_c;
         const int idx_ad = Tad ?  idx_d * ldad + idx_a :  idx_a * ldad + idx_d;
         const int idx_bc = Tbc ?  idx_c * ldbc + idx_b :  idx_b * ldbc + idx_c;
         const int idx_bd = Tbd ?  idx_d * ldbd + idx_b :  idx_b * ldbd + idx_d;

         const int Iac = offset_L_set_atom_ac + idx_ac;
         const int Iad = offset_L_set_atom_ad + idx_ad;
         const int Ibc = offset_L_set_atom_bc + idx_bc;
         const int Ibd = offset_L_set_atom_bd + idx_bd;
         const double iabcd = - fac * I[ Oq + t ];
         const double pac = P [Iac];
         const double pad = P [Iad];
         const double pbc = P [Ibc];
         const double pbd = P [Ibd];
         // must be atomics
         // MAYBE move the hf_fac multiplication outside this kernel, since K initial value is zero, only multiply once at the end
         // Cost would go from N(integral computeted) / n_gpu to SIZEOF( K ) for every gpu, so it may not be always useful
         K[ Ibd ] += hf_fac * pac * iabcd;
         K[ Ibc ] += hf_fac * pad * iabcd;
         K[ Iad ] += hf_fac * pbc * iabcd;
         K[ Iac ] += hf_fac * pbd * iabcd;
      }
   }
}



#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include "compute_HRR.h"
#include "define.h"
#include "util.h"


// computes HHR of the type ascd = as+- + CD asc-
__device__ void execute_HRR1_gpu(
      const int AL, const int CL, const int DL,
      double* __restrict__ abcd,
      const double* const __restrict__ abpm,
      const double* const __restrict__ abcm,
      const double CD[3], const int hrr_blocksize, const int nlabcd,
      const int tid=0, const int Nt=1 ){

   const int NcoA  = NLco_dev(AL);
   const int NcoC  = NLco_dev(CL);
   const int NcoCp = NLco_dev(CL+1);
   const int NcoD  = NLco_dev(DL);
   const int NcoDm = NLco_dev(DL-1);
   
   for( int ikl=tid ; ikl < NcoA*NcoC*NcoD ; ikl+=Nt ){
      int l = ikl%NcoD;
      int k = (ikl/NcoD)%NcoC;
      int i = ikl/NcoD/NcoC;
      int kp, lm, d ;
      int fx = lx_dev(k,CL);

      if ( l < NcoDm ){
         kp = k;
         lm = l;
         d = 0;
      } else if ( l < NcoD-1 ){
         kp = k + (CL-fx) + 1;
         lm = l - DL;
         d = 1;
      } else {
         lm = l - DL - 1;
         kp = k + (CL-fx) + 2;
         d = 2;
      }
      for ( int nl_idx = 0; nl_idx < nlabcd; nl_idx++ ){
         int idx_off = nl_idx * hrr_blocksize;
         int idx_00 = idx_off + ikl ; // (i*NcoC +k )*NcoD  + l;
         int idx_pm = idx_off + (i*NcoCp+kp)*NcoDm + lm;
         int idx_0m = idx_off + (i*NcoC +k )*NcoDm + lm;
         abcd[ idx_00 ] = abpm[idx_pm] + CD[d] * abcm[idx_0m];
      }
   }
}


// General HHR of the type abcd = +-cd + AB a-cd
__device__ void execute_HRR2_gpu(
      const int AL, const int BL, const int CL, const int DL,
      double* const __restrict__ abcd,
      const double* const __restrict__ pmcd,
      const double* const __restrict__ amcd,
      const double AB[3], const int hrr_blocksize, const int nlabcd,
      const int tid=0, const int Nt=1 ){

   const int NcoA = NLco_dev(AL);
   const int NcoB = NLco_dev(BL);
   const int NcoBm = NLco_dev(BL-1);
   const int NcoC = NLco_dev(CL);
   const int NcoD = NLco_dev(DL);
   const int NcoABCD = NcoA*NcoB*NcoC*NcoD;

   for( int ijkl=tid ; ijkl < NcoABCD ; ijkl+=Nt ){

      int l = ijkl%NcoD;
      int k = (ijkl/NcoD)%NcoC;
      int j = (ijkl/NcoD/NcoC)%NcoB;
      int i = ijkl/NcoD/NcoC/NcoB;// % NcoA
      int ex = lx_dev(i,AL);
      int ip, jm, d ;

      if ( j < NcoBm ){
         ip = i;
         jm = j;
         d = 0;
      } else if ( j<NcoBm+BL ) {
         ip = i + (AL-ex) + 1;
         jm = j - BL;
         d = 1;
      } else {
         ip = i + (AL-ex) + 2;
         jm = j - BL - 1;
         d = 2;
      }
      for ( int nl_idx = 0; nl_idx < nlabcd; nl_idx++ ){ // TODO Maybe fuse ?
         int idx_off = nl_idx * hrr_blocksize;
         int idx_00 = idx_off + ijkl ; // (i *NcoB +j )*NcoC*NcoD+k*NcoD+l;
         int idx_pm = idx_off + (ip*NcoBm+jm)*NcoC*NcoD+k*NcoD+l;
         int idx_0m = idx_off + (i *NcoBm+jm)*NcoC*NcoD+k*NcoD+l;
         abcd[ idx_00 ] = pmcd[ idx_pm ] + AB[d] * amcd[ idx_0m ];
      }
   }
}


__global__ void compute_HRR_batched_gpu_low(
      const int Ncells,
      const int* const __restrict__ plan,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ data,
      double* const __restrict__ ABCD,
      double* const __restrict__ ABCD0,
      bool periodic,
      const double* const __restrict__ cell,
      const double* const __restrict__ neighs,
      int hrr_blocksize, int Nc, int numVC, int numVCH ){

   for( int block=blockIdx.x; block < Ncells ; block+=gridDim.x ){

      unsigned int Og     = FVH[block*FVH_SIZE+FVH_OFFSET_OG];
      unsigned int Oq     = FVH[block*FVH_SIZE+FVH_OFFSET_OQ];
      unsigned int idx_A  = FVH[block*FVH_SIZE+FVH_OFFSET_IDX_A];
      unsigned int idx_B  = FVH[block*FVH_SIZE+FVH_OFFSET_IDX_B];
      unsigned int idx_C  = FVH[block*FVH_SIZE+FVH_OFFSET_IDX_C];
      unsigned int idx_D  = FVH[block*FVH_SIZE+FVH_OFFSET_IDX_D];
      unsigned int elabcd = FVH[block*FVH_SIZE+FVH_OFFSET_ELABCD];
      int nlabcd = (int)    FVH[block*FVH_SIZE+FVH_OFFSET_NLABCD];

//      if (blockIdx.x == 0 and threadIdx.x == 0 ){
//         printf(" HRR FVH: " );
//         for ( int ii = 0 ; ii < FVH_SIZE ; ii++ ){
//            printf( " %u " , FVH[block*FVH_SIZE+ii] );
//         } printf("\n"); ; 
//      }

      unsigned int nla,nlb,nlc,nld, n1,n2;


      decode_shell( elabcd, &nla,&nlb,&nlc,&nld,&n1,&n2);


      
//      if (threadIdx.x == 0 ){
//         printf(" e %u | nla %u nlb %u nlc %u nld %u n1 %d n2 %d nlabcd %d \n" , encoded_nlabcd_12, nla,nlb,nlc,nld,n1,n2,nlabcd);
//      }

      const double * const Ao =  &data[idx_A];
      const double * const Bo =  &data[idx_B];
      double A[3], B[3], ABs[3];
      compute_pbc( Ao, Bo, cell, ABs );

      A[0] = Ao[0];
      A[1] = Ao[1];
      A[2] = Ao[2];
      B[0] = Ao[0] + ABs[0] + neighs[n1*3+0];
      B[1] = Ao[1] + ABs[1] + neighs[n1*3+1];
      B[2] = Ao[2] + ABs[2] + neighs[n1*3+2];

      const double AB[3] = { A[0]-B[0], A[1]-B[1], A[2]-B[2] };

      const double * const Co =  &data[idx_C];
      const double * const Do =  &data[idx_D];
      double C[3], D[3], CDs[3];
      compute_pbc( Co, Do, cell, CDs );
      C[0] = Co[0];
      C[1] = Co[1];
      C[2] = Co[2];
      D[0] = Co[0] + CDs[0] + neighs[n2*3+0];
      D[1] = Co[1] + CDs[1] + neighs[n2*3+1];
      D[2] = Co[2] + CDs[2] + neighs[n2*3+2];
      const double CD[3] = { C[0]-D[0], C[1]-D[1], C[2]-D[2] };

//      if (threadIdx.x == 0 ){
//         printf(" shifting A %lf %lf %lf and B %lf %lf %lf by %lf %lf %lf \n", 
//            Ao[0], Ao[1], Ao[2], Bo[0], Bo[1], Bo[2], ABs[0], ABs[1], ABs[2] );
//         printf(" shifting C %lf %lf %lf and D %lf %lf %lf by %lf %lf %lf \n", 
//            Co[0], Co[1], Co[2], Do[0], Do[1], Do[2], CDs[0], CDs[1], CDs[2] );
//         printf(" n1 %d n2 %d \n" , n1 , n2 );
//      }

      double* sh_mem = &ABCD[ Og * hrr_blocksize ];

//      for ( unsigned int ilabcd=0 ; ilabcd < nlabcd; ilabcd++ ){
//         int s0_st = (Oq + ilabcd) * Nc;
//         for ( int i=threadIdx.x ; i < Nc; i+=blockDim.x ){
//            ABCD0[ s0_st + i ] = 0. ;
//         }
//      }

      __syncthreads();

      const int tid = threadIdx.x;
      const int Nt = blockDim.x;

      for ( int op=numVC; op < numVCH; op++ ){
         int t  = plan[ op*OP_SIZE + T__OFFSET ];
         int la = plan[ op*OP_SIZE + LA_OFFSET ];
         int lb = plan[ op*OP_SIZE + LB_OFFSET ];
         int lc = plan[ op*OP_SIZE + LC_OFFSET ];
         int ld = plan[ op*OP_SIZE + LD_OFFSET ];
         int off_m1 = plan[ op*OP_SIZE + M1_OFFSET ];
         int off_m2 = plan[ op*OP_SIZE + M2_OFFSET ];
         int off_m3 = plan[ op*OP_SIZE + M3_OFFSET ];

         if ( t == HRR1 ){
            execute_HRR1_gpu(
               la, lc, ld, &sh_mem[off_m1], &sh_mem[off_m2], &sh_mem[off_m3],
               CD, hrr_blocksize, nlabcd, tid, Nt );
         } else if ( t == HRR2 ){
            execute_HRR2_gpu(
               la, lb, lc, ld, &sh_mem[off_m1], &sh_mem[off_m2], &sh_mem[off_m3],
               AB, hrr_blocksize, nlabcd, tid, Nt );
         } else if ( t == SYBL ){
//            #pragma omp team barrier (?)
            __syncthreads();
         }
      }

      __syncthreads();
      // sums over cell from ABCD[Og:Og+nlabcd*Nc] to ABCD0[Oq:Oq+nlabcd*Nc]
      for ( unsigned int ilabcd=threadIdx.x ; ilabcd < nlabcd; ilabcd+=blockDim.x ){
         int sh_st = ilabcd * hrr_blocksize;
         int s0_st = (Oq + ilabcd) * Nc;
         for ( int i=0 ; i < Nc; i++ ){
            // must be atomic
            atomicAdd( &ABCD0[ s0_st + i ] , sh_mem[ sh_st + i ]);
         }
      } 
   }
}





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



#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include<cstdint>
#include <iostream>
#include "compute_VRR.h"
#include "util.h"
#include "define.h"

using std::cout;
using std::endl;

#define FM_N_VEC 4
#define FM_N_SCA 5
__device__ __host__ constexpr int Fsize_but_constexpr( int L ){ return L + 1 + FM_N_VEC * 3 + FM_N_SCA; }
#undef FM_N_VEC
#undef FM_N_SCA



__host__ __device__ constexpr int SA ( int L ) {
   int tmp[] = {0,1,4,10,20,35,56, 84,120, 165};
   return tmp[L];
}

__host__ __device__ constexpr int NLco_dev_but_constexpr( int L ){ return (L+1)*(L+2) / 2; }

constexpr int MAXL = 4*2;

__constant__ int _d[ SA(MAXL+1) ] = {
0, 
0, 1, 2, 
0, 0, 0, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
};

__constant__ int _idx_m[ SA(MAXL+1) ] = {
0, 
0, 0, 0, 
0, 1, 2, 1, 2, 2, 
0, 1, 2, 3, 4, 5, 3, 4, 5, 5, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 9, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 14, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 15, 16, 17, 18, 19, 20, 20, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 21, 22, 23, 24, 25, 26, 27, 27, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 28, 29, 30, 31, 32, 33, 34, 35, 35, 
};

__constant__ int _idx_w[ SA(MAXL+1) ] = {
0, 
0, 0, 0, 
0, 1, 2, 0, 1, 0, 
0, 1, 2, 3, 4, 5, 1, 2, 3, 2, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 5, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 6, 7, 8, 9, 10, 9, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10, 11, 12, 13, 14, 15, 14, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 15, 16, 17, 18, 19, 20, 21, 20, 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 21, 22, 23, 24, 25, 26, 27, 28, 27, 
};

__constant__ double _f2[ SA(MAXL+1) ] = {
0, 
0, 0, 0, 
1, 0, 0, 1, 0, 1, 
2, 1, 1, 0, 0, 0, 2, 1, 0, 2, 
3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 2, 1, 0, 3, 
4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 3, 2, 1, 0, 4, 
5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 5, 
6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4, 3, 2, 1, 0, 6, 
7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0, 7, 
};

__constant__ int _idx_off[ 3*(MAXL+1) ] = {
     0,  0,  0,
     3,  6,  9,
    12, 18, 24,
    30, 40, 50,
    60, 75, 90,
   105,126,147,
   168,196,224,
   252,288,324,
   360,405,450
};


__constant__ uint8_t _idx_r[ 3*SA(MAXL+1) ] = {
0, 
0, 
0, // 3
0, 1, 2, 
0, 0, 1, 
0, 0, 0, // 12
0, 1, 2, 3, 4, 5, 
0, 0, 1, 1, 2, 3, 
0, 0, 0, 0, 1, 2, // 30
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, // 60
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, // 105
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, // 168
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, // 84*3 = 252
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, // 120*3 = 360
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 33, 34, 35,  // 165*3 = 495
};

__constant__ double _e2[ 3*SA(MAXL+1) ] = {
0, 
0, 
0, 
1, 0, 0, 
0, 1, 0, 
0, 0, 1, 
2, 1, 1, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 
3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 
4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 
5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 
6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 
7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 
8, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0, 
0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8,   
};


template < int VTS, int AL, int CL, int m >
__device__ void execute_VRR1_gpu( // int AL, int CL, int m, 
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double PA_WP[6] ){

   int my_vrr_rank = threadIdx.x % VTS ;
   for( int imm=my_vrr_rank; imm < 3*m; imm+=VTS ){
      int mm = imm / 3 ;
      int i  = imm % 3 ;
      int d  = i ;
      int im = 0 ; 
      int idx_000 = i  + 3*mm;
      int idx_m00 = im + 1*mm;
      int idx_m0p = im + 1*(1+mm);
      a0c0m0[idx_000] = PA_WP[  d] * amc0m0[idx_m00] + 
                        PA_WP[d+3] * amc0m0[idx_m0p];

//      printf(" (1) i mm d %d %d %d | PA/QC * M00 %lg %d | WP/WQ * M0P %lg %d | \n", i,mm,d,PA_WP[d],idx_m00, PA_WP[d+3],idx_m0p);

   }
}

template < int VTS, int AL, int CL, int m >
__device__ void execute_VRR2_gpu(
      double* __restrict__ a0c0m0,
      const double* __restrict__ amc0m0,
      const double* __restrict__ awc0m0,
      const double PA_WP[3], const double inv_2zab[2] ){
 
   int my_vrr_rank = threadIdx.x % VTS ; 
   const int NcoA   = ((AL+1)*(AL+2))/2;
   const int NcoAm  = ((AL+0)*(AL+1))/2;
   const int NcoAw  = ((AL-1)*(AL+0))/2;

   for ( int imm = my_vrr_rank; imm < m*NcoA; imm+=VTS ){

      int i  = imm % NcoA ;
      int mm = imm / NcoA ;
      int d  = _d[SA(AL)+i];
      int im = _idx_m[SA(AL)+i];
      int iw = _idx_w[SA(AL)+i];
      double e2 = _f2[SA(AL)+i];

      int idx_000 = imm ;
      int idx_m00 = im + NcoAm*mm ;
      int idx_w00 = iw + NcoAw*mm ;
      int idx_m0p = idx_m00 + NcoAm;
      int idx_w0p = idx_w00 + NcoAw;

      a0c0m0[idx_000] = PA_WP[d  ] * amc0m0[idx_m00] + 
                        PA_WP[d+3] * amc0m0[idx_m0p] + 
                        e2*( inv_2zab[0] * awc0m0[idx_w00] + inv_2zab[1] * awc0m0[idx_w0p] );

//      printf(" (2) i m d im iw %d %d %d %d %d | PA/QC * M00 %lg %d | WP/WQ * M0P %lg %d | e2 * z * W00 + e2 * z * W0P %lg %lg %d %lg %d \n", 
//                 i,mm,d,im,iw,               PA_WP[d],idx_m00, PA_WP[d+3],idx_m0p,         e2,inv_2zab[0],idx_w00,inv_2zab[1],idx_w0p );
   }
}


// more general case, for al>0 and cl>1
template < int VTS, int AL, int CL, int m >
__device__ void execute_VRR5_gpu(
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ a0cwm0,
      const double* __restrict__ amcmmp,
      const double QC_WQ[6],
      const double inv_2zcd[3] ){

   int my_vrr_rank = threadIdx.x % VTS ;

   constexpr int NcoA  = NLco_dev_but_constexpr(AL);
   constexpr int NcoAm = NLco_dev_but_constexpr(AL-1);
   constexpr int NcoC  = NLco_dev_but_constexpr(CL);
   constexpr int NcoCm = NLco_dev_but_constexpr(CL-1);
   constexpr int NcoCw = NLco_dev_but_constexpr(CL-2);

   // k is the faster variable, followed by i
   for ( int ik=my_vrr_rank; ik < NcoA*NcoC; ik+=VTS ){

      double f2, e2;
      int i, k, d, km, kw, im, o;
      int idx_000, idx_0m0, idx_0w0, idx_mmp, idx_0mp, idx_0wp;
      double i_0m0, i_0mp, i_0w0, i_0wp, i_mmp;

      k  = ik % NcoC;
      i  = ik / NcoC;
      constexpr int t = SA(CL);
      d  = _d[t+k];
      km = _idx_m[t+k];
      kw = _idx_w[t+k];
      f2 = _f2[t+k];
      o  = _idx_off[3*AL+d];
      im = _idx_r[o+i];
      e2 = _e2[o+i];

      for ( int mm = 0; mm < m; mm++ ){
         idx_000 = k +i *NcoC  + NcoA *NcoC *mm;
         idx_0m0 = km+i *NcoCm + NcoA *NcoCm*mm;
         idx_0mp = km+i *NcoCm + NcoA *NcoCm*(1+mm);
         idx_0w0 = kw+i *NcoCw + NcoA *NcoCw*mm;
         idx_0wp = kw+i *NcoCw + NcoA *NcoCw*(1+mm);
         idx_mmp = km+im*NcoCm + NcoAm*NcoCm*(mm);

         i_0m0 = a0cmm0[ idx_0m0 ];
         i_0mp = a0cmm0[ idx_0mp ];
         i_0w0 = a0cwm0[ idx_0w0 ];
         i_0wp = a0cwm0[ idx_0wp ];
         i_mmp = amcmmp[ idx_mmp ];

         a0c0m0[ idx_000 ] = QC_WQ[d] * i_0m0 + QC_WQ[d+3] * i_0mp + f2*( inv_2zcd[0] * i_0w0 + inv_2zcd[1] * i_0wp ) + e2*inv_2zcd[2]*i_mmp;

//         printf(" (5) %d %d | i k m d km kw f2 o im e2 %d %d %d %d %d %d %lg %d %d %lg \n", AL,CL, i,k,mm,d,km,kw,f2,o,im,e2 );
 
      }
   }
}

template< int VTS, int  AL, int CL, int m >
__device__ void execute_VRR6_gpu( 
      double* __restrict__ a0c0m0,
      const double* __restrict__ a0cmm0,
      const double* __restrict__ amcmmp,
      const double QC_WQ[6], const double inv_2z ){

   int my_vrr_rank = threadIdx.x % VTS ;
      
   const int NcoA = NLco_dev_but_constexpr(AL);
   const int NcoAm= NLco_dev_but_constexpr(AL-1);
   const int NcoC = NLco_dev_but_constexpr(CL);
   const int NcoCm= NLco_dev_but_constexpr(CL-1);
   // k is the faster variable, followed by i, then m

   for ( int ik=my_vrr_rank; ik < NcoA*NcoC; ik+=VTS ){

      int i, k, km, d, im, o;
      double e2 ;
      int idx_000, idx_0m0, idx_0mp, idx_mmp;
      double i_0m0, i_0mp, i_mmp;

      k  = ik % NcoC;
      i  = ik / NcoC;

      d = k;
      o  = _idx_off[3*AL+d];
      im = _idx_r[o+i];
      e2 = _e2[o+i];
      km = 0;

      for ( int mm = 0; mm < m; mm++ ){
         idx_000 = k +i *NcoC  + NcoA *NcoC *(  mm);
         idx_0m0 = km+i *NcoCm + NcoA *NcoCm*(  mm);
         idx_0mp = km+i *NcoCm + NcoA *NcoCm*(1+mm);
         idx_mmp = km+im*NcoCm + NcoAm*NcoCm*(mm);

         i_0m0 = a0cmm0[ idx_0m0 ];
         i_0mp = a0cmm0[ idx_0mp ];
         i_mmp = amcmmp[ idx_mmp ];

         a0c0m0[ idx_000 ] = QC_WQ[d] * i_0m0 + QC_WQ[d+3] * i_0mp + e2*inv_2z*i_mmp;

//         printf(" (6) %d %d | i k m d km o im e2 %d %d %d %d %d %d %d %lg \n", AL,CL, i,k,mm,d,km,o,im,e2 );
      }
   }
}





template< int VTS >
__device__ void ssss(double * ss0 , double para[4*3+5] ){
}

template< int VTS >
__device__ void sssp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double*  sp0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
}

template< int VTS >
__device__ void sssd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 3 ;
   double*  sd0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sssf(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssps(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double*  sp0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
}

template< int VTS >
__device__ void sspp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 3 ;
   double*  sd0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sspd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sspf(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 5 ;
   double*  sd0  = ss0 + 17 ;
   double*  sf0  = ss0 + 35 ;
   double*  sg0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssds(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 3 ;
   double*  sd0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssdp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssdd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 5 ;
   double*  sd0  = ss0 + 17 ;
   double*  sf0  = ss0 + 35 ;
   double*  sg0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssdf(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 6 ;
   double*  sd0  = ss0 + 21 ;
   double*  sf0  = ss0 + 45 ;
   double*  sg0  = ss0 + 75 ;
   double*  sh0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssfs(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 4 ;
   double*  sd0  = ss0 + 13 ;
   double*  sf0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssfp(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 5 ;
   double*  sd0  = ss0 + 17 ;
   double*  sf0  = ss0 + 35 ;
   double*  sg0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssfd(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 6 ;
   double*  sd0  = ss0 + 21 ;
   double*  sf0  = ss0 + 45 ;
   double*  sg0  = ss0 + 75 ;
   double*  sh0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ssff(double * ss0 , double para[4*3+5] ){
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double*  sp0  = ss0 + 7 ;
   double*  sd0  = ss0 + 25 ;
   double*  sf0  = ss0 + 55 ;
   double*  sg0  = ss0 + 95 ;
   double*  sh0  = ss0 + 140 ;
   double*  si0  = ss0 + 182 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double*  ps0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( ps0 , ss0 , PA_WP);
}

template< int VTS >
__device__ void spsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  sp0  = ss0 + 9 ;
   double*  pp0  = ss0 + 12 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void spsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp0  = ss0 + 13 ;
   double*  sp1  = ss0 + 16 ;
   double*  sd0  = ss0 + 19 ;
   double*  pp0  = ss0 + 25 ;
   double*  pd0  = ss0 + 43 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  pp0  = ss0 + 26 ;
   double*  sd0  = ss0 + 53 ;
   double*  sd1  = ss0 + 59 ;
   double*  sf0  = ss0 + 65 ;
   double*  pd0  = ss0 + 75 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  sp0  = ss0 + 9 ;
   double*  pp0  = ss0 + 12 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void sppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp0  = ss0 + 13 ;
   double*  sp1  = ss0 + 16 ;
   double*  pp0  = ss0 + 19 ;
   double*  sd0  = ss0 + 37 ;
   double*  pd0  = ss0 + 43 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  sd0  = ss0 + 26 ;
   double*  sd1  = ss0 + 32 ;
   double*  pp0  = ss0 + 38 ;
   double*  pd0  = ss0 + 65 ;
   double*  sf0  = ss0 + 101 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp0  = ss0 + 21 ;
   double*  sp1  = ss0 + 24 ;
   double*  pp0  = ss0 + 33 ;
   double*  sd0  = ss0 + 69 ;
   double*  sd1  = ss0 + 75 ;
   double*  sf0  = ss0 + 87 ;
   double*  sf1  = ss0 + 97 ;
   double*  pd0  = ss0 + 107 ;
   double*  sg0  = ss0 + 161 ;
   double*  pf0  = ss0 + 176 ;
   double*  pg0  = ss0 + 236 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp0  = ss0 + 13 ;
   double*  sp1  = ss0 + 16 ;
   double*  sd0  = ss0 + 19 ;
   double*  pp0  = ss0 + 25 ;
   double*  pd0  = ss0 + 43 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  sd0  = ss0 + 26 ;
   double*  sd1  = ss0 + 32 ;
   double*  pp0  = ss0 + 38 ;
   double*  pd0  = ss0 + 65 ;
   double*  sf0  = ss0 + 101 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp0  = ss0 + 21 ;
   double*  sp1  = ss0 + 24 ;
   double*  pp0  = ss0 + 33 ;
   double*  sd0  = ss0 + 69 ;
   double*  sd1  = ss0 + 75 ;
   double*  sf0  = ss0 + 87 ;
   double*  sf1  = ss0 + 97 ;
   double*  pd0  = ss0 + 107 ;
   double*  pf0  = ss0 + 161 ;
   double*  sg0  = ss0 + 221 ;
   double*  pg0  = ss0 + 236 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp0  = ss0 + 25 ;
   double*  sp1  = ss0 + 28 ;
   double*  pp0  = ss0 + 40 ;
   double*  sd0  = ss0 + 85 ;
   double*  sd1  = ss0 + 91 ;
   double*  pd0  = ss0 + 109 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  sg0  = ss0 + 211 ;
   double*  sg1  = ss0 + 226 ;
   double*  pf0  = ss0 + 241 ;
   double*  sh0  = ss0 + 331 ;
   double*  pg0  = ss0 + 352 ;
   double*  ph0  = ss0 + 442 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp0  = ss0 + 17 ;
   double*  sp1  = ss0 + 20 ;
   double*  sd0  = ss0 + 26 ;
   double*  sd1  = ss0 + 32 ;
   double*  pp0  = ss0 + 38 ;
   double*  pd0  = ss0 + 65 ;
   double*  sf0  = ss0 + 101 ;
   double*  pf0  = ss0 + 111 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp0  = ss0 + 21 ;
   double*  sp1  = ss0 + 24 ;
   double*  sd0  = ss0 + 33 ;
   double*  sd1  = ss0 + 39 ;
   double*  pp0  = ss0 + 51 ;
   double*  pd0  = ss0 + 87 ;
   double*  sf0  = ss0 + 141 ;
   double*  sf1  = ss0 + 151 ;
   double*  sg0  = ss0 + 161 ;
   double*  pf0  = ss0 + 176 ;
   double*  pg0  = ss0 + 236 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp0  = ss0 + 25 ;
   double*  sp1  = ss0 + 28 ;
   double*  pp0  = ss0 + 40 ;
   double*  sd0  = ss0 + 85 ;
   double*  sd1  = ss0 + 91 ;
   double*  pd0  = ss0 + 109 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  sg0  = ss0 + 211 ;
   double*  sg1  = ss0 + 226 ;
   double*  pf0  = ss0 + 241 ;
   double*  sh0  = ss0 + 331 ;
   double*  pg0  = ss0 + 352 ;
   double*  ph0  = ss0 + 442 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void spff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  sp0  = ss0 + 29 ;
   double*  sp1  = ss0 + 32 ;
   double*  sd0  = ss0 + 47 ;
   double*  sd1  = ss0 + 53 ;
   double*  pp0  = ss0 + 77 ;
   double*  pd0  = ss0 + 131 ;
   double*  sf0  = ss0 + 221 ;
   double*  sf1  = ss0 + 231 ;
   double*  sg0  = ss0 + 261 ;
   double*  sg1  = ss0 + 276 ;
   double*  pf0  = ss0 + 306 ;
   double*  sh0  = ss0 + 426 ;
   double*  sh1  = ss0 + 447 ;
   double*  pg0  = ss0 + 468 ;
   double*  ph0  = ss0 + 603 ;
   double*  si0  = ss0 + 729 ;
   double*  pi0  = ss0 + 757 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 3 ;
   double*  ds0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( ds0 , ps0 , ss0 , PA_WP, zab );
}

template< int VTS >
__device__ void sdsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  sp0  = ss0 + 25 ;
   double*  pp0  = ss0 + 28 ;
   double*  dp0  = ss0 + 37 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void sdsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp0  = ss0 + 35 ;
   double*  sp1  = ss0 + 38 ;
   double*  sd0  = ss0 + 41 ;
   double*  pp0  = ss0 + 47 ;
   double*  pp1  = ss0 + 56 ;
   double*  pd0  = ss0 + 65 ;
   double*  dp0  = ss0 + 83 ;
   double*  dd0  = ss0 + 119 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  sd0  = ss0 + 54 ;
   double*  sd1  = ss0 + 60 ;
   double*  pp0  = ss0 + 66 ;
   double*  pp1  = ss0 + 75 ;
   double*  sf0  = ss0 + 93 ;
   double*  pd0  = ss0 + 103 ;
   double*  pd1  = ss0 + 121 ;
   double*  dp0  = ss0 + 139 ;
   double*  pf0  = ss0 + 193 ;
   double*  dd0  = ss0 + 223 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  sp0  = ss0 + 25 ;
   double*  pp0  = ss0 + 28 ;
   double*  dp0  = ss0 + 37 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void sdpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp0  = ss0 + 35 ;
   double*  sp1  = ss0 + 38 ;
   double*  sd0  = ss0 + 41 ;
   double*  pp0  = ss0 + 47 ;
   double*  pp1  = ss0 + 56 ;
   double*  pd0  = ss0 + 65 ;
   double*  dp0  = ss0 + 83 ;
   double*  dd0  = ss0 + 119 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  pp0  = ss0 + 54 ;
   double*  pp1  = ss0 + 63 ;
   double*  sd0  = ss0 + 81 ;
   double*  sd1  = ss0 + 87 ;
   double*  sf0  = ss0 + 93 ;
   double*  dp0  = ss0 + 103 ;
   double*  pd0  = ss0 + 157 ;
   double*  pd1  = ss0 + 175 ;
   double*  dd0  = ss0 + 193 ;
   double*  pf0  = ss0 + 265 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp0  = ss0 + 55 ;
   double*  sp1  = ss0 + 58 ;
   double*  pp0  = ss0 + 67 ;
   double*  pp1  = ss0 + 76 ;
   double*  sd0  = ss0 + 103 ;
   double*  sd1  = ss0 + 109 ;
   double*  dp0  = ss0 + 121 ;
   double*  pd0  = ss0 + 193 ;
   double*  pd1  = ss0 + 211 ;
   double*  sf0  = ss0 + 247 ;
   double*  sf1  = ss0 + 257 ;
   double*  dd0  = ss0 + 267 ;
   double*  pf0  = ss0 + 375 ;
   double*  pf1  = ss0 + 405 ;
   double*  sg0  = ss0 + 435 ;
   double*  df0  = ss0 + 450 ;
   double*  pg0  = ss0 + 570 ;
   double*  dg0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp0  = ss0 + 35 ;
   double*  sp1  = ss0 + 38 ;
   double*  sd0  = ss0 + 41 ;
   double*  pp0  = ss0 + 47 ;
   double*  pp1  = ss0 + 56 ;
   double*  pd0  = ss0 + 65 ;
   double*  dp0  = ss0 + 83 ;
   double*  dd0  = ss0 + 119 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  pp0  = ss0 + 54 ;
   double*  pp1  = ss0 + 63 ;
   double*  sd0  = ss0 + 81 ;
   double*  sd1  = ss0 + 87 ;
   double*  sf0  = ss0 + 93 ;
   double*  dp0  = ss0 + 103 ;
   double*  pd0  = ss0 + 157 ;
   double*  pd1  = ss0 + 175 ;
   double*  dd0  = ss0 + 193 ;
   double*  pf0  = ss0 + 265 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp0  = ss0 + 55 ;
   double*  sp1  = ss0 + 58 ;
   double*  sd0  = ss0 + 67 ;
   double*  sd1  = ss0 + 73 ;
   double*  pp0  = ss0 + 85 ;
   double*  pp1  = ss0 + 94 ;
   double*  sf0  = ss0 + 121 ;
   double*  sf1  = ss0 + 131 ;
   double*  pd0  = ss0 + 141 ;
   double*  pd1  = ss0 + 159 ;
   double*  dp0  = ss0 + 195 ;
   double*  sg0  = ss0 + 267 ;
   double*  pf0  = ss0 + 282 ;
   double*  pf1  = ss0 + 312 ;
   double*  dd0  = ss0 + 342 ;
   double*  pg0  = ss0 + 450 ;
   double*  df0  = ss0 + 495 ;
   double*  dg0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp0  = ss0 + 65 ;
   double*  sp1  = ss0 + 68 ;
   double*  pp0  = ss0 + 80 ;
   double*  pp1  = ss0 + 89 ;
   double*  sd0  = ss0 + 125 ;
   double*  sd1  = ss0 + 131 ;
   double*  dp0  = ss0 + 149 ;
   double*  pd0  = ss0 + 239 ;
   double*  pd1  = ss0 + 257 ;
   double*  sf0  = ss0 + 311 ;
   double*  sf1  = ss0 + 321 ;
   double*  sg0  = ss0 + 341 ;
   double*  sg1  = ss0 + 356 ;
   double*  dd0  = ss0 + 371 ;
   double*  pf0  = ss0 + 515 ;
   double*  pf1  = ss0 + 545 ;
   double*  sh0  = ss0 + 605 ;
   double*  df0  = ss0 + 626 ;
   double*  pg0  = ss0 + 806 ;
   double*  pg1  = ss0 + 851 ;
   double*  dg0  = ss0 + 896 ;
   double*  ph0  = ss0 + 1076 ;
   double*  dh0  = ss0 + 1139 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp0  = ss0 + 45 ;
   double*  sp1  = ss0 + 48 ;
   double*  pp0  = ss0 + 54 ;
   double*  pp1  = ss0 + 63 ;
   double*  sd0  = ss0 + 81 ;
   double*  sd1  = ss0 + 87 ;
   double*  pd0  = ss0 + 93 ;
   double*  pd1  = ss0 + 111 ;
   double*  dp0  = ss0 + 129 ;
   double*  sf0  = ss0 + 183 ;
   double*  dd0  = ss0 + 193 ;
   double*  pf0  = ss0 + 265 ;
   double*  df0  = ss0 + 295 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp0  = ss0 + 55 ;
   double*  sp1  = ss0 + 58 ;
   double*  sd0  = ss0 + 67 ;
   double*  sd1  = ss0 + 73 ;
   double*  pp0  = ss0 + 85 ;
   double*  pp1  = ss0 + 94 ;
   double*  sf0  = ss0 + 121 ;
   double*  sf1  = ss0 + 131 ;
   double*  pd0  = ss0 + 141 ;
   double*  pd1  = ss0 + 159 ;
   double*  dp0  = ss0 + 195 ;
   double*  sg0  = ss0 + 267 ;
   double*  pf0  = ss0 + 282 ;
   double*  pf1  = ss0 + 312 ;
   double*  dd0  = ss0 + 342 ;
   double*  pg0  = ss0 + 450 ;
   double*  df0  = ss0 + 495 ;
   double*  dg0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp0  = ss0 + 65 ;
   double*  sp1  = ss0 + 68 ;
   double*  pp0  = ss0 + 80 ;
   double*  pp1  = ss0 + 89 ;
   double*  sd0  = ss0 + 125 ;
   double*  sd1  = ss0 + 131 ;
   double*  dp0  = ss0 + 149 ;
   double*  pd0  = ss0 + 239 ;
   double*  pd1  = ss0 + 257 ;
   double*  sf0  = ss0 + 311 ;
   double*  sf1  = ss0 + 321 ;
   double*  sg0  = ss0 + 341 ;
   double*  sg1  = ss0 + 356 ;
   double*  dd0  = ss0 + 371 ;
   double*  pf0  = ss0 + 515 ;
   double*  pf1  = ss0 + 545 ;
   double*  sh0  = ss0 + 605 ;
   double*  df0  = ss0 + 626 ;
   double*  pg0  = ss0 + 806 ;
   double*  pg1  = ss0 + 851 ;
   double*  dg0  = ss0 + 896 ;
   double*  ph0  = ss0 + 1076 ;
   double*  dh0  = ss0 + 1139 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sdff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  sd0  = ss0 + 93 ;
   double*  sd1  = ss0 + 99 ;
   double*  pp0  = ss0 + 123 ;
   double*  pp1  = ss0 + 132 ;
   double*  sf0  = ss0 + 177 ;
   double*  sf1  = ss0 + 187 ;
   double*  dp0  = ss0 + 217 ;
   double*  pd0  = ss0 + 325 ;
   double*  pd1  = ss0 + 343 ;
   double*  sg0  = ss0 + 415 ;
   double*  sg1  = ss0 + 430 ;
   double*  dd0  = ss0 + 460 ;
   double*  pf0  = ss0 + 640 ;
   double*  pf1  = ss0 + 670 ;
   double*  pg0  = ss0 + 760 ;
   double*  pg1  = ss0 + 805 ;
   double*  sh0  = ss0 + 895 ;
   double*  sh1  = ss0 + 916 ;
   double*  df0  = ss0 + 937 ;
   double*  si0  = ss0 + 1177 ;
   double*  ph0  = ss0 + 1205 ;
   double*  ph1  = ss0 + 1268 ;
   double*  dg0  = ss0 + 1331 ;
   double*  pi0  = ss0 + 1601 ;
   double*  dh0  = ss0 + 1685 ;
   double*  di0  = ss0 + 1937 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void sfsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  sp0  = ss0 + 55 ;
   double*  pp0  = ss0 + 58 ;
   double*  dp0  = ss0 + 67 ;
   double*  fp0  = ss0 + 85 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void sfsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  pp0  = ss0 + 81 ;
   double*  pp1  = ss0 + 90 ;
   double*  sd0  = ss0 + 99 ;
   double*  pd0  = ss0 + 105 ;
   double*  dp0  = ss0 + 123 ;
   double*  dp1  = ss0 + 141 ;
   double*  fp0  = ss0 + 159 ;
   double*  dd0  = ss0 + 219 ;
   double*  fd0  = ss0 + 255 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  sf0  = ss0 + 143 ;
   double*  dp0  = ss0 + 153 ;
   double*  dp1  = ss0 + 171 ;
   double*  pd0  = ss0 + 207 ;
   double*  pd1  = ss0 + 225 ;
   double*  fp0  = ss0 + 243 ;
   double*  dd0  = ss0 + 333 ;
   double*  dd1  = ss0 + 369 ;
   double*  pf0  = ss0 + 405 ;
   double*  fd0  = ss0 + 435 ;
   double*  df0  = ss0 + 555 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  sp0  = ss0 + 55 ;
   double*  pp0  = ss0 + 58 ;
   double*  dp0  = ss0 + 67 ;
   double*  fp0  = ss0 + 85 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void sfpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  sd0  = ss0 + 81 ;
   double*  pp0  = ss0 + 87 ;
   double*  pp1  = ss0 + 96 ;
   double*  pd0  = ss0 + 105 ;
   double*  dp0  = ss0 + 123 ;
   double*  dp1  = ss0 + 141 ;
   double*  fp0  = ss0 + 159 ;
   double*  dd0  = ss0 + 219 ;
   double*  fd0  = ss0 + 255 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  pd0  = ss0 + 143 ;
   double*  pd1  = ss0 + 161 ;
   double*  sf0  = ss0 + 179 ;
   double*  dp0  = ss0 + 189 ;
   double*  dp1  = ss0 + 207 ;
   double*  pf0  = ss0 + 243 ;
   double*  dd0  = ss0 + 273 ;
   double*  dd1  = ss0 + 309 ;
   double*  fp0  = ss0 + 345 ;
   double*  df0  = ss0 + 435 ;
   double*  fd0  = ss0 + 495 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp0  = ss0 + 115 ;
   double*  sp1  = ss0 + 118 ;
   double*  pp0  = ss0 + 127 ;
   double*  pp1  = ss0 + 136 ;
   double*  sd0  = ss0 + 163 ;
   double*  sd1  = ss0 + 169 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  dp0  = ss0 + 201 ;
   double*  dp1  = ss0 + 219 ;
   double*  pd0  = ss0 + 273 ;
   double*  pd1  = ss0 + 291 ;
   double*  sg0  = ss0 + 327 ;
   double*  fp0  = ss0 + 342 ;
   double*  dd0  = ss0 + 462 ;
   double*  dd1  = ss0 + 498 ;
   double*  pf0  = ss0 + 570 ;
   double*  pf1  = ss0 + 600 ;
   double*  pg0  = ss0 + 630 ;
   double*  fd0  = ss0 + 675 ;
   double*  df0  = ss0 + 855 ;
   double*  df1  = ss0 + 915 ;
   double*  ff0  = ss0 + 975 ;
   double*  dg0  = ss0 + 1175 ;
   double*  fg0  = ss0 + 1265 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp0  = ss0 + 75 ;
   double*  sp1  = ss0 + 78 ;
   double*  sd0  = ss0 + 81 ;
   double*  pp0  = ss0 + 87 ;
   double*  pp1  = ss0 + 96 ;
   double*  dp0  = ss0 + 105 ;
   double*  dp1  = ss0 + 123 ;
   double*  pd0  = ss0 + 141 ;
   double*  fp0  = ss0 + 159 ;
   double*  dd0  = ss0 + 219 ;
   double*  fd0  = ss0 + 255 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  pd0  = ss0 + 143 ;
   double*  pd1  = ss0 + 161 ;
   double*  dp0  = ss0 + 179 ;
   double*  dp1  = ss0 + 197 ;
   double*  sf0  = ss0 + 233 ;
   double*  pf0  = ss0 + 243 ;
   double*  dd0  = ss0 + 273 ;
   double*  dd1  = ss0 + 309 ;
   double*  fp0  = ss0 + 345 ;
   double*  df0  = ss0 + 435 ;
   double*  fd0  = ss0 + 495 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp0  = ss0 + 115 ;
   double*  sp1  = ss0 + 118 ;
   double*  pp0  = ss0 + 127 ;
   double*  pp1  = ss0 + 136 ;
   double*  sd0  = ss0 + 163 ;
   double*  sd1  = ss0 + 169 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  dp0  = ss0 + 201 ;
   double*  dp1  = ss0 + 219 ;
   double*  pd0  = ss0 + 273 ;
   double*  pd1  = ss0 + 291 ;
   double*  sg0  = ss0 + 327 ;
   double*  fp0  = ss0 + 342 ;
   double*  dd0  = ss0 + 462 ;
   double*  dd1  = ss0 + 498 ;
   double*  pf0  = ss0 + 570 ;
   double*  pf1  = ss0 + 600 ;
   double*  pg0  = ss0 + 630 ;
   double*  fd0  = ss0 + 675 ;
   double*  df0  = ss0 + 855 ;
   double*  df1  = ss0 + 915 ;
   double*  ff0  = ss0 + 975 ;
   double*  dg0  = ss0 + 1175 ;
   double*  fg0  = ss0 + 1265 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp0  = ss0 + 135 ;
   double*  sp1  = ss0 + 138 ;
   double*  sd0  = ss0 + 150 ;
   double*  sd1  = ss0 + 156 ;
   double*  pp0  = ss0 + 174 ;
   double*  pp1  = ss0 + 183 ;
   double*  sf0  = ss0 + 219 ;
   double*  sf1  = ss0 + 229 ;
   double*  dp0  = ss0 + 249 ;
   double*  dp1  = ss0 + 267 ;
   double*  pd0  = ss0 + 339 ;
   double*  pd1  = ss0 + 357 ;
   double*  sg0  = ss0 + 411 ;
   double*  sg1  = ss0 + 426 ;
   double*  pf0  = ss0 + 441 ;
   double*  pf1  = ss0 + 471 ;
   double*  fp0  = ss0 + 531 ;
   double*  dd0  = ss0 + 681 ;
   double*  dd1  = ss0 + 717 ;
   double*  sh0  = ss0 + 825 ;
   double*  pg0  = ss0 + 846 ;
   double*  pg1  = ss0 + 891 ;
   double*  df0  = ss0 + 936 ;
   double*  df1  = ss0 + 996 ;
   double*  fd0  = ss0 + 1116 ;
   double*  ph0  = ss0 + 1356 ;
   double*  dg0  = ss0 + 1419 ;
   double*  dg1  = ss0 + 1509 ;
   double*  ff0  = ss0 + 1599 ;
   double*  dh0  = ss0 + 1899 ;
   double*  fg0  = ss0 + 2025 ;
   double*  fh0  = ss0 + 2325 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp0  = ss0 + 95 ;
   double*  sp1  = ss0 + 98 ;
   double*  sd0  = ss0 + 104 ;
   double*  sd1  = ss0 + 110 ;
   double*  pp0  = ss0 + 116 ;
   double*  pp1  = ss0 + 125 ;
   double*  pd0  = ss0 + 143 ;
   double*  pd1  = ss0 + 161 ;
   double*  sf0  = ss0 + 179 ;
   double*  dp0  = ss0 + 189 ;
   double*  dp1  = ss0 + 207 ;
   double*  pf0  = ss0 + 243 ;
   double*  dd0  = ss0 + 273 ;
   double*  dd1  = ss0 + 309 ;
   double*  fp0  = ss0 + 345 ;
   double*  df0  = ss0 + 435 ;
   double*  fd0  = ss0 + 495 ;
   double*  ff0  = ss0 + 615 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp0  = ss0 + 115 ;
   double*  sp1  = ss0 + 118 ;
   double*  pp0  = ss0 + 127 ;
   double*  pp1  = ss0 + 136 ;
   double*  sd0  = ss0 + 163 ;
   double*  sd1  = ss0 + 169 ;
   double*  sf0  = ss0 + 181 ;
   double*  sf1  = ss0 + 191 ;
   double*  dp0  = ss0 + 201 ;
   double*  dp1  = ss0 + 219 ;
   double*  pd0  = ss0 + 273 ;
   double*  pd1  = ss0 + 291 ;
   double*  sg0  = ss0 + 327 ;
   double*  dd0  = ss0 + 342 ;
   double*  dd1  = ss0 + 378 ;
   double*  fp0  = ss0 + 450 ;
   double*  pf0  = ss0 + 570 ;
   double*  pf1  = ss0 + 600 ;
   double*  pg0  = ss0 + 630 ;
   double*  fd0  = ss0 + 675 ;
   double*  df0  = ss0 + 855 ;
   double*  df1  = ss0 + 915 ;
   double*  ff0  = ss0 + 975 ;
   double*  dg0  = ss0 + 1175 ;
   double*  fg0  = ss0 + 1265 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp0  = ss0 + 135 ;
   double*  sp1  = ss0 + 138 ;
   double*  sd0  = ss0 + 150 ;
   double*  sd1  = ss0 + 156 ;
   double*  pp0  = ss0 + 174 ;
   double*  pp1  = ss0 + 183 ;
   double*  sf0  = ss0 + 219 ;
   double*  sf1  = ss0 + 229 ;
   double*  pd0  = ss0 + 249 ;
   double*  pd1  = ss0 + 267 ;
   double*  dp0  = ss0 + 321 ;
   double*  dp1  = ss0 + 339 ;
   double*  sg0  = ss0 + 411 ;
   double*  sg1  = ss0 + 426 ;
   double*  fp0  = ss0 + 441 ;
   double*  dd0  = ss0 + 591 ;
   double*  dd1  = ss0 + 627 ;
   double*  pf0  = ss0 + 735 ;
   double*  pf1  = ss0 + 765 ;
   double*  sh0  = ss0 + 825 ;
   double*  pg0  = ss0 + 846 ;
   double*  pg1  = ss0 + 891 ;
   double*  df0  = ss0 + 936 ;
   double*  df1  = ss0 + 996 ;
   double*  fd0  = ss0 + 1116 ;
   double*  ph0  = ss0 + 1356 ;
   double*  dg0  = ss0 + 1419 ;
   double*  dg1  = ss0 + 1509 ;
   double*  ff0  = ss0 + 1599 ;
   double*  dh0  = ss0 + 1899 ;
   double*  fg0  = ss0 + 2025 ;
   double*  fh0  = ss0 + 2325 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void sfff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp0  = ss0 + 155 ;
   double*  sp1  = ss0 + 158 ;
   double*  pp0  = ss0 + 173 ;
   double*  pp1  = ss0 + 182 ;
   double*  sd0  = ss0 + 227 ;
   double*  sd1  = ss0 + 233 ;
   double*  dp0  = ss0 + 257 ;
   double*  dp1  = ss0 + 275 ;
   double*  sf0  = ss0 + 365 ;
   double*  sf1  = ss0 + 375 ;
   double*  pd0  = ss0 + 405 ;
   double*  pd1  = ss0 + 423 ;
   double*  pf0  = ss0 + 495 ;
   double*  pf1  = ss0 + 525 ;
   double*  sg0  = ss0 + 615 ;
   double*  sg1  = ss0 + 630 ;
   double*  dd0  = ss0 + 660 ;
   double*  dd1  = ss0 + 696 ;
   double*  fp0  = ss0 + 840 ;
   double*  df0  = ss0 + 1020 ;
   double*  df1  = ss0 + 1080 ;
   double*  pg0  = ss0 + 1260 ;
   double*  pg1  = ss0 + 1305 ;
   double*  fd0  = ss0 + 1395 ;
   double*  sh0  = ss0 + 1695 ;
   double*  sh1  = ss0 + 1716 ;
   double*  si0  = ss0 + 1737 ;
   double*  ff0  = ss0 + 1765 ;
   double*  ph0  = ss0 + 2165 ;
   double*  ph1  = ss0 + 2228 ;
   double*  dg0  = ss0 + 2291 ;
   double*  dg1  = ss0 + 2381 ;
   double*  fg0  = ss0 + 2561 ;
   double*  dh0  = ss0 + 3011 ;
   double*  dh1  = ss0 + 3137 ;
   double*  pi0  = ss0 + 3263 ;
   double*  fh0  = ss0 + 3347 ;
   double*  di0  = ss0 + 3767 ;
   double*  fi0  = ss0 + 3935 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 6 >( sp0 , ss0 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 5 >( sd0 , sp0 , ss0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 4 >( sf0 , sd0 , sp0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 3 >( sg0 , sf0 , sd0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 2 >( sh0 , sg0 , sf0 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( si0 , sh0 , sg0 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double*  ps0  = ss0 + 2 ;
   execute_VRR1_gpu< VTS, 1, 0, 1 >( ps0 , ss0 , PA_WP);
}

template< int VTS >
__device__ void pssp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  pp0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void pssd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp1  = ss0 + 13 ;
   double*  pp0  = ss0 + 16 ;
   double*  pd0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pssf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 3 ;
   double*  pp0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
}

template< int VTS >
__device__ void pspp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp1  = ss0 + 13 ;
   double*  pp0  = ss0 + 16 ;
   double*  pd0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pspd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pspf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp1  = ss0 + 21 ;
   double*  sd1  = ss0 + 30 ;
   double*  pp0  = ss0 + 42 ;
   double*  pd0  = ss0 + 78 ;
   double*  sf1  = ss0 + 132 ;
   double*  pf0  = ss0 + 142 ;
   double*  pg0  = ss0 + 202 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  sp1  = ss0 + 13 ;
   double*  pp0  = ss0 + 16 ;
   double*  pd0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp1  = ss0 + 21 ;
   double*  sd1  = ss0 + 30 ;
   double*  pp0  = ss0 + 42 ;
   double*  pd0  = ss0 + 78 ;
   double*  sf1  = ss0 + 132 ;
   double*  pf0  = ss0 + 142 ;
   double*  pg0  = ss0 + 202 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp1  = ss0 + 25 ;
   double*  pp0  = ss0 + 37 ;
   double*  sd1  = ss0 + 82 ;
   double*  pd0  = ss0 + 100 ;
   double*  sf1  = ss0 + 172 ;
   double*  sg1  = ss0 + 192 ;
   double*  pf0  = ss0 + 207 ;
   double*  pg0  = ss0 + 297 ;
   double*  ph0  = ss0 + 387 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  sp1  = ss0 + 17 ;
   double*  sd1  = ss0 + 23 ;
   double*  pp0  = ss0 + 29 ;
   double*  pd0  = ss0 + 56 ;
   double*  pf0  = ss0 + 92 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  sp1  = ss0 + 21 ;
   double*  sd1  = ss0 + 30 ;
   double*  pp0  = ss0 + 42 ;
   double*  pd0  = ss0 + 78 ;
   double*  sf1  = ss0 + 132 ;
   double*  pf0  = ss0 + 142 ;
   double*  pg0  = ss0 + 202 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  sp1  = ss0 + 25 ;
   double*  pp0  = ss0 + 37 ;
   double*  sd1  = ss0 + 82 ;
   double*  pd0  = ss0 + 100 ;
   double*  sf1  = ss0 + 172 ;
   double*  sg1  = ss0 + 192 ;
   double*  pf0  = ss0 + 207 ;
   double*  pg0  = ss0 + 297 ;
   double*  ph0  = ss0 + 387 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void psff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  sp1  = ss0 + 29 ;
   double*  pp0  = ss0 + 44 ;
   double*  sd1  = ss0 + 98 ;
   double*  sf1  = ss0 + 122 ;
   double*  pd0  = ss0 + 152 ;
   double*  pf0  = ss0 + 242 ;
   double*  sg1  = ss0 + 362 ;
   double*  pg0  = ss0 + 392 ;
   double*  sh1  = ss0 + 527 ;
   double*  ph0  = ss0 + 548 ;
   double*  pi0  = ss0 + 674 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 3 ;
   double*  ds0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( ds0 , ps0 , ss0 , PA_WP, zab );
}

template< int VTS >
__device__ void ppsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  pp0  = ss0 + 25 ;
   double*  dp0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void ppsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp1  = ss0 + 35 ;
   double*  pp0  = ss0 + 38 ;
   double*  pp1  = ss0 + 47 ;
   double*  pd0  = ss0 + 56 ;
   double*  dp0  = ss0 + 74 ;
   double*  dd0  = ss0 + 110 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  sd1  = ss0 + 51 ;
   double*  pp0  = ss0 + 57 ;
   double*  pp1  = ss0 + 66 ;
   double*  pd0  = ss0 + 84 ;
   double*  pd1  = ss0 + 102 ;
   double*  dp0  = ss0 + 120 ;
   double*  pf0  = ss0 + 174 ;
   double*  dd0  = ss0 + 204 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  pp0  = ss0 + 25 ;
   double*  dp0  = ss0 + 34 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void pppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp1  = ss0 + 35 ;
   double*  pp0  = ss0 + 38 ;
   double*  pp1  = ss0 + 47 ;
   double*  pd0  = ss0 + 56 ;
   double*  dp0  = ss0 + 74 ;
   double*  dd0  = ss0 + 110 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  pp0  = ss0 + 51 ;
   double*  pp1  = ss0 + 60 ;
   double*  sd1  = ss0 + 78 ;
   double*  dp0  = ss0 + 84 ;
   double*  pd0  = ss0 + 138 ;
   double*  pd1  = ss0 + 156 ;
   double*  dd0  = ss0 + 174 ;
   double*  pf0  = ss0 + 246 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp1  = ss0 + 55 ;
   double*  sd1  = ss0 + 64 ;
   double*  pp0  = ss0 + 76 ;
   double*  pp1  = ss0 + 85 ;
   double*  sf1  = ss0 + 112 ;
   double*  pd0  = ss0 + 122 ;
   double*  pd1  = ss0 + 140 ;
   double*  dp0  = ss0 + 176 ;
   double*  pf0  = ss0 + 248 ;
   double*  pf1  = ss0 + 278 ;
   double*  dd0  = ss0 + 308 ;
   double*  pg0  = ss0 + 416 ;
   double*  df0  = ss0 + 461 ;
   double*  dg0  = ss0 + 581 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  sp1  = ss0 + 35 ;
   double*  pp0  = ss0 + 38 ;
   double*  pp1  = ss0 + 47 ;
   double*  pd0  = ss0 + 56 ;
   double*  dp0  = ss0 + 74 ;
   double*  dd0  = ss0 + 110 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  pp0  = ss0 + 51 ;
   double*  pp1  = ss0 + 60 ;
   double*  sd1  = ss0 + 78 ;
   double*  pd0  = ss0 + 84 ;
   double*  pd1  = ss0 + 102 ;
   double*  dp0  = ss0 + 120 ;
   double*  pf0  = ss0 + 174 ;
   double*  dd0  = ss0 + 204 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp1  = ss0 + 55 ;
   double*  sd1  = ss0 + 64 ;
   double*  pp0  = ss0 + 76 ;
   double*  pp1  = ss0 + 85 ;
   double*  sf1  = ss0 + 112 ;
   double*  pd0  = ss0 + 122 ;
   double*  pd1  = ss0 + 140 ;
   double*  dp0  = ss0 + 176 ;
   double*  pf0  = ss0 + 248 ;
   double*  pf1  = ss0 + 278 ;
   double*  dd0  = ss0 + 308 ;
   double*  pg0  = ss0 + 416 ;
   double*  df0  = ss0 + 461 ;
   double*  dg0  = ss0 + 581 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp1  = ss0 + 65 ;
   double*  pp0  = ss0 + 77 ;
   double*  pp1  = ss0 + 86 ;
   double*  sd1  = ss0 + 122 ;
   double*  dp0  = ss0 + 140 ;
   double*  pd0  = ss0 + 230 ;
   double*  pd1  = ss0 + 248 ;
   double*  sf1  = ss0 + 302 ;
   double*  sg1  = ss0 + 322 ;
   double*  dd0  = ss0 + 337 ;
   double*  pf0  = ss0 + 481 ;
   double*  pf1  = ss0 + 511 ;
   double*  df0  = ss0 + 571 ;
   double*  pg0  = ss0 + 751 ;
   double*  pg1  = ss0 + 796 ;
   double*  dg0  = ss0 + 841 ;
   double*  ph0  = ss0 + 1021 ;
   double*  dh0  = ss0 + 1084 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp1  = ss0 + 45 ;
   double*  pp0  = ss0 + 51 ;
   double*  pp1  = ss0 + 60 ;
   double*  sd1  = ss0 + 78 ;
   double*  dp0  = ss0 + 84 ;
   double*  pd0  = ss0 + 138 ;
   double*  pd1  = ss0 + 156 ;
   double*  pf0  = ss0 + 174 ;
   double*  dd0  = ss0 + 204 ;
   double*  df0  = ss0 + 276 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp1  = ss0 + 55 ;
   double*  sd1  = ss0 + 64 ;
   double*  pp0  = ss0 + 76 ;
   double*  pp1  = ss0 + 85 ;
   double*  sf1  = ss0 + 112 ;
   double*  pd0  = ss0 + 122 ;
   double*  pd1  = ss0 + 140 ;
   double*  dp0  = ss0 + 176 ;
   double*  pf0  = ss0 + 248 ;
   double*  pf1  = ss0 + 278 ;
   double*  dd0  = ss0 + 308 ;
   double*  pg0  = ss0 + 416 ;
   double*  df0  = ss0 + 461 ;
   double*  dg0  = ss0 + 581 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp1  = ss0 + 65 ;
   double*  pp0  = ss0 + 77 ;
   double*  pp1  = ss0 + 86 ;
   double*  sd1  = ss0 + 122 ;
   double*  dp0  = ss0 + 140 ;
   double*  pd0  = ss0 + 230 ;
   double*  pd1  = ss0 + 248 ;
   double*  sf1  = ss0 + 302 ;
   double*  dd0  = ss0 + 322 ;
   double*  pf0  = ss0 + 466 ;
   double*  pf1  = ss0 + 496 ;
   double*  sg1  = ss0 + 556 ;
   double*  df0  = ss0 + 571 ;
   double*  pg0  = ss0 + 751 ;
   double*  pg1  = ss0 + 796 ;
   double*  dg0  = ss0 + 841 ;
   double*  ph0  = ss0 + 1021 ;
   double*  dh0  = ss0 + 1084 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ppff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  sp1  = ss0 + 75 ;
   double*  sd1  = ss0 + 90 ;
   double*  pp0  = ss0 + 114 ;
   double*  pp1  = ss0 + 123 ;
   double*  sf1  = ss0 + 168 ;
   double*  dp0  = ss0 + 198 ;
   double*  pd0  = ss0 + 306 ;
   double*  pd1  = ss0 + 324 ;
   double*  sg1  = ss0 + 396 ;
   double*  dd0  = ss0 + 426 ;
   double*  pf0  = ss0 + 606 ;
   double*  pf1  = ss0 + 636 ;
   double*  pg0  = ss0 + 726 ;
   double*  pg1  = ss0 + 771 ;
   double*  sh1  = ss0 + 861 ;
   double*  df0  = ss0 + 882 ;
   double*  ph0  = ss0 + 1122 ;
   double*  ph1  = ss0 + 1185 ;
   double*  dg0  = ss0 + 1248 ;
   double*  pi0  = ss0 + 1518 ;
   double*  dh0  = ss0 + 1602 ;
   double*  di0  = ss0 + 1854 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void pdsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  pp0  = ss0 + 55 ;
   double*  dp0  = ss0 + 64 ;
   double*  fp0  = ss0 + 82 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void pdsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp1  = ss0 + 75 ;
   double*  pp0  = ss0 + 78 ;
   double*  pp1  = ss0 + 87 ;
   double*  pd0  = ss0 + 96 ;
   double*  dp0  = ss0 + 114 ;
   double*  dp1  = ss0 + 132 ;
   double*  fp0  = ss0 + 150 ;
   double*  dd0  = ss0 + 210 ;
   double*  fd0  = ss0 + 246 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  pp0  = ss0 + 55 ;
   double*  dp0  = ss0 + 64 ;
   double*  fp0  = ss0 + 82 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void pdpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp1  = ss0 + 75 ;
   double*  pp0  = ss0 + 78 ;
   double*  pp1  = ss0 + 87 ;
   double*  pd0  = ss0 + 96 ;
   double*  dp0  = ss0 + 114 ;
   double*  dp1  = ss0 + 132 ;
   double*  dd0  = ss0 + 150 ;
   double*  fp0  = ss0 + 186 ;
   double*  fd0  = ss0 + 246 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp1  = ss0 + 115 ;
   double*  pp0  = ss0 + 124 ;
   double*  pp1  = ss0 + 133 ;
   double*  sd1  = ss0 + 160 ;
   double*  sf1  = ss0 + 172 ;
   double*  dp0  = ss0 + 182 ;
   double*  dp1  = ss0 + 200 ;
   double*  pd0  = ss0 + 254 ;
   double*  pd1  = ss0 + 272 ;
   double*  fp0  = ss0 + 308 ;
   double*  dd0  = ss0 + 428 ;
   double*  dd1  = ss0 + 464 ;
   double*  pf0  = ss0 + 536 ;
   double*  pf1  = ss0 + 566 ;
   double*  pg0  = ss0 + 596 ;
   double*  fd0  = ss0 + 641 ;
   double*  df0  = ss0 + 821 ;
   double*  df1  = ss0 + 881 ;
   double*  ff0  = ss0 + 941 ;
   double*  dg0  = ss0 + 1141 ;
   double*  fg0  = ss0 + 1231 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  sp1  = ss0 + 75 ;
   double*  pp0  = ss0 + 78 ;
   double*  pp1  = ss0 + 87 ;
   double*  pd0  = ss0 + 96 ;
   double*  dp0  = ss0 + 114 ;
   double*  dp1  = ss0 + 132 ;
   double*  dd0  = ss0 + 150 ;
   double*  fp0  = ss0 + 186 ;
   double*  fd0  = ss0 + 246 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp1  = ss0 + 115 ;
   double*  pp0  = ss0 + 124 ;
   double*  pp1  = ss0 + 133 ;
   double*  sd1  = ss0 + 160 ;
   double*  sf1  = ss0 + 172 ;
   double*  dp0  = ss0 + 182 ;
   double*  dp1  = ss0 + 200 ;
   double*  pd0  = ss0 + 254 ;
   double*  pd1  = ss0 + 272 ;
   double*  fp0  = ss0 + 308 ;
   double*  dd0  = ss0 + 428 ;
   double*  dd1  = ss0 + 464 ;
   double*  pf0  = ss0 + 536 ;
   double*  pf1  = ss0 + 566 ;
   double*  pg0  = ss0 + 596 ;
   double*  fd0  = ss0 + 641 ;
   double*  df0  = ss0 + 821 ;
   double*  df1  = ss0 + 881 ;
   double*  ff0  = ss0 + 941 ;
   double*  dg0  = ss0 + 1141 ;
   double*  fg0  = ss0 + 1231 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp1  = ss0 + 135 ;
   double*  sd1  = ss0 + 147 ;
   double*  pp0  = ss0 + 165 ;
   double*  pp1  = ss0 + 174 ;
   double*  sf1  = ss0 + 210 ;
   double*  dp0  = ss0 + 230 ;
   double*  dp1  = ss0 + 248 ;
   double*  pd0  = ss0 + 320 ;
   double*  pd1  = ss0 + 338 ;
   double*  sg1  = ss0 + 392 ;
   double*  fp0  = ss0 + 407 ;
   double*  dd0  = ss0 + 557 ;
   double*  dd1  = ss0 + 593 ;
   double*  pf0  = ss0 + 701 ;
   double*  pf1  = ss0 + 731 ;
   double*  pg0  = ss0 + 791 ;
   double*  pg1  = ss0 + 836 ;
   double*  df0  = ss0 + 881 ;
   double*  df1  = ss0 + 941 ;
   double*  fd0  = ss0 + 1061 ;
   double*  ph0  = ss0 + 1301 ;
   double*  dg0  = ss0 + 1364 ;
   double*  dg1  = ss0 + 1454 ;
   double*  ff0  = ss0 + 1544 ;
   double*  dh0  = ss0 + 1844 ;
   double*  fg0  = ss0 + 1970 ;
   double*  fh0  = ss0 + 2270 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp1  = ss0 + 95 ;
   double*  sd1  = ss0 + 101 ;
   double*  pp0  = ss0 + 107 ;
   double*  pp1  = ss0 + 116 ;
   double*  pd0  = ss0 + 134 ;
   double*  pd1  = ss0 + 152 ;
   double*  dp0  = ss0 + 170 ;
   double*  dp1  = ss0 + 188 ;
   double*  pf0  = ss0 + 224 ;
   double*  dd0  = ss0 + 254 ;
   double*  dd1  = ss0 + 290 ;
   double*  fp0  = ss0 + 326 ;
   double*  df0  = ss0 + 416 ;
   double*  fd0  = ss0 + 476 ;
   double*  ff0  = ss0 + 596 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp1  = ss0 + 115 ;
   double*  pp0  = ss0 + 124 ;
   double*  pp1  = ss0 + 133 ;
   double*  sd1  = ss0 + 160 ;
   double*  sf1  = ss0 + 172 ;
   double*  dp0  = ss0 + 182 ;
   double*  dp1  = ss0 + 200 ;
   double*  pd0  = ss0 + 254 ;
   double*  pd1  = ss0 + 272 ;
   double*  dd0  = ss0 + 308 ;
   double*  dd1  = ss0 + 344 ;
   double*  fp0  = ss0 + 416 ;
   double*  pf0  = ss0 + 536 ;
   double*  pf1  = ss0 + 566 ;
   double*  pg0  = ss0 + 596 ;
   double*  fd0  = ss0 + 641 ;
   double*  df0  = ss0 + 821 ;
   double*  df1  = ss0 + 881 ;
   double*  ff0  = ss0 + 941 ;
   double*  dg0  = ss0 + 1141 ;
   double*  fg0  = ss0 + 1231 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp1  = ss0 + 135 ;
   double*  pp0  = ss0 + 147 ;
   double*  pp1  = ss0 + 156 ;
   double*  sd1  = ss0 + 192 ;
   double*  sf1  = ss0 + 210 ;
   double*  pd0  = ss0 + 230 ;
   double*  pd1  = ss0 + 248 ;
   double*  dp0  = ss0 + 302 ;
   double*  dp1  = ss0 + 320 ;
   double*  sg1  = ss0 + 392 ;
   double*  fp0  = ss0 + 407 ;
   double*  dd0  = ss0 + 557 ;
   double*  dd1  = ss0 + 593 ;
   double*  pf0  = ss0 + 701 ;
   double*  pf1  = ss0 + 731 ;
   double*  pg0  = ss0 + 791 ;
   double*  pg1  = ss0 + 836 ;
   double*  df0  = ss0 + 881 ;
   double*  df1  = ss0 + 941 ;
   double*  fd0  = ss0 + 1061 ;
   double*  ph0  = ss0 + 1301 ;
   double*  dg0  = ss0 + 1364 ;
   double*  dg1  = ss0 + 1454 ;
   double*  ff0  = ss0 + 1544 ;
   double*  dh0  = ss0 + 1844 ;
   double*  fg0  = ss0 + 1970 ;
   double*  fh0  = ss0 + 2270 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pdff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp1  = ss0 + 155 ;
   double*  sd1  = ss0 + 170 ;
   double*  pp0  = ss0 + 194 ;
   double*  pp1  = ss0 + 203 ;
   double*  sf1  = ss0 + 248 ;
   double*  dp0  = ss0 + 278 ;
   double*  dp1  = ss0 + 296 ;
   double*  pd0  = ss0 + 386 ;
   double*  pd1  = ss0 + 404 ;
   double*  fp0  = ss0 + 476 ;
   double*  sg1  = ss0 + 656 ;
   double*  dd0  = ss0 + 686 ;
   double*  dd1  = ss0 + 722 ;
   double*  pf0  = ss0 + 866 ;
   double*  pf1  = ss0 + 896 ;
   double*  sh1  = ss0 + 986 ;
   double*  pg0  = ss0 + 1007 ;
   double*  pg1  = ss0 + 1052 ;
   double*  fd0  = ss0 + 1142 ;
   double*  df0  = ss0 + 1442 ;
   double*  df1  = ss0 + 1502 ;
   double*  dg0  = ss0 + 1682 ;
   double*  dg1  = ss0 + 1772 ;
   double*  ph0  = ss0 + 1952 ;
   double*  ph1  = ss0 + 2015 ;
   double*  ff0  = ss0 + 2078 ;
   double*  pi0  = ss0 + 2478 ;
   double*  dh0  = ss0 + 2562 ;
   double*  dh1  = ss0 + 2688 ;
   double*  fg0  = ss0 + 2814 ;
   double*  di0  = ss0 + 3264 ;
   double*  fh0  = ss0 + 3432 ;
   double*  fi0  = ss0 + 3852 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  fs0  = ss0 + 35 ;
   double*  gs0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( gs0 , fs0 , ds0 , PA_WP, zab );
}

template< int VTS >
__device__ void pfsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  pp0  = ss0 + 105 ;
   double*  dp0  = ss0 + 114 ;
   double*  fp0  = ss0 + 132 ;
   double*  gp0  = ss0 + 162 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void pfsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  sp1  = ss0 + 140 ;
   double*  pp0  = ss0 + 143 ;
   double*  pp1  = ss0 + 152 ;
   double*  pd0  = ss0 + 161 ;
   double*  dp0  = ss0 + 179 ;
   double*  dp1  = ss0 + 197 ;
   double*  dd0  = ss0 + 215 ;
   double*  fp0  = ss0 + 251 ;
   double*  fp1  = ss0 + 281 ;
   double*  fd0  = ss0 + 311 ;
   double*  gp0  = ss0 + 371 ;
   double*  gd0  = ss0 + 461 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  sd1  = ss0 + 181 ;
   double*  pp0  = ss0 + 187 ;
   double*  pp1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  pd0  = ss0 + 268 ;
   double*  pd1  = ss0 + 286 ;
   double*  pf0  = ss0 + 304 ;
   double*  fp0  = ss0 + 334 ;
   double*  fp1  = ss0 + 364 ;
   double*  dd0  = ss0 + 424 ;
   double*  dd1  = ss0 + 460 ;
   double*  df0  = ss0 + 496 ;
   double*  gp0  = ss0 + 556 ;
   double*  fd0  = ss0 + 691 ;
   double*  fd1  = ss0 + 751 ;
   double*  gd0  = ss0 + 811 ;
   double*  ff0  = ss0 + 991 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  pp0  = ss0 + 105 ;
   double*  dp0  = ss0 + 114 ;
   double*  fp0  = ss0 + 132 ;
   double*  gp0  = ss0 + 162 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void pfpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  sp1  = ss0 + 140 ;
   double*  pp0  = ss0 + 143 ;
   double*  pp1  = ss0 + 152 ;
   double*  pd0  = ss0 + 161 ;
   double*  dp0  = ss0 + 179 ;
   double*  dp1  = ss0 + 197 ;
   double*  dd0  = ss0 + 215 ;
   double*  fp0  = ss0 + 251 ;
   double*  fp1  = ss0 + 281 ;
   double*  gp0  = ss0 + 311 ;
   double*  fd0  = ss0 + 401 ;
   double*  gd0  = ss0 + 461 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  sd1  = ss0 + 181 ;
   double*  pp0  = ss0 + 187 ;
   double*  pp1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  pd0  = ss0 + 268 ;
   double*  pd1  = ss0 + 286 ;
   double*  pf0  = ss0 + 304 ;
   double*  fp0  = ss0 + 334 ;
   double*  fp1  = ss0 + 364 ;
   double*  dd0  = ss0 + 424 ;
   double*  dd1  = ss0 + 460 ;
   double*  gp0  = ss0 + 496 ;
   double*  fd0  = ss0 + 631 ;
   double*  fd1  = ss0 + 691 ;
   double*  df0  = ss0 + 751 ;
   double*  gd0  = ss0 + 811 ;
   double*  ff0  = ss0 + 991 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp1  = ss0 + 210 ;
   double*  sd1  = ss0 + 219 ;
   double*  pp0  = ss0 + 231 ;
   double*  pp1  = ss0 + 240 ;
   double*  sf1  = ss0 + 267 ;
   double*  dp0  = ss0 + 277 ;
   double*  dp1  = ss0 + 295 ;
   double*  pd0  = ss0 + 349 ;
   double*  pd1  = ss0 + 367 ;
   double*  pf0  = ss0 + 403 ;
   double*  pf1  = ss0 + 433 ;
   double*  fp0  = ss0 + 463 ;
   double*  fp1  = ss0 + 493 ;
   double*  dd0  = ss0 + 583 ;
   double*  dd1  = ss0 + 619 ;
   double*  pg0  = ss0 + 691 ;
   double*  df0  = ss0 + 736 ;
   double*  df1  = ss0 + 796 ;
   double*  gp0  = ss0 + 856 ;
   double*  fd0  = ss0 + 1036 ;
   double*  fd1  = ss0 + 1096 ;
   double*  dg0  = ss0 + 1216 ;
   double*  gd0  = ss0 + 1306 ;
   double*  ff0  = ss0 + 1576 ;
   double*  ff1  = ss0 + 1676 ;
   double*  gf0  = ss0 + 1776 ;
   double*  fg0  = ss0 + 2076 ;
   double*  gg0  = ss0 + 2226 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  sp1  = ss0 + 140 ;
   double*  pp0  = ss0 + 143 ;
   double*  pp1  = ss0 + 152 ;
   double*  dp0  = ss0 + 161 ;
   double*  dp1  = ss0 + 179 ;
   double*  pd0  = ss0 + 197 ;
   double*  dd0  = ss0 + 215 ;
   double*  fp0  = ss0 + 251 ;
   double*  fp1  = ss0 + 281 ;
   double*  gp0  = ss0 + 311 ;
   double*  fd0  = ss0 + 401 ;
   double*  gd0  = ss0 + 461 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  sd1  = ss0 + 181 ;
   double*  pp0  = ss0 + 187 ;
   double*  pp1  = ss0 + 196 ;
   double*  pd0  = ss0 + 214 ;
   double*  pd1  = ss0 + 232 ;
   double*  dp0  = ss0 + 250 ;
   double*  dp1  = ss0 + 268 ;
   double*  pf0  = ss0 + 304 ;
   double*  dd0  = ss0 + 334 ;
   double*  dd1  = ss0 + 370 ;
   double*  fp0  = ss0 + 406 ;
   double*  fp1  = ss0 + 436 ;
   double*  df0  = ss0 + 496 ;
   double*  fd0  = ss0 + 556 ;
   double*  fd1  = ss0 + 616 ;
   double*  gp0  = ss0 + 676 ;
   double*  ff0  = ss0 + 811 ;
   double*  gd0  = ss0 + 911 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp1  = ss0 + 210 ;
   double*  sd1  = ss0 + 219 ;
   double*  pp0  = ss0 + 231 ;
   double*  pp1  = ss0 + 240 ;
   double*  sf1  = ss0 + 267 ;
   double*  pd0  = ss0 + 277 ;
   double*  pd1  = ss0 + 295 ;
   double*  dp0  = ss0 + 331 ;
   double*  dp1  = ss0 + 349 ;
   double*  pf0  = ss0 + 403 ;
   double*  pf1  = ss0 + 433 ;
   double*  fp0  = ss0 + 463 ;
   double*  fp1  = ss0 + 493 ;
   double*  dd0  = ss0 + 583 ;
   double*  dd1  = ss0 + 619 ;
   double*  pg0  = ss0 + 691 ;
   double*  gp0  = ss0 + 736 ;
   double*  fd0  = ss0 + 916 ;
   double*  fd1  = ss0 + 976 ;
   double*  df0  = ss0 + 1096 ;
   double*  df1  = ss0 + 1156 ;
   double*  dg0  = ss0 + 1216 ;
   double*  gd0  = ss0 + 1306 ;
   double*  ff0  = ss0 + 1576 ;
   double*  ff1  = ss0 + 1676 ;
   double*  gf0  = ss0 + 1776 ;
   double*  fg0  = ss0 + 2076 ;
   double*  gg0  = ss0 + 2226 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp1  = ss0 + 245 ;
   double*  pp0  = ss0 + 257 ;
   double*  pp1  = ss0 + 266 ;
   double*  sd1  = ss0 + 302 ;
   double*  dp0  = ss0 + 320 ;
   double*  dp1  = ss0 + 338 ;
   double*  pd0  = ss0 + 410 ;
   double*  pd1  = ss0 + 428 ;
   double*  sf1  = ss0 + 482 ;
   double*  pf0  = ss0 + 502 ;
   double*  pf1  = ss0 + 532 ;
   double*  sg1  = ss0 + 592 ;
   double*  dd0  = ss0 + 607 ;
   double*  dd1  = ss0 + 643 ;
   double*  fp0  = ss0 + 751 ;
   double*  fp1  = ss0 + 781 ;
   double*  df0  = ss0 + 901 ;
   double*  df1  = ss0 + 961 ;
   double*  fd0  = ss0 + 1081 ;
   double*  fd1  = ss0 + 1141 ;
   double*  gp0  = ss0 + 1321 ;
   double*  pg0  = ss0 + 1546 ;
   double*  pg1  = ss0 + 1591 ;
   double*  gd0  = ss0 + 1636 ;
   double*  dg0  = ss0 + 1996 ;
   double*  dg1  = ss0 + 2086 ;
   double*  ff0  = ss0 + 2176 ;
   double*  ff1  = ss0 + 2276 ;
   double*  ph0  = ss0 + 2476 ;
   double*  gf0  = ss0 + 2539 ;
   double*  fg0  = ss0 + 2989 ;
   double*  fg1  = ss0 + 3139 ;
   double*  dh0  = ss0 + 3289 ;
   double*  gg0  = ss0 + 3415 ;
   double*  fh0  = ss0 + 3865 ;
   double*  gh0  = ss0 + 4075 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp1  = ss0 + 175 ;
   double*  pp0  = ss0 + 181 ;
   double*  pp1  = ss0 + 190 ;
   double*  sd1  = ss0 + 208 ;
   double*  pd0  = ss0 + 214 ;
   double*  pd1  = ss0 + 232 ;
   double*  dp0  = ss0 + 250 ;
   double*  dp1  = ss0 + 268 ;
   double*  dd0  = ss0 + 304 ;
   double*  dd1  = ss0 + 340 ;
   double*  fp0  = ss0 + 376 ;
   double*  fp1  = ss0 + 406 ;
   double*  pf0  = ss0 + 466 ;
   double*  df0  = ss0 + 496 ;
   double*  fd0  = ss0 + 556 ;
   double*  fd1  = ss0 + 616 ;
   double*  gp0  = ss0 + 676 ;
   double*  ff0  = ss0 + 811 ;
   double*  gd0  = ss0 + 911 ;
   double*  gf0  = ss0 + 1091 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp0 , ps0 , ss1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp1  = ss0 + 210 ;
   double*  sd1  = ss0 + 219 ;
   double*  pp0  = ss0 + 231 ;
   double*  pp1  = ss0 + 240 ;
   double*  sf1  = ss0 + 267 ;
   double*  pd0  = ss0 + 277 ;
   double*  pd1  = ss0 + 295 ;
   double*  dp0  = ss0 + 331 ;
   double*  dp1  = ss0 + 349 ;
   double*  pf0  = ss0 + 403 ;
   double*  pf1  = ss0 + 433 ;
   double*  fp0  = ss0 + 463 ;
   double*  fp1  = ss0 + 493 ;
   double*  dd0  = ss0 + 583 ;
   double*  dd1  = ss0 + 619 ;
   double*  pg0  = ss0 + 691 ;
   double*  fd0  = ss0 + 736 ;
   double*  fd1  = ss0 + 796 ;
   double*  gp0  = ss0 + 916 ;
   double*  df0  = ss0 + 1096 ;
   double*  df1  = ss0 + 1156 ;
   double*  dg0  = ss0 + 1216 ;
   double*  gd0  = ss0 + 1306 ;
   double*  ff0  = ss0 + 1576 ;
   double*  ff1  = ss0 + 1676 ;
   double*  gf0  = ss0 + 1776 ;
   double*  fg0  = ss0 + 2076 ;
   double*  gg0  = ss0 + 2226 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp1  = ss0 + 245 ;
   double*  sd1  = ss0 + 257 ;
   double*  pp0  = ss0 + 275 ;
   double*  pp1  = ss0 + 284 ;
   double*  sf1  = ss0 + 320 ;
   double*  dp0  = ss0 + 340 ;
   double*  dp1  = ss0 + 358 ;
   double*  pd0  = ss0 + 430 ;
   double*  pd1  = ss0 + 448 ;
   double*  fp0  = ss0 + 502 ;
   double*  fp1  = ss0 + 532 ;
   double*  sg1  = ss0 + 652 ;
   double*  dd0  = ss0 + 667 ;
   double*  dd1  = ss0 + 703 ;
   double*  pf0  = ss0 + 811 ;
   double*  pf1  = ss0 + 841 ;
   double*  pg0  = ss0 + 901 ;
   double*  pg1  = ss0 + 946 ;
   double*  fd0  = ss0 + 991 ;
   double*  fd1  = ss0 + 1051 ;
   double*  df0  = ss0 + 1231 ;
   double*  df1  = ss0 + 1291 ;
   double*  gp0  = ss0 + 1411 ;
   double*  ph0  = ss0 + 1636 ;
   double*  dg0  = ss0 + 1699 ;
   double*  dg1  = ss0 + 1789 ;
   double*  ff0  = ss0 + 1879 ;
   double*  ff1  = ss0 + 1979 ;
   double*  gd0  = ss0 + 2179 ;
   double*  dh0  = ss0 + 2539 ;
   double*  fg0  = ss0 + 2665 ;
   double*  fg1  = ss0 + 2815 ;
   double*  gf0  = ss0 + 2965 ;
   double*  fh0  = ss0 + 3415 ;
   double*  gg0  = ss0 + 3625 ;
   double*  gh0  = ss0 + 4075 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void pfff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss1  = ss0 + 1 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  sp1  = ss0 + 280 ;
   double*  sd1  = ss0 + 295 ;
   double*  pp0  = ss0 + 319 ;
   double*  pp1  = ss0 + 328 ;
   double*  dp0  = ss0 + 373 ;
   double*  dp1  = ss0 + 391 ;
   double*  pd0  = ss0 + 481 ;
   double*  pd1  = ss0 + 499 ;
   double*  sf1  = ss0 + 571 ;
   double*  sg1  = ss0 + 601 ;
   double*  pf0  = ss0 + 631 ;
   double*  pf1  = ss0 + 661 ;
   double*  fp0  = ss0 + 751 ;
   double*  fp1  = ss0 + 781 ;
   double*  dd0  = ss0 + 931 ;
   double*  dd1  = ss0 + 967 ;
   double*  sh1  = ss0 + 1111 ;
   double*  df0  = ss0 + 1132 ;
   double*  df1  = ss0 + 1192 ;
   double*  fd0  = ss0 + 1372 ;
   double*  fd1  = ss0 + 1432 ;
   double*  pg0  = ss0 + 1672 ;
   double*  pg1  = ss0 + 1717 ;
   double*  gp0  = ss0 + 1807 ;
   double*  dg0  = ss0 + 2077 ;
   double*  dg1  = ss0 + 2167 ;
   double*  ff0  = ss0 + 2347 ;
   double*  ff1  = ss0 + 2447 ;
   double*  ph0  = ss0 + 2747 ;
   double*  ph1  = ss0 + 2810 ;
   double*  gd0  = ss0 + 2873 ;
   double*  pi0  = ss0 + 3323 ;
   double*  gf0  = ss0 + 3407 ;
   double*  dh0  = ss0 + 4007 ;
   double*  dh1  = ss0 + 4133 ;
   double*  fg0  = ss0 + 4259 ;
   double*  fg1  = ss0 + 4409 ;
   double*  gg0  = ss0 + 4709 ;
   double*  fh0  = ss0 + 5384 ;
   double*  fh1  = ss0 + 5594 ;
   double*  di0  = ss0 + 5804 ;
   double*  gh0  = ss0 + 5972 ;
   double*  fi0  = ss0 + 6602 ;
   double*  gi0  = ss0 + 6882 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 5 >( sp1 , ss1 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( sd1 , sp1 , ss1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 6 >( pp0 , ps0 , ss1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 5 >( pd0 , pp0 , ps0 , sp1 , QC_WQ, zcd );
   execute_VRR2_gpu< VTS, 3, 0, 3 >( sf1 , sd1 , sp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( sg1 , sf1 , sd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 4 >( pf0 , pd0 , pp0 , sd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( sh1 , sg1 , sf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 3 >( pg0 , pf0 , pd0 , sf1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 2 >( ph0 , pg0 , pf0 , sg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 6, 1 >( pi0 , ph0 , pg0 , sh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 3 ;
   double*  ds0  = ss0 + 9 ;
   execute_VRR1_gpu< VTS, 1, 0, 2 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( ds0 , ps0 , ss0 , PA_WP, zab );
}

template< int VTS >
__device__ void dssp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  dp0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void dssd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  pp1  = ss0 + 35 ;
   double*  dp0  = ss0 + 44 ;
   double*  dd0  = ss0 + 80 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dssf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  dp0  = ss0 + 66 ;
   double*  pd1  = ss0 + 120 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 4 ;
   double*  ps1  = ss0 + 7 ;
   double*  ds0  = ss0 + 13 ;
   double*  dp0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
}

template< int VTS >
__device__ void dspp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  pp1  = ss0 + 35 ;
   double*  dp0  = ss0 + 44 ;
   double*  dd0  = ss0 + 80 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dspd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  pd1  = ss0 + 66 ;
   double*  dp0  = ss0 + 84 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dspf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp2  = ss0 + 55 ;
   double*  sd2  = ss0 + 61 ;
   double*  pp1  = ss0 + 67 ;
   double*  pd1  = ss0 + 94 ;
   double*  dp0  = ss0 + 130 ;
   double*  pf1  = ss0 + 202 ;
   double*  dd0  = ss0 + 232 ;
   double*  df0  = ss0 + 340 ;
   double*  dg0  = ss0 + 460 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  pp1  = ss0 + 35 ;
   double*  dp0  = ss0 + 44 ;
   double*  dd0  = ss0 + 80 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  pd1  = ss0 + 66 ;
   double*  dp0  = ss0 + 84 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp2  = ss0 + 55 ;
   double*  sd2  = ss0 + 61 ;
   double*  pp1  = ss0 + 67 ;
   double*  pd1  = ss0 + 94 ;
   double*  dp0  = ss0 + 130 ;
   double*  pf1  = ss0 + 202 ;
   double*  dd0  = ss0 + 232 ;
   double*  df0  = ss0 + 340 ;
   double*  dg0  = ss0 + 460 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp2  = ss0 + 65 ;
   double*  sd2  = ss0 + 74 ;
   double*  pp1  = ss0 + 86 ;
   double*  sf2  = ss0 + 122 ;
   double*  pd1  = ss0 + 132 ;
   double*  dp0  = ss0 + 186 ;
   double*  pf1  = ss0 + 276 ;
   double*  dd0  = ss0 + 336 ;
   double*  pg1  = ss0 + 480 ;
   double*  df0  = ss0 + 525 ;
   double*  dg0  = ss0 + 705 ;
   double*  dh0  = ss0 + 885 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  sp2  = ss0 + 45 ;
   double*  pp1  = ss0 + 48 ;
   double*  pd1  = ss0 + 66 ;
   double*  dp0  = ss0 + 84 ;
   double*  dd0  = ss0 + 138 ;
   double*  df0  = ss0 + 210 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  sp2  = ss0 + 55 ;
   double*  sd2  = ss0 + 61 ;
   double*  pp1  = ss0 + 67 ;
   double*  pd1  = ss0 + 94 ;
   double*  dp0  = ss0 + 130 ;
   double*  pf1  = ss0 + 202 ;
   double*  dd0  = ss0 + 232 ;
   double*  df0  = ss0 + 340 ;
   double*  dg0  = ss0 + 460 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  sp2  = ss0 + 65 ;
   double*  sd2  = ss0 + 74 ;
   double*  pp1  = ss0 + 86 ;
   double*  sf2  = ss0 + 122 ;
   double*  pd1  = ss0 + 132 ;
   double*  dp0  = ss0 + 186 ;
   double*  pf1  = ss0 + 276 ;
   double*  dd0  = ss0 + 336 ;
   double*  pg1  = ss0 + 480 ;
   double*  df0  = ss0 + 525 ;
   double*  dg0  = ss0 + 705 ;
   double*  dh0  = ss0 + 885 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dsff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  sp2  = ss0 + 75 ;
   double*  sd2  = ss0 + 87 ;
   double*  pp1  = ss0 + 105 ;
   double*  sf2  = ss0 + 150 ;
   double*  dp0  = ss0 + 170 ;
   double*  pd1  = ss0 + 278 ;
   double*  sg2  = ss0 + 350 ;
   double*  dd0  = ss0 + 365 ;
   double*  pf1  = ss0 + 545 ;
   double*  pg1  = ss0 + 635 ;
   double*  df0  = ss0 + 725 ;
   double*  ph1  = ss0 + 965 ;
   double*  dg0  = ss0 + 1028 ;
   double*  dh0  = ss0 + 1298 ;
   double*  di0  = ss0 + 1550 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void dpsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  dp0  = ss0 + 55 ;
   double*  fp0  = ss0 + 73 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void dpsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  pp1  = ss0 + 75 ;
   double*  dp0  = ss0 + 84 ;
   double*  dp1  = ss0 + 102 ;
   double*  dd0  = ss0 + 120 ;
   double*  fp0  = ss0 + 156 ;
   double*  fd0  = ss0 + 216 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ps1  = ss0 + 8 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  dp0  = ss0 + 55 ;
   double*  fp0  = ss0 + 73 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void dppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  pp1  = ss0 + 75 ;
   double*  dp0  = ss0 + 84 ;
   double*  dp1  = ss0 + 102 ;
   double*  dd0  = ss0 + 120 ;
   double*  fp0  = ss0 + 156 ;
   double*  fd0  = ss0 + 216 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp2  = ss0 + 115 ;
   double*  pp1  = ss0 + 121 ;
   double*  sd2  = ss0 + 148 ;
   double*  dp0  = ss0 + 154 ;
   double*  dp1  = ss0 + 172 ;
   double*  pd1  = ss0 + 226 ;
   double*  fp0  = ss0 + 262 ;
   double*  dd0  = ss0 + 382 ;
   double*  dd1  = ss0 + 418 ;
   double*  pf1  = ss0 + 490 ;
   double*  fd0  = ss0 + 520 ;
   double*  df0  = ss0 + 700 ;
   double*  df1  = ss0 + 760 ;
   double*  ff0  = ss0 + 820 ;
   double*  dg0  = ss0 + 1020 ;
   double*  fg0  = ss0 + 1110 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  pp1  = ss0 + 75 ;
   double*  dp0  = ss0 + 84 ;
   double*  dp1  = ss0 + 102 ;
   double*  dd0  = ss0 + 120 ;
   double*  fp0  = ss0 + 156 ;
   double*  fd0  = ss0 + 216 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp2  = ss0 + 115 ;
   double*  sd2  = ss0 + 121 ;
   double*  pp1  = ss0 + 127 ;
   double*  pd1  = ss0 + 154 ;
   double*  dp0  = ss0 + 190 ;
   double*  dp1  = ss0 + 208 ;
   double*  pf1  = ss0 + 262 ;
   double*  dd0  = ss0 + 292 ;
   double*  dd1  = ss0 + 328 ;
   double*  fp0  = ss0 + 400 ;
   double*  df0  = ss0 + 520 ;
   double*  df1  = ss0 + 580 ;
   double*  fd0  = ss0 + 640 ;
   double*  dg0  = ss0 + 820 ;
   double*  ff0  = ss0 + 910 ;
   double*  fg0  = ss0 + 1110 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp2  = ss0 + 135 ;
   double*  pp1  = ss0 + 144 ;
   double*  sd2  = ss0 + 180 ;
   double*  sf2  = ss0 + 192 ;
   double*  dp0  = ss0 + 202 ;
   double*  dp1  = ss0 + 220 ;
   double*  pd1  = ss0 + 292 ;
   double*  fp0  = ss0 + 346 ;
   double*  dd0  = ss0 + 496 ;
   double*  dd1  = ss0 + 532 ;
   double*  pf1  = ss0 + 640 ;
   double*  pg1  = ss0 + 700 ;
   double*  df0  = ss0 + 745 ;
   double*  df1  = ss0 + 805 ;
   double*  fd0  = ss0 + 925 ;
   double*  dg0  = ss0 + 1165 ;
   double*  dg1  = ss0 + 1255 ;
   double*  ff0  = ss0 + 1345 ;
   double*  dh0  = ss0 + 1645 ;
   double*  fg0  = ss0 + 1771 ;
   double*  fh0  = ss0 + 2071 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  sp2  = ss0 + 95 ;
   double*  pp1  = ss0 + 98 ;
   double*  pd1  = ss0 + 116 ;
   double*  dp0  = ss0 + 134 ;
   double*  dp1  = ss0 + 152 ;
   double*  dd0  = ss0 + 188 ;
   double*  dd1  = ss0 + 224 ;
   double*  fp0  = ss0 + 260 ;
   double*  df0  = ss0 + 350 ;
   double*  fd0  = ss0 + 410 ;
   double*  ff0  = ss0 + 530 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp2  = ss0 + 115 ;
   double*  sd2  = ss0 + 121 ;
   double*  pp1  = ss0 + 127 ;
   double*  pd1  = ss0 + 154 ;
   double*  dp0  = ss0 + 190 ;
   double*  dp1  = ss0 + 208 ;
   double*  pf1  = ss0 + 262 ;
   double*  fp0  = ss0 + 292 ;
   double*  dd0  = ss0 + 412 ;
   double*  dd1  = ss0 + 448 ;
   double*  df0  = ss0 + 520 ;
   double*  df1  = ss0 + 580 ;
   double*  fd0  = ss0 + 640 ;
   double*  dg0  = ss0 + 820 ;
   double*  ff0  = ss0 + 910 ;
   double*  fg0  = ss0 + 1110 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp2  = ss0 + 135 ;
   double*  pp1  = ss0 + 144 ;
   double*  sd2  = ss0 + 180 ;
   double*  sf2  = ss0 + 192 ;
   double*  pd1  = ss0 + 202 ;
   double*  dp0  = ss0 + 256 ;
   double*  dp1  = ss0 + 274 ;
   double*  fp0  = ss0 + 346 ;
   double*  dd0  = ss0 + 496 ;
   double*  dd1  = ss0 + 532 ;
   double*  pf1  = ss0 + 640 ;
   double*  pg1  = ss0 + 700 ;
   double*  df0  = ss0 + 745 ;
   double*  df1  = ss0 + 805 ;
   double*  fd0  = ss0 + 925 ;
   double*  dg0  = ss0 + 1165 ;
   double*  dg1  = ss0 + 1255 ;
   double*  ff0  = ss0 + 1345 ;
   double*  dh0  = ss0 + 1645 ;
   double*  fg0  = ss0 + 1771 ;
   double*  fh0  = ss0 + 2071 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dpff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp2  = ss0 + 155 ;
   double*  sd2  = ss0 + 167 ;
   double*  pp1  = ss0 + 185 ;
   double*  sf2  = ss0 + 230 ;
   double*  dp0  = ss0 + 250 ;
   double*  dp1  = ss0 + 268 ;
   double*  pd1  = ss0 + 358 ;
   double*  fp0  = ss0 + 430 ;
   double*  sg2  = ss0 + 610 ;
   double*  dd0  = ss0 + 625 ;
   double*  dd1  = ss0 + 661 ;
   double*  pf1  = ss0 + 805 ;
   double*  pg1  = ss0 + 895 ;
   double*  fd0  = ss0 + 985 ;
   double*  df0  = ss0 + 1285 ;
   double*  df1  = ss0 + 1345 ;
   double*  dg0  = ss0 + 1525 ;
   double*  dg1  = ss0 + 1615 ;
   double*  ph1  = ss0 + 1795 ;
   double*  ff0  = ss0 + 1858 ;
   double*  dh0  = ss0 + 2258 ;
   double*  dh1  = ss0 + 2384 ;
   double*  fg0  = ss0 + 2510 ;
   double*  di0  = ss0 + 2960 ;
   double*  fh0  = ss0 + 3128 ;
   double*  fi0  = ss0 + 3548 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  fs0  = ss0 + 35 ;
   double*  gs0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( gs0 , fs0 , ds0 , PA_WP, zab );
}

template< int VTS >
__device__ void ddsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  dp0  = ss0 + 105 ;
   double*  fp0  = ss0 + 123 ;
   double*  gp0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ddsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  pp1  = ss0 + 140 ;
   double*  dp0  = ss0 + 149 ;
   double*  dp1  = ss0 + 167 ;
   double*  dd0  = ss0 + 185 ;
   double*  fp0  = ss0 + 221 ;
   double*  fp1  = ss0 + 251 ;
   double*  fd0  = ss0 + 281 ;
   double*  gp0  = ss0 + 341 ;
   double*  gd0  = ss0 + 431 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  dp0  = ss0 + 196 ;
   double*  dp1  = ss0 + 214 ;
   double*  pd1  = ss0 + 250 ;
   double*  fp0  = ss0 + 268 ;
   double*  fp1  = ss0 + 298 ;
   double*  dd0  = ss0 + 358 ;
   double*  dd1  = ss0 + 394 ;
   double*  gp0  = ss0 + 430 ;
   double*  fd0  = ss0 + 565 ;
   double*  fd1  = ss0 + 625 ;
   double*  df0  = ss0 + 685 ;
   double*  gd0  = ss0 + 745 ;
   double*  ff0  = ss0 + 925 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps1  = ss0 + 9 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  dp0  = ss0 + 105 ;
   double*  fp0  = ss0 + 123 ;
   double*  gp0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ddpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  pp1  = ss0 + 140 ;
   double*  dp0  = ss0 + 149 ;
   double*  dp1  = ss0 + 167 ;
   double*  dd0  = ss0 + 185 ;
   double*  fp0  = ss0 + 221 ;
   double*  fp1  = ss0 + 251 ;
   double*  gp0  = ss0 + 281 ;
   double*  fd0  = ss0 + 371 ;
   double*  gd0  = ss0 + 431 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  pd1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  dd0  = ss0 + 268 ;
   double*  dd1  = ss0 + 304 ;
   double*  fp0  = ss0 + 340 ;
   double*  fp1  = ss0 + 370 ;
   double*  df0  = ss0 + 430 ;
   double*  fd0  = ss0 + 490 ;
   double*  fd1  = ss0 + 550 ;
   double*  gp0  = ss0 + 610 ;
   double*  ff0  = ss0 + 745 ;
   double*  gd0  = ss0 + 845 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp2  = ss0 + 210 ;
   double*  sd2  = ss0 + 216 ;
   double*  pp1  = ss0 + 222 ;
   double*  dp0  = ss0 + 249 ;
   double*  dp1  = ss0 + 267 ;
   double*  pd1  = ss0 + 321 ;
   double*  pf1  = ss0 + 357 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 507 ;
   double*  dd1  = ss0 + 543 ;
   double*  df0  = ss0 + 615 ;
   double*  df1  = ss0 + 675 ;
   double*  gp0  = ss0 + 735 ;
   double*  fd0  = ss0 + 915 ;
   double*  fd1  = ss0 + 975 ;
   double*  dg0  = ss0 + 1095 ;
   double*  gd0  = ss0 + 1185 ;
   double*  ff0  = ss0 + 1455 ;
   double*  ff1  = ss0 + 1555 ;
   double*  gf0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 1955 ;
   double*  gg0  = ss0 + 2105 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  pp1  = ss0 + 140 ;
   double*  dp0  = ss0 + 149 ;
   double*  dp1  = ss0 + 167 ;
   double*  fp0  = ss0 + 185 ;
   double*  fp1  = ss0 + 215 ;
   double*  dd0  = ss0 + 245 ;
   double*  gp0  = ss0 + 281 ;
   double*  fd0  = ss0 + 371 ;
   double*  gd0  = ss0 + 431 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  pd1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  dd0  = ss0 + 268 ;
   double*  dd1  = ss0 + 304 ;
   double*  fp0  = ss0 + 340 ;
   double*  fp1  = ss0 + 370 ;
   double*  df0  = ss0 + 430 ;
   double*  fd0  = ss0 + 490 ;
   double*  fd1  = ss0 + 550 ;
   double*  gp0  = ss0 + 610 ;
   double*  ff0  = ss0 + 745 ;
   double*  gd0  = ss0 + 845 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp2  = ss0 + 210 ;
   double*  sd2  = ss0 + 216 ;
   double*  pp1  = ss0 + 222 ;
   double*  pd1  = ss0 + 249 ;
   double*  dp0  = ss0 + 285 ;
   double*  dp1  = ss0 + 303 ;
   double*  pf1  = ss0 + 357 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 507 ;
   double*  dd1  = ss0 + 543 ;
   double*  gp0  = ss0 + 615 ;
   double*  fd0  = ss0 + 795 ;
   double*  fd1  = ss0 + 855 ;
   double*  df0  = ss0 + 975 ;
   double*  df1  = ss0 + 1035 ;
   double*  dg0  = ss0 + 1095 ;
   double*  gd0  = ss0 + 1185 ;
   double*  ff0  = ss0 + 1455 ;
   double*  ff1  = ss0 + 1555 ;
   double*  gf0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 1955 ;
   double*  gg0  = ss0 + 2105 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp2  = ss0 + 245 ;
   double*  sd2  = ss0 + 254 ;
   double*  pp1  = ss0 + 266 ;
   double*  sf2  = ss0 + 302 ;
   double*  dp0  = ss0 + 312 ;
   double*  dp1  = ss0 + 330 ;
   double*  pd1  = ss0 + 402 ;
   double*  fp0  = ss0 + 456 ;
   double*  fp1  = ss0 + 486 ;
   double*  dd0  = ss0 + 606 ;
   double*  dd1  = ss0 + 642 ;
   double*  pf1  = ss0 + 750 ;
   double*  pg1  = ss0 + 810 ;
   double*  fd0  = ss0 + 855 ;
   double*  fd1  = ss0 + 915 ;
   double*  df0  = ss0 + 1095 ;
   double*  df1  = ss0 + 1155 ;
   double*  gp0  = ss0 + 1275 ;
   double*  dg0  = ss0 + 1500 ;
   double*  dg1  = ss0 + 1590 ;
   double*  ff0  = ss0 + 1680 ;
   double*  ff1  = ss0 + 1780 ;
   double*  gd0  = ss0 + 1980 ;
   double*  dh0  = ss0 + 2340 ;
   double*  fg0  = ss0 + 2466 ;
   double*  fg1  = ss0 + 2616 ;
   double*  gf0  = ss0 + 2766 ;
   double*  fh0  = ss0 + 3216 ;
   double*  gg0  = ss0 + 3426 ;
   double*  gh0  = ss0 + 3876 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  sp2  = ss0 + 175 ;
   double*  pp1  = ss0 + 178 ;
   double*  pd1  = ss0 + 196 ;
   double*  dp0  = ss0 + 214 ;
   double*  dp1  = ss0 + 232 ;
   double*  dd0  = ss0 + 268 ;
   double*  dd1  = ss0 + 304 ;
   double*  fp0  = ss0 + 340 ;
   double*  fp1  = ss0 + 370 ;
   double*  df0  = ss0 + 430 ;
   double*  fd0  = ss0 + 490 ;
   double*  fd1  = ss0 + 550 ;
   double*  gp0  = ss0 + 610 ;
   double*  ff0  = ss0 + 745 ;
   double*  gd0  = ss0 + 845 ;
   double*  gf0  = ss0 + 1025 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp2  = ss0 + 210 ;
   double*  sd2  = ss0 + 216 ;
   double*  pp1  = ss0 + 222 ;
   double*  dp0  = ss0 + 249 ;
   double*  dp1  = ss0 + 267 ;
   double*  pd1  = ss0 + 321 ;
   double*  pf1  = ss0 + 357 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 507 ;
   double*  dd1  = ss0 + 543 ;
   double*  fd0  = ss0 + 615 ;
   double*  fd1  = ss0 + 675 ;
   double*  gp0  = ss0 + 795 ;
   double*  df0  = ss0 + 975 ;
   double*  df1  = ss0 + 1035 ;
   double*  dg0  = ss0 + 1095 ;
   double*  gd0  = ss0 + 1185 ;
   double*  ff0  = ss0 + 1455 ;
   double*  ff1  = ss0 + 1555 ;
   double*  gf0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 1955 ;
   double*  gg0  = ss0 + 2105 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp2  = ss0 + 245 ;
   double*  sd2  = ss0 + 254 ;
   double*  pp1  = ss0 + 266 ;
   double*  sf2  = ss0 + 302 ;
   double*  dp0  = ss0 + 312 ;
   double*  dp1  = ss0 + 330 ;
   double*  pd1  = ss0 + 402 ;
   double*  fp0  = ss0 + 456 ;
   double*  fp1  = ss0 + 486 ;
   double*  dd0  = ss0 + 606 ;
   double*  dd1  = ss0 + 642 ;
   double*  pf1  = ss0 + 750 ;
   double*  pg1  = ss0 + 810 ;
   double*  fd0  = ss0 + 855 ;
   double*  fd1  = ss0 + 915 ;
   double*  df0  = ss0 + 1095 ;
   double*  df1  = ss0 + 1155 ;
   double*  gp0  = ss0 + 1275 ;
   double*  dg0  = ss0 + 1500 ;
   double*  dg1  = ss0 + 1590 ;
   double*  ff0  = ss0 + 1680 ;
   double*  ff1  = ss0 + 1780 ;
   double*  gd0  = ss0 + 1980 ;
   double*  dh0  = ss0 + 2340 ;
   double*  fg0  = ss0 + 2466 ;
   double*  fg1  = ss0 + 2616 ;
   double*  gf0  = ss0 + 2766 ;
   double*  fh0  = ss0 + 3216 ;
   double*  gg0  = ss0 + 3426 ;
   double*  gh0  = ss0 + 3876 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ddff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  sp2  = ss0 + 280 ;
   double*  sd2  = ss0 + 292 ;
   double*  pp1  = ss0 + 310 ;
   double*  dp0  = ss0 + 355 ;
   double*  dp1  = ss0 + 373 ;
   double*  sf2  = ss0 + 463 ;
   double*  pd1  = ss0 + 483 ;
   double*  sg2  = ss0 + 555 ;
   double*  pf1  = ss0 + 570 ;
   double*  fp0  = ss0 + 660 ;
   double*  fp1  = ss0 + 690 ;
   double*  dd0  = ss0 + 840 ;
   double*  dd1  = ss0 + 876 ;
   double*  df0  = ss0 + 1020 ;
   double*  df1  = ss0 + 1080 ;
   double*  fd0  = ss0 + 1260 ;
   double*  fd1  = ss0 + 1320 ;
   double*  pg1  = ss0 + 1560 ;
   double*  gp0  = ss0 + 1650 ;
   double*  dg0  = ss0 + 1920 ;
   double*  dg1  = ss0 + 2010 ;
   double*  ff0  = ss0 + 2190 ;
   double*  ff1  = ss0 + 2290 ;
   double*  gd0  = ss0 + 2590 ;
   double*  ph1  = ss0 + 3040 ;
   double*  gf0  = ss0 + 3103 ;
   double*  dh0  = ss0 + 3703 ;
   double*  dh1  = ss0 + 3829 ;
   double*  fg0  = ss0 + 3955 ;
   double*  fg1  = ss0 + 4105 ;
   double*  gg0  = ss0 + 4405 ;
   double*  fh0  = ss0 + 5080 ;
   double*  fh1  = ss0 + 5290 ;
   double*  di0  = ss0 + 5500 ;
   double*  gh0  = ss0 + 5668 ;
   double*  fi0  = ss0 + 6298 ;
   double*  gi0  = ss0 + 6578 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  fs0  = ss0 + 45 ;
   double*  gs0  = ss0 + 75 ;
   double*  hs0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( hs0 , gs0 , fs0 , PA_WP, zab );
}

template< int VTS >
__device__ void dfsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  dp0  = ss0 + 182 ;
   double*  fp0  = ss0 + 200 ;
   double*  gp0  = ss0 + 230 ;
   double*  hp0  = ss0 + 275 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void dfsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  pp1  = ss0 + 238 ;
   double*  dp0  = ss0 + 247 ;
   double*  dp1  = ss0 + 265 ;
   double*  dd0  = ss0 + 283 ;
   double*  fp0  = ss0 + 319 ;
   double*  fp1  = ss0 + 349 ;
   double*  fd0  = ss0 + 379 ;
   double*  gp0  = ss0 + 439 ;
   double*  gp1  = ss0 + 484 ;
   double*  gd0  = ss0 + 529 ;
   double*  hp0  = ss0 + 619 ;
   double*  hd0  = ss0 + 745 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  pd1  = ss0 + 315 ;
   double*  dp0  = ss0 + 333 ;
   double*  dp1  = ss0 + 351 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  ff0  = ss0 + 864 ;
   double*  hp0  = ss0 + 964 ;
   double*  gd0  = ss0 + 1153 ;
   double*  gd1  = ss0 + 1243 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps1  = ss0 + 10 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  dp0  = ss0 + 182 ;
   double*  fp0  = ss0 + 200 ;
   double*  gp0  = ss0 + 230 ;
   double*  hp0  = ss0 + 275 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void dfpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  pp1  = ss0 + 238 ;
   double*  dp0  = ss0 + 247 ;
   double*  dp1  = ss0 + 265 ;
   double*  dd0  = ss0 + 283 ;
   double*  fp0  = ss0 + 319 ;
   double*  fp1  = ss0 + 349 ;
   double*  fd0  = ss0 + 379 ;
   double*  gp0  = ss0 + 439 ;
   double*  gp1  = ss0 + 484 ;
   double*  gd0  = ss0 + 529 ;
   double*  hp0  = ss0 + 619 ;
   double*  hd0  = ss0 + 745 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  pd1  = ss0 + 315 ;
   double*  dp0  = ss0 + 333 ;
   double*  dp1  = ss0 + 351 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  hp0  = ss0 + 864 ;
   double*  ff0  = ss0 + 1053 ;
   double*  gd0  = ss0 + 1153 ;
   double*  gd1  = ss0 + 1243 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp2  = ss0 + 350 ;
   double*  sd2  = ss0 + 356 ;
   double*  pp1  = ss0 + 362 ;
   double*  pd1  = ss0 + 389 ;
   double*  dp0  = ss0 + 425 ;
   double*  dp1  = ss0 + 443 ;
   double*  pf1  = ss0 + 497 ;
   double*  fp0  = ss0 + 527 ;
   double*  fp1  = ss0 + 557 ;
   double*  dd0  = ss0 + 647 ;
   double*  dd1  = ss0 + 683 ;
   double*  df0  = ss0 + 755 ;
   double*  df1  = ss0 + 815 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  gp0  = ss0 + 1055 ;
   double*  gp1  = ss0 + 1100 ;
   double*  dg0  = ss0 + 1235 ;
   double*  ff0  = ss0 + 1325 ;
   double*  ff1  = ss0 + 1425 ;
   double*  hp0  = ss0 + 1525 ;
   double*  gd0  = ss0 + 1777 ;
   double*  gd1  = ss0 + 1867 ;
   double*  fg0  = ss0 + 2047 ;
   double*  hd0  = ss0 + 2197 ;
   double*  gf0  = ss0 + 2575 ;
   double*  gf1  = ss0 + 2725 ;
   double*  gg0  = ss0 + 2875 ;
   double*  hf0  = ss0 + 3100 ;
   double*  hg0  = ss0 + 3520 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps1  = ss0 + 11 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  pp1  = ss0 + 238 ;
   double*  dp0  = ss0 + 247 ;
   double*  dp1  = ss0 + 265 ;
   double*  fp0  = ss0 + 283 ;
   double*  fp1  = ss0 + 313 ;
   double*  dd0  = ss0 + 343 ;
   double*  fd0  = ss0 + 379 ;
   double*  gp0  = ss0 + 439 ;
   double*  gp1  = ss0 + 484 ;
   double*  gd0  = ss0 + 529 ;
   double*  hp0  = ss0 + 619 ;
   double*  hd0  = ss0 + 745 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  dp0  = ss0 + 315 ;
   double*  dp1  = ss0 + 333 ;
   double*  pd1  = ss0 + 369 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  hp0  = ss0 + 864 ;
   double*  gd0  = ss0 + 1053 ;
   double*  gd1  = ss0 + 1143 ;
   double*  ff0  = ss0 + 1233 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp2  = ss0 + 350 ;
   double*  sd2  = ss0 + 356 ;
   double*  pp1  = ss0 + 362 ;
   double*  pd1  = ss0 + 389 ;
   double*  dp0  = ss0 + 425 ;
   double*  dp1  = ss0 + 443 ;
   double*  pf1  = ss0 + 497 ;
   double*  dd0  = ss0 + 527 ;
   double*  dd1  = ss0 + 563 ;
   double*  fp0  = ss0 + 635 ;
   double*  fp1  = ss0 + 665 ;
   double*  df0  = ss0 + 755 ;
   double*  df1  = ss0 + 815 ;
   double*  gp0  = ss0 + 875 ;
   double*  gp1  = ss0 + 920 ;
   double*  fd0  = ss0 + 1055 ;
   double*  fd1  = ss0 + 1115 ;
   double*  dg0  = ss0 + 1235 ;
   double*  ff0  = ss0 + 1325 ;
   double*  ff1  = ss0 + 1425 ;
   double*  hp0  = ss0 + 1525 ;
   double*  gd0  = ss0 + 1777 ;
   double*  gd1  = ss0 + 1867 ;
   double*  fg0  = ss0 + 2047 ;
   double*  hd0  = ss0 + 2197 ;
   double*  gf0  = ss0 + 2575 ;
   double*  gf1  = ss0 + 2725 ;
   double*  gg0  = ss0 + 2875 ;
   double*  hf0  = ss0 + 3100 ;
   double*  hg0  = ss0 + 3520 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp2  = ss0 + 406 ;
   double*  pp1  = ss0 + 415 ;
   double*  sd2  = ss0 + 451 ;
   double*  sf2  = ss0 + 463 ;
   double*  pd1  = ss0 + 473 ;
   double*  dp0  = ss0 + 527 ;
   double*  dp1  = ss0 + 545 ;
   double*  pf1  = ss0 + 617 ;
   double*  fp0  = ss0 + 677 ;
   double*  fp1  = ss0 + 707 ;
   double*  dd0  = ss0 + 827 ;
   double*  dd1  = ss0 + 863 ;
   double*  pg1  = ss0 + 971 ;
   double*  df0  = ss0 + 1016 ;
   double*  df1  = ss0 + 1076 ;
   double*  fd0  = ss0 + 1196 ;
   double*  fd1  = ss0 + 1256 ;
   double*  gp0  = ss0 + 1436 ;
   double*  gp1  = ss0 + 1481 ;
   double*  gd0  = ss0 + 1661 ;
   double*  gd1  = ss0 + 1751 ;
   double*  ff0  = ss0 + 2021 ;
   double*  ff1  = ss0 + 2121 ;
   double*  hp0  = ss0 + 2321 ;
   double*  dg0  = ss0 + 2636 ;
   double*  dg1  = ss0 + 2726 ;
   double*  hd0  = ss0 + 2816 ;
   double*  gf0  = ss0 + 3320 ;
   double*  gf1  = ss0 + 3470 ;
   double*  dh0  = ss0 + 3770 ;
   double*  fg0  = ss0 + 3896 ;
   double*  fg1  = ss0 + 4046 ;
   double*  hf0  = ss0 + 4196 ;
   double*  gg0  = ss0 + 4826 ;
   double*  gg1  = ss0 + 5051 ;
   double*  fh0  = ss0 + 5276 ;
   double*  hg0  = ss0 + 5486 ;
   double*  gh0  = ss0 + 6116 ;
   double*  hh0  = ss0 + 6431 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps1  = ss0 + 12 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  sp2  = ss0 + 294 ;
   double*  pp1  = ss0 + 297 ;
   double*  dp0  = ss0 + 315 ;
   double*  dp1  = ss0 + 333 ;
   double*  pd1  = ss0 + 369 ;
   double*  fp0  = ss0 + 387 ;
   double*  fp1  = ss0 + 417 ;
   double*  dd0  = ss0 + 477 ;
   double*  dd1  = ss0 + 513 ;
   double*  df0  = ss0 + 549 ;
   double*  gp0  = ss0 + 609 ;
   double*  gp1  = ss0 + 654 ;
   double*  fd0  = ss0 + 744 ;
   double*  fd1  = ss0 + 804 ;
   double*  hp0  = ss0 + 864 ;
   double*  gd0  = ss0 + 1053 ;
   double*  gd1  = ss0 + 1143 ;
   double*  ff0  = ss0 + 1233 ;
   double*  hd0  = ss0 + 1333 ;
   double*  gf0  = ss0 + 1585 ;
   double*  hf0  = ss0 + 1735 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps1  = ss0 + 13 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp2  = ss0 + 350 ;
   double*  sd2  = ss0 + 356 ;
   double*  pp1  = ss0 + 362 ;
   double*  dp0  = ss0 + 389 ;
   double*  dp1  = ss0 + 407 ;
   double*  pd1  = ss0 + 461 ;
   double*  pf1  = ss0 + 497 ;
   double*  dd0  = ss0 + 527 ;
   double*  dd1  = ss0 + 563 ;
   double*  fp0  = ss0 + 635 ;
   double*  fp1  = ss0 + 665 ;
   double*  df0  = ss0 + 755 ;
   double*  df1  = ss0 + 815 ;
   double*  gp0  = ss0 + 875 ;
   double*  gp1  = ss0 + 920 ;
   double*  fd0  = ss0 + 1055 ;
   double*  fd1  = ss0 + 1115 ;
   double*  dg0  = ss0 + 1235 ;
   double*  ff0  = ss0 + 1325 ;
   double*  ff1  = ss0 + 1425 ;
   double*  gd0  = ss0 + 1525 ;
   double*  gd1  = ss0 + 1615 ;
   double*  hp0  = ss0 + 1795 ;
   double*  fg0  = ss0 + 2047 ;
   double*  hd0  = ss0 + 2197 ;
   double*  gf0  = ss0 + 2575 ;
   double*  gf1  = ss0 + 2725 ;
   double*  gg0  = ss0 + 2875 ;
   double*  hf0  = ss0 + 3100 ;
   double*  hg0  = ss0 + 3520 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps1  = ss0 + 14 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp2  = ss0 + 406 ;
   double*  pp1  = ss0 + 415 ;
   double*  sd2  = ss0 + 451 ;
   double*  sf2  = ss0 + 463 ;
   double*  dp0  = ss0 + 473 ;
   double*  dp1  = ss0 + 491 ;
   double*  pd1  = ss0 + 563 ;
   double*  pf1  = ss0 + 617 ;
   double*  fp0  = ss0 + 677 ;
   double*  fp1  = ss0 + 707 ;
   double*  dd0  = ss0 + 827 ;
   double*  dd1  = ss0 + 863 ;
   double*  pg1  = ss0 + 971 ;
   double*  df0  = ss0 + 1016 ;
   double*  df1  = ss0 + 1076 ;
   double*  fd0  = ss0 + 1196 ;
   double*  fd1  = ss0 + 1256 ;
   double*  gp0  = ss0 + 1436 ;
   double*  gp1  = ss0 + 1481 ;
   double*  gd0  = ss0 + 1661 ;
   double*  gd1  = ss0 + 1751 ;
   double*  ff0  = ss0 + 2021 ;
   double*  ff1  = ss0 + 2121 ;
   double*  dg0  = ss0 + 2321 ;
   double*  dg1  = ss0 + 2411 ;
   double*  hp0  = ss0 + 2501 ;
   double*  hd0  = ss0 + 2816 ;
   double*  gf0  = ss0 + 3320 ;
   double*  gf1  = ss0 + 3470 ;
   double*  fg0  = ss0 + 3770 ;
   double*  fg1  = ss0 + 3920 ;
   double*  dh0  = ss0 + 4070 ;
   double*  hf0  = ss0 + 4196 ;
   double*  gg0  = ss0 + 4826 ;
   double*  gg1  = ss0 + 5051 ;
   double*  fh0  = ss0 + 5276 ;
   double*  hg0  = ss0 + 5486 ;
   double*  gh0  = ss0 + 6116 ;
   double*  hh0  = ss0 + 6431 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp1 , ps1 , ss2 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp0 , ds0 , ps1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void dfff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss2  = ss0 + 2 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps1  = ss0 + 15 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  sp2  = ss0 + 462 ;
   double*  sd2  = ss0 + 474 ;
   double*  pp1  = ss0 + 492 ;
   double*  sf2  = ss0 + 537 ;
   double*  pd1  = ss0 + 557 ;
   double*  dp0  = ss0 + 629 ;
   double*  dp1  = ss0 + 647 ;
   double*  sg2  = ss0 + 737 ;
   double*  pf1  = ss0 + 752 ;
   double*  dd0  = ss0 + 842 ;
   double*  dd1  = ss0 + 878 ;
   double*  fp0  = ss0 + 1022 ;
   double*  fp1  = ss0 + 1052 ;
   double*  pg1  = ss0 + 1202 ;
   double*  df0  = ss0 + 1292 ;
   double*  df1  = ss0 + 1352 ;
   double*  gp0  = ss0 + 1532 ;
   double*  gp1  = ss0 + 1577 ;
   double*  fd0  = ss0 + 1802 ;
   double*  fd1  = ss0 + 1862 ;
   double*  ph1  = ss0 + 2102 ;
   double*  dg0  = ss0 + 2165 ;
   double*  dg1  = ss0 + 2255 ;
   double*  gd0  = ss0 + 2435 ;
   double*  gd1  = ss0 + 2525 ;
   double*  ff0  = ss0 + 2885 ;
   double*  ff1  = ss0 + 2985 ;
   double*  hp0  = ss0 + 3285 ;
   double*  dh0  = ss0 + 3663 ;
   double*  dh1  = ss0 + 3789 ;
   double*  gf0  = ss0 + 3915 ;
   double*  gf1  = ss0 + 4065 ;
   double*  hd0  = ss0 + 4515 ;
   double*  fg0  = ss0 + 5145 ;
   double*  fg1  = ss0 + 5295 ;
   double*  di0  = ss0 + 5595 ;
   double*  fh0  = ss0 + 5763 ;
   double*  fh1  = ss0 + 5973 ;
   double*  hf0  = ss0 + 6183 ;
   double*  gg0  = ss0 + 7023 ;
   double*  gg1  = ss0 + 7248 ;
   double*  fi0  = ss0 + 7698 ;
   double*  hg0  = ss0 + 7978 ;
   double*  gh0  = ss0 + 8923 ;
   double*  gh1  = ss0 + 9238 ;
   double*  hh0  = ss0 + 9553 ;
   double*  gi0  = ss0 + 10435 ;
   double*  hi0  = ss0 + 10855 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 4 >( sp2 , ss2 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( sd2 , sp2 , ss2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 5 >( pp1 , ps1 , ss2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( sf2 , sd2 , sp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 4 >( pd1 , pp1 , ps1 , sp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 6 >( dp0 , ds0 , ps1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( sg2 , sf2 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 3 >( pf1 , pd1 , pp1 , sd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 5 >( dd0 , dp0 , ds0 , pp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 2 >( pg1 , pf1 , pd1 , sf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 4 >( df0 , dd0 , dp0 , pd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 5, 1 >( ph1 , pg1 , pf1 , sg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 3 >( dg0 , df0 , dd0 , pf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 6 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 2 >( dh0 , dg0 , df0 , pg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 5 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 6, 1 >( di0 , dh0 , dg0 , ph1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 4 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 3 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 2 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 6, 1 >( hi0 , hh0 , hg0 , gh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 4 ;
   double*  ds0  = ss0 + 13 ;
   double*  fs0  = ss0 + 25 ;
   execute_VRR1_gpu< VTS, 1, 0, 3 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( fs0 , ds0 , ps0 , PA_WP, zab );
}

template< int VTS >
__device__ void fssp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  fp0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void fssd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps2  = ss0 + 12 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  dp1  = ss0 + 75 ;
   double*  fp0  = ss0 + 93 ;
   double*  fd0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fssf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  dd1  = ss0 + 140 ;
   double*  fp0  = ss0 + 176 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  ds1  = ss0 + 23 ;
   double*  fs0  = ss0 + 35 ;
   double*  fp0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
}

template< int VTS >
__device__ void fspp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps2  = ss0 + 12 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  dp1  = ss0 + 75 ;
   double*  fp0  = ss0 + 93 ;
   double*  fd0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fspd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  fp0  = ss0 + 140 ;
   double*  dd1  = ss0 + 230 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fspf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp3  = ss0 + 115 ;
   double*  pp2  = ss0 + 118 ;
   double*  pd2  = ss0 + 136 ;
   double*  dp1  = ss0 + 154 ;
   double*  dd1  = ss0 + 208 ;
   double*  fp0  = ss0 + 280 ;
   double*  df1  = ss0 + 400 ;
   double*  fd0  = ss0 + 460 ;
   double*  ff0  = ss0 + 640 ;
   double*  fg0  = ss0 + 840 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ps2  = ss0 + 12 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  dp1  = ss0 + 75 ;
   double*  fp0  = ss0 + 93 ;
   double*  fd0  = ss0 + 153 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  fp0  = ss0 + 140 ;
   double*  dd1  = ss0 + 230 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp3  = ss0 + 115 ;
   double*  pp2  = ss0 + 118 ;
   double*  pd2  = ss0 + 136 ;
   double*  dp1  = ss0 + 154 ;
   double*  dd1  = ss0 + 208 ;
   double*  fp0  = ss0 + 280 ;
   double*  df1  = ss0 + 400 ;
   double*  fd0  = ss0 + 460 ;
   double*  ff0  = ss0 + 640 ;
   double*  fg0  = ss0 + 840 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp3  = ss0 + 135 ;
   double*  pp2  = ss0 + 141 ;
   double*  sd3  = ss0 + 168 ;
   double*  dp1  = ss0 + 174 ;
   double*  pd2  = ss0 + 246 ;
   double*  fp0  = ss0 + 282 ;
   double*  dd1  = ss0 + 432 ;
   double*  pf2  = ss0 + 540 ;
   double*  df1  = ss0 + 570 ;
   double*  fd0  = ss0 + 690 ;
   double*  dg1  = ss0 + 930 ;
   double*  ff0  = ss0 + 1020 ;
   double*  fg0  = ss0 + 1320 ;
   double*  fh0  = ss0 + 1620 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  pp2  = ss0 + 95 ;
   double*  dp1  = ss0 + 104 ;
   double*  fp0  = ss0 + 140 ;
   double*  dd1  = ss0 + 230 ;
   double*  fd0  = ss0 + 266 ;
   double*  ff0  = ss0 + 386 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  sp3  = ss0 + 115 ;
   double*  pp2  = ss0 + 118 ;
   double*  pd2  = ss0 + 136 ;
   double*  dp1  = ss0 + 154 ;
   double*  dd1  = ss0 + 208 ;
   double*  fp0  = ss0 + 280 ;
   double*  df1  = ss0 + 400 ;
   double*  fd0  = ss0 + 460 ;
   double*  ff0  = ss0 + 640 ;
   double*  fg0  = ss0 + 840 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  sp3  = ss0 + 135 ;
   double*  sd3  = ss0 + 141 ;
   double*  pp2  = ss0 + 147 ;
   double*  pd2  = ss0 + 174 ;
   double*  dp1  = ss0 + 210 ;
   double*  pf2  = ss0 + 282 ;
   double*  dd1  = ss0 + 312 ;
   double*  fp0  = ss0 + 420 ;
   double*  fd0  = ss0 + 570 ;
   double*  df1  = ss0 + 810 ;
   double*  ff0  = ss0 + 930 ;
   double*  dg1  = ss0 + 1230 ;
   double*  fg0  = ss0 + 1320 ;
   double*  fh0  = ss0 + 1620 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fsff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  sp3  = ss0 + 155 ;
   double*  sd3  = ss0 + 164 ;
   double*  pp2  = ss0 + 176 ;
   double*  sf3  = ss0 + 212 ;
   double*  dp1  = ss0 + 222 ;
   double*  pd2  = ss0 + 312 ;
   double*  fp0  = ss0 + 366 ;
   double*  dd1  = ss0 + 546 ;
   double*  pf2  = ss0 + 690 ;
   double*  pg2  = ss0 + 750 ;
   double*  fd0  = ss0 + 795 ;
   double*  df1  = ss0 + 1095 ;
   double*  dg1  = ss0 + 1275 ;
   double*  ff0  = ss0 + 1455 ;
   double*  dh1  = ss0 + 1855 ;
   double*  fg0  = ss0 + 1981 ;
   double*  fh0  = ss0 + 2431 ;
   double*  fi0  = ss0 + 2851 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 5 ;
   double*  ds0  = ss0 + 17 ;
   double*  fs0  = ss0 + 35 ;
   double*  gs0  = ss0 + 55 ;
   execute_VRR1_gpu< VTS, 1, 0, 4 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 3 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 2 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 1 >( gs0 , fs0 , ds0 , PA_WP, zab );
}

template< int VTS >
__device__ void fpsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  fp0  = ss0 + 105 ;
   double*  gp0  = ss0 + 135 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fpsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  dp1  = ss0 + 140 ;
   double*  fp0  = ss0 + 158 ;
   double*  fp1  = ss0 + 188 ;
   double*  gp0  = ss0 + 218 ;
   double*  fd0  = ss0 + 308 ;
   double*  gd0  = ss0 + 368 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  ds1  = ss0 + 27 ;
   double*  fs0  = ss0 + 45 ;
   double*  fs1  = ss0 + 55 ;
   double*  gs0  = ss0 + 75 ;
   double*  fp0  = ss0 + 105 ;
   double*  gp0  = ss0 + 135 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fppp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  dp1  = ss0 + 140 ;
   double*  fp0  = ss0 + 158 ;
   double*  fp1  = ss0 + 188 ;
   double*  fd0  = ss0 + 218 ;
   double*  gp0  = ss0 + 278 ;
   double*  gd0  = ss0 + 368 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fppd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fppf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp3  = ss0 + 210 ;
   double*  pp2  = ss0 + 213 ;
   double*  dp1  = ss0 + 231 ;
   double*  pd2  = ss0 + 285 ;
   double*  fp0  = ss0 + 303 ;
   double*  fp1  = ss0 + 333 ;
   double*  dd1  = ss0 + 423 ;
   double*  df1  = ss0 + 495 ;
   double*  gp0  = ss0 + 555 ;
   double*  fd0  = ss0 + 735 ;
   double*  fd1  = ss0 + 795 ;
   double*  gd0  = ss0 + 915 ;
   double*  ff0  = ss0 + 1185 ;
   double*  ff1  = ss0 + 1285 ;
   double*  gf0  = ss0 + 1385 ;
   double*  fg0  = ss0 + 1685 ;
   double*  gg0  = ss0 + 1835 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ps2  = ss0 + 13 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  dp1  = ss0 + 140 ;
   double*  fp0  = ss0 + 158 ;
   double*  fp1  = ss0 + 188 ;
   double*  fd0  = ss0 + 218 ;
   double*  gp0  = ss0 + 278 ;
   double*  gd0  = ss0 + 368 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp3  = ss0 + 210 ;
   double*  pp2  = ss0 + 213 ;
   double*  pd2  = ss0 + 231 ;
   double*  dp1  = ss0 + 249 ;
   double*  fp0  = ss0 + 303 ;
   double*  fp1  = ss0 + 333 ;
   double*  dd1  = ss0 + 423 ;
   double*  gp0  = ss0 + 495 ;
   double*  fd0  = ss0 + 675 ;
   double*  fd1  = ss0 + 735 ;
   double*  df1  = ss0 + 855 ;
   double*  gd0  = ss0 + 915 ;
   double*  ff0  = ss0 + 1185 ;
   double*  ff1  = ss0 + 1285 ;
   double*  gf0  = ss0 + 1385 ;
   double*  fg0  = ss0 + 1685 ;
   double*  gg0  = ss0 + 1835 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp3  = ss0 + 245 ;
   double*  sd3  = ss0 + 251 ;
   double*  pp2  = ss0 + 257 ;
   double*  dp1  = ss0 + 284 ;
   double*  pd2  = ss0 + 356 ;
   double*  fp0  = ss0 + 392 ;
   double*  fp1  = ss0 + 422 ;
   double*  dd1  = ss0 + 542 ;
   double*  pf2  = ss0 + 650 ;
   double*  fd0  = ss0 + 680 ;
   double*  fd1  = ss0 + 740 ;
   double*  df1  = ss0 + 920 ;
   double*  gp0  = ss0 + 1040 ;
   double*  dg1  = ss0 + 1265 ;
   double*  ff0  = ss0 + 1355 ;
   double*  ff1  = ss0 + 1455 ;
   double*  gd0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 2015 ;
   double*  fg1  = ss0 + 2165 ;
   double*  gf0  = ss0 + 2315 ;
   double*  fh0  = ss0 + 2765 ;
   double*  gg0  = ss0 + 2975 ;
   double*  gh0  = ss0 + 3425 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  pp2  = ss0 + 175 ;
   double*  dp1  = ss0 + 184 ;
   double*  dd1  = ss0 + 220 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 346 ;
   double*  fd1  = ss0 + 406 ;
   double*  gp0  = ss0 + 466 ;
   double*  ff0  = ss0 + 601 ;
   double*  gd0  = ss0 + 701 ;
   double*  gf0  = ss0 + 881 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  sp3  = ss0 + 210 ;
   double*  pp2  = ss0 + 213 ;
   double*  pd2  = ss0 + 231 ;
   double*  dp1  = ss0 + 249 ;
   double*  dd1  = ss0 + 303 ;
   double*  fp0  = ss0 + 375 ;
   double*  fp1  = ss0 + 405 ;
   double*  df1  = ss0 + 495 ;
   double*  gp0  = ss0 + 555 ;
   double*  fd0  = ss0 + 735 ;
   double*  fd1  = ss0 + 795 ;
   double*  ff0  = ss0 + 915 ;
   double*  ff1  = ss0 + 1015 ;
   double*  gd0  = ss0 + 1115 ;
   double*  fg0  = ss0 + 1385 ;
   double*  gf0  = ss0 + 1535 ;
   double*  gg0  = ss0 + 1835 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  sp3  = ss0 + 245 ;
   double*  sd3  = ss0 + 251 ;
   double*  pp2  = ss0 + 257 ;
   double*  dp1  = ss0 + 284 ;
   double*  pd2  = ss0 + 356 ;
   double*  dd1  = ss0 + 392 ;
   double*  pf2  = ss0 + 500 ;
   double*  fp0  = ss0 + 530 ;
   double*  fp1  = ss0 + 560 ;
   double*  fd0  = ss0 + 680 ;
   double*  fd1  = ss0 + 740 ;
   double*  df1  = ss0 + 920 ;
   double*  gp0  = ss0 + 1040 ;
   double*  dg1  = ss0 + 1265 ;
   double*  ff0  = ss0 + 1355 ;
   double*  ff1  = ss0 + 1455 ;
   double*  gd0  = ss0 + 1655 ;
   double*  fg0  = ss0 + 2015 ;
   double*  fg1  = ss0 + 2165 ;
   double*  gf0  = ss0 + 2315 ;
   double*  fh0  = ss0 + 2765 ;
   double*  gg0  = ss0 + 2975 ;
   double*  gh0  = ss0 + 3425 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fpff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  sp3  = ss0 + 280 ;
   double*  pp2  = ss0 + 289 ;
   double*  sd3  = ss0 + 325 ;
   double*  sf3  = ss0 + 337 ;
   double*  pd2  = ss0 + 347 ;
   double*  dp1  = ss0 + 401 ;
   double*  fp0  = ss0 + 491 ;
   double*  fp1  = ss0 + 521 ;
   double*  pf2  = ss0 + 671 ;
   double*  dd1  = ss0 + 731 ;
   double*  gp0  = ss0 + 875 ;
   double*  pg2  = ss0 + 1145 ;
   double*  fd0  = ss0 + 1190 ;
   double*  fd1  = ss0 + 1250 ;
   double*  df1  = ss0 + 1490 ;
   double*  dg1  = ss0 + 1670 ;
   double*  gd0  = ss0 + 1850 ;
   double*  ff0  = ss0 + 2300 ;
   double*  ff1  = ss0 + 2400 ;
   double*  fg0  = ss0 + 2700 ;
   double*  fg1  = ss0 + 2850 ;
   double*  dh1  = ss0 + 3150 ;
   double*  gf0  = ss0 + 3276 ;
   double*  fh0  = ss0 + 3876 ;
   double*  fh1  = ss0 + 4086 ;
   double*  gg0  = ss0 + 4296 ;
   double*  fi0  = ss0 + 4971 ;
   double*  gh0  = ss0 + 5251 ;
   double*  gi0  = ss0 + 5881 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 6 ;
   double*  ds0  = ss0 + 21 ;
   double*  fs0  = ss0 + 45 ;
   double*  gs0  = ss0 + 75 ;
   double*  hs0  = ss0 + 105 ;
   execute_VRR1_gpu< VTS, 1, 0, 5 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 4 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 3 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 2 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 1 >( hs0 , gs0 , fs0 , PA_WP, zab );
}

template< int VTS >
__device__ void fdsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  fp0  = ss0 + 182 ;
   double*  gp0  = ss0 + 212 ;
   double*  hp0  = ss0 + 257 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fdsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  dp1  = ss0 + 238 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 316 ;
   double*  gp0  = ss0 + 376 ;
   double*  gp1  = ss0 + 421 ;
   double*  gd0  = ss0 + 466 ;
   double*  hp0  = ss0 + 556 ;
   double*  hd0  = ss0 + 682 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  fp0  = ss0 + 339 ;
   double*  fp1  = ss0 + 369 ;
   double*  dd1  = ss0 + 429 ;
   double*  gp0  = ss0 + 465 ;
   double*  gp1  = ss0 + 510 ;
   double*  fd0  = ss0 + 600 ;
   double*  fd1  = ss0 + 660 ;
   double*  ff0  = ss0 + 720 ;
   double*  hp0  = ss0 + 820 ;
   double*  gd0  = ss0 + 1009 ;
   double*  gd1  = ss0 + 1099 ;
   double*  hd0  = ss0 + 1189 ;
   double*  gf0  = ss0 + 1441 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 7 ;
   double*  ds0  = ss0 + 25 ;
   double*  ds1  = ss0 + 31 ;
   double*  fs0  = ss0 + 55 ;
   double*  fs1  = ss0 + 65 ;
   double*  gs0  = ss0 + 95 ;
   double*  gs1  = ss0 + 110 ;
   double*  hs0  = ss0 + 140 ;
   double*  fp0  = ss0 + 182 ;
   double*  gp0  = ss0 + 212 ;
   double*  hp0  = ss0 + 257 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
}

template< int VTS >
__device__ void fdpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  dp1  = ss0 + 238 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 316 ;
   double*  gp0  = ss0 + 376 ;
   double*  gp1  = ss0 + 421 ;
   double*  gd0  = ss0 + 466 ;
   double*  hp0  = ss0 + 556 ;
   double*  hd0  = ss0 + 682 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  fp0  = ss0 + 339 ;
   double*  fp1  = ss0 + 369 ;
   double*  dd1  = ss0 + 429 ;
   double*  gp0  = ss0 + 465 ;
   double*  gp1  = ss0 + 510 ;
   double*  fd0  = ss0 + 600 ;
   double*  fd1  = ss0 + 660 ;
   double*  hp0  = ss0 + 720 ;
   double*  ff0  = ss0 + 909 ;
   double*  gd0  = ss0 + 1009 ;
   double*  gd1  = ss0 + 1099 ;
   double*  hd0  = ss0 + 1189 ;
   double*  gf0  = ss0 + 1441 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp3  = ss0 + 350 ;
   double*  pp2  = ss0 + 353 ;
   double*  pd2  = ss0 + 371 ;
   double*  dp1  = ss0 + 389 ;
   double*  fp0  = ss0 + 443 ;
   double*  fp1  = ss0 + 473 ;
   double*  dd1  = ss0 + 563 ;
   double*  df1  = ss0 + 635 ;
   double*  gp0  = ss0 + 695 ;
   double*  gp1  = ss0 + 740 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  ff0  = ss0 + 1055 ;
   double*  ff1  = ss0 + 1155 ;
   double*  hp0  = ss0 + 1255 ;
   double*  gd0  = ss0 + 1507 ;
   double*  gd1  = ss0 + 1597 ;
   double*  fg0  = ss0 + 1777 ;
   double*  hd0  = ss0 + 1927 ;
   double*  gf0  = ss0 + 2305 ;
   double*  gf1  = ss0 + 2455 ;
   double*  gg0  = ss0 + 2605 ;
   double*  hf0  = ss0 + 2830 ;
   double*  hg0  = ss0 + 3250 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ps2  = ss0 + 14 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  dp1  = ss0 + 238 ;
   double*  fp0  = ss0 + 256 ;
   double*  fp1  = ss0 + 286 ;
   double*  fd0  = ss0 + 316 ;
   double*  gp0  = ss0 + 376 ;
   double*  gp1  = ss0 + 421 ;
   double*  gd0  = ss0 + 466 ;
   double*  hp0  = ss0 + 556 ;
   double*  hd0  = ss0 + 682 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fddp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  dd1  = ss0 + 339 ;
   double*  fp0  = ss0 + 375 ;
   double*  fp1  = ss0 + 405 ;
   double*  fd0  = ss0 + 465 ;
   double*  fd1  = ss0 + 525 ;
   double*  gp0  = ss0 + 585 ;
   double*  gp1  = ss0 + 630 ;
   double*  gd0  = ss0 + 720 ;
   double*  gd1  = ss0 + 810 ;
   double*  ff0  = ss0 + 900 ;
   double*  hp0  = ss0 + 1000 ;
   double*  gf0  = ss0 + 1189 ;
   double*  hd0  = ss0 + 1339 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fddd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp3  = ss0 + 350 ;
   double*  pp2  = ss0 + 353 ;
   double*  pd2  = ss0 + 371 ;
   double*  dp1  = ss0 + 389 ;
   double*  dd1  = ss0 + 443 ;
   double*  fp0  = ss0 + 515 ;
   double*  fp1  = ss0 + 545 ;
   double*  df1  = ss0 + 635 ;
   double*  gp0  = ss0 + 695 ;
   double*  gp1  = ss0 + 740 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  ff0  = ss0 + 1055 ;
   double*  ff1  = ss0 + 1155 ;
   double*  hp0  = ss0 + 1255 ;
   double*  gd0  = ss0 + 1507 ;
   double*  gd1  = ss0 + 1597 ;
   double*  fg0  = ss0 + 1777 ;
   double*  hd0  = ss0 + 1927 ;
   double*  gf0  = ss0 + 2305 ;
   double*  gf1  = ss0 + 2455 ;
   double*  gg0  = ss0 + 2605 ;
   double*  hf0  = ss0 + 2830 ;
   double*  hg0  = ss0 + 3250 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fddf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp3  = ss0 + 406 ;
   double*  sd3  = ss0 + 412 ;
   double*  pp2  = ss0 + 418 ;
   double*  pd2  = ss0 + 445 ;
   double*  dp1  = ss0 + 481 ;
   double*  pf2  = ss0 + 553 ;
   double*  fp0  = ss0 + 583 ;
   double*  fp1  = ss0 + 613 ;
   double*  dd1  = ss0 + 733 ;
   double*  gp0  = ss0 + 841 ;
   double*  gp1  = ss0 + 886 ;
   double*  fd0  = ss0 + 1066 ;
   double*  fd1  = ss0 + 1126 ;
   double*  df1  = ss0 + 1306 ;
   double*  dg1  = ss0 + 1426 ;
   double*  hp0  = ss0 + 1516 ;
   double*  gd0  = ss0 + 1831 ;
   double*  gd1  = ss0 + 1921 ;
   double*  ff0  = ss0 + 2191 ;
   double*  ff1  = ss0 + 2291 ;
   double*  fg0  = ss0 + 2491 ;
   double*  fg1  = ss0 + 2641 ;
   double*  hd0  = ss0 + 2791 ;
   double*  gf0  = ss0 + 3295 ;
   double*  gf1  = ss0 + 3445 ;
   double*  fh0  = ss0 + 3745 ;
   double*  gg0  = ss0 + 3955 ;
   double*  gg1  = ss0 + 4180 ;
   double*  hf0  = ss0 + 4405 ;
   double*  gh0  = ss0 + 5035 ;
   double*  hg0  = ss0 + 5350 ;
   double*  hh0  = ss0 + 5980 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdfs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  pp2  = ss0 + 294 ;
   double*  dp1  = ss0 + 303 ;
   double*  dd1  = ss0 + 339 ;
   double*  fp0  = ss0 + 375 ;
   double*  fp1  = ss0 + 405 ;
   double*  fd0  = ss0 + 465 ;
   double*  fd1  = ss0 + 525 ;
   double*  gp0  = ss0 + 585 ;
   double*  gp1  = ss0 + 630 ;
   double*  gd0  = ss0 + 720 ;
   double*  gd1  = ss0 + 810 ;
   double*  ff0  = ss0 + 900 ;
   double*  hp0  = ss0 + 1000 ;
   double*  gf0  = ss0 + 1189 ;
   double*  hd0  = ss0 + 1339 ;
   double*  hf0  = ss0 + 1591 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdfp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  sp3  = ss0 + 350 ;
   double*  pp2  = ss0 + 353 ;
   double*  pd2  = ss0 + 371 ;
   double*  dp1  = ss0 + 389 ;
   double*  dd1  = ss0 + 443 ;
   double*  fp0  = ss0 + 515 ;
   double*  fp1  = ss0 + 545 ;
   double*  df1  = ss0 + 635 ;
   double*  gp0  = ss0 + 695 ;
   double*  gp1  = ss0 + 740 ;
   double*  fd0  = ss0 + 875 ;
   double*  fd1  = ss0 + 935 ;
   double*  ff0  = ss0 + 1055 ;
   double*  ff1  = ss0 + 1155 ;
   double*  gd0  = ss0 + 1255 ;
   double*  gd1  = ss0 + 1345 ;
   double*  hp0  = ss0 + 1525 ;
   double*  fg0  = ss0 + 1777 ;
   double*  hd0  = ss0 + 1927 ;
   double*  gf0  = ss0 + 2305 ;
   double*  gf1  = ss0 + 2455 ;
   double*  gg0  = ss0 + 2605 ;
   double*  hf0  = ss0 + 2830 ;
   double*  hg0  = ss0 + 3250 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdfd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  sp3  = ss0 + 406 ;
   double*  sd3  = ss0 + 412 ;
   double*  pp2  = ss0 + 418 ;
   double*  dp1  = ss0 + 445 ;
   double*  pd2  = ss0 + 517 ;
   double*  pf2  = ss0 + 553 ;
   double*  fp0  = ss0 + 583 ;
   double*  fp1  = ss0 + 613 ;
   double*  dd1  = ss0 + 733 ;
   double*  gp0  = ss0 + 841 ;
   double*  gp1  = ss0 + 886 ;
   double*  fd0  = ss0 + 1066 ;
   double*  fd1  = ss0 + 1126 ;
   double*  df1  = ss0 + 1306 ;
   double*  dg1  = ss0 + 1426 ;
   double*  hp0  = ss0 + 1516 ;
   double*  gd0  = ss0 + 1831 ;
   double*  gd1  = ss0 + 1921 ;
   double*  ff0  = ss0 + 2191 ;
   double*  ff1  = ss0 + 2291 ;
   double*  fg0  = ss0 + 2491 ;
   double*  fg1  = ss0 + 2641 ;
   double*  hd0  = ss0 + 2791 ;
   double*  gf0  = ss0 + 3295 ;
   double*  gf1  = ss0 + 3445 ;
   double*  fh0  = ss0 + 3745 ;
   double*  gg0  = ss0 + 3955 ;
   double*  gg1  = ss0 + 4180 ;
   double*  hf0  = ss0 + 4405 ;
   double*  gh0  = ss0 + 5035 ;
   double*  hg0  = ss0 + 5350 ;
   double*  hh0  = ss0 + 5980 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fdff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps2  = ss0 + 18 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  sp3  = ss0 + 462 ;
   double*  sd3  = ss0 + 471 ;
   double*  pp2  = ss0 + 483 ;
   double*  sf3  = ss0 + 519 ;
   double*  pd2  = ss0 + 529 ;
   double*  dp1  = ss0 + 583 ;
   double*  pf2  = ss0 + 673 ;
   double*  dd1  = ss0 + 733 ;
   double*  fp0  = ss0 + 877 ;
   double*  fp1  = ss0 + 907 ;
   double*  pg2  = ss0 + 1057 ;
   double*  df1  = ss0 + 1102 ;
   double*  gp0  = ss0 + 1282 ;
   double*  gp1  = ss0 + 1327 ;
   double*  fd0  = ss0 + 1552 ;
   double*  fd1  = ss0 + 1612 ;
   double*  dg1  = ss0 + 1852 ;
   double*  ff0  = ss0 + 2032 ;
   double*  ff1  = ss0 + 2132 ;
   double*  gd0  = ss0 + 2432 ;
   double*  gd1  = ss0 + 2522 ;
   double*  hp0  = ss0 + 2882 ;
   double*  dh1  = ss0 + 3260 ;
   double*  gf0  = ss0 + 3386 ;
   double*  gf1  = ss0 + 3536 ;
   double*  hd0  = ss0 + 3986 ;
   double*  fg0  = ss0 + 4616 ;
   double*  fg1  = ss0 + 4766 ;
   double*  fh0  = ss0 + 5066 ;
   double*  fh1  = ss0 + 5276 ;
   double*  hf0  = ss0 + 5486 ;
   double*  gg0  = ss0 + 6326 ;
   double*  gg1  = ss0 + 6551 ;
   double*  fi0  = ss0 + 7001 ;
   double*  hg0  = ss0 + 7281 ;
   double*  gh0  = ss0 + 8226 ;
   double*  gh1  = ss0 + 8541 ;
   double*  hh0  = ss0 + 8856 ;
   double*  gi0  = ss0 + 9738 ;
   double*  hi0  = ss0 + 10158 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 6 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 5 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 4 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 3 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 5, 2 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 6, 1 >( hi0 , hh0 , hg0 , gh1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffss(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * zab   = para + 12;
   double*  ps0  = ss0 + 7 ;
   double*  ds0  = ss0 + 25 ;
   double*  fs0  = ss0 + 55 ;
   double*  gs0  = ss0 + 95 ;
   double*  hs0  = ss0 + 140 ;
   double*  is0  = ss0 + 182 ;
   execute_VRR1_gpu< VTS, 1, 0, 6 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 5 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 4 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 3 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 2 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 1 >( is0 , hs0 , gs0 , PA_WP, zab );
}

template< int VTS >
__device__ void ffsp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  hs1  = ss0 + 196 ;
   double*  is0  = ss0 + 238 ;
   double*  fp0  = ss0 + 294 ;
   double*  gp0  = ss0 + 324 ;
   double*  hp0  = ss0 + 369 ;
   double*  ip0  = ss0 + 432 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 2 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 6, 1, 1 >( ip0 , is0 , hs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ffsd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  hs1  = ss0 + 231 ;
   double*  is0  = ss0 + 294 ;
   double*  dp1  = ss0 + 378 ;
   double*  fp0  = ss0 + 396 ;
   double*  fp1  = ss0 + 426 ;
   double*  fd0  = ss0 + 456 ;
   double*  gp0  = ss0 + 516 ;
   double*  gp1  = ss0 + 561 ;
   double*  hp0  = ss0 + 606 ;
   double*  hp1  = ss0 + 669 ;
   double*  gd0  = ss0 + 732 ;
   double*  ip0  = ss0 + 822 ;
   double*  hd0  = ss0 + 990 ;
   double*  id0  = ss0 + 1116 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 3 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 6, 1, 2 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 1 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffsf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  fd0  = ss0 + 633 ;
   double*  fd1  = ss0 + 693 ;
   double*  gp0  = ss0 + 753 ;
   double*  gp1  = ss0 + 798 ;
   double*  ff0  = ss0 + 888 ;
   double*  gd0  = ss0 + 988 ;
   double*  gd1  = ss0 + 1078 ;
   double*  hp0  = ss0 + 1168 ;
   double*  hp1  = ss0 + 1231 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffps(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double   z     = para[16];
   double*  ps0  = ss0 + 8 ;
   double*  ds0  = ss0 + 29 ;
   double*  ds1  = ss0 + 35 ;
   double*  fs0  = ss0 + 65 ;
   double*  fs1  = ss0 + 75 ;
   double*  gs0  = ss0 + 115 ;
   double*  gs1  = ss0 + 130 ;
   double*  hs0  = ss0 + 175 ;
   double*  hs1  = ss0 + 196 ;
   double*  is0  = ss0 + 238 ;
   double*  fp0  = ss0 + 294 ;
   double*  gp0  = ss0 + 324 ;
   double*  hp0  = ss0 + 369 ;
   double*  ip0  = ss0 + 432 ;
   execute_VRR1_gpu< VTS, 1, 0, 7 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 6 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 5 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 4 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 3 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 2 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 1 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 1 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 5, 1, 1 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 6, 1, 1 >( ip0 , is0 , hs1 , QC_WQ, z );
}

template< int VTS >
__device__ void ffpp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  hs1  = ss0 + 231 ;
   double*  is0  = ss0 + 294 ;
   double*  dp1  = ss0 + 378 ;
   double*  fp0  = ss0 + 396 ;
   double*  fp1  = ss0 + 426 ;
   double*  fd0  = ss0 + 456 ;
   double*  gp0  = ss0 + 516 ;
   double*  gp1  = ss0 + 561 ;
   double*  gd0  = ss0 + 606 ;
   double*  hp0  = ss0 + 696 ;
   double*  hp1  = ss0 + 759 ;
   double*  hd0  = ss0 + 822 ;
   double*  ip0  = ss0 + 948 ;
   double*  id0  = ss0 + 1116 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 3 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 2 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 1 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffpd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  gp0  = ss0 + 633 ;
   double*  gp1  = ss0 + 678 ;
   double*  fd0  = ss0 + 768 ;
   double*  fd1  = ss0 + 828 ;
   double*  ff0  = ss0 + 888 ;
   double*  hp0  = ss0 + 988 ;
   double*  hp1  = ss0 + 1051 ;
   double*  gd0  = ss0 + 1177 ;
   double*  gd1  = ss0 + 1267 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffpf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  hs1  = ss0 + 301 ;
   double*  is0  = ss0 + 406 ;
   double*  sp3  = ss0 + 546 ;
   double*  pp2  = ss0 + 549 ;
   double*  pd2  = ss0 + 567 ;
   double*  dp1  = ss0 + 585 ;
   double*  dd1  = ss0 + 639 ;
   double*  fp0  = ss0 + 711 ;
   double*  fp1  = ss0 + 741 ;
   double*  df1  = ss0 + 831 ;
   double*  gp0  = ss0 + 891 ;
   double*  gp1  = ss0 + 936 ;
   double*  fd0  = ss0 + 1071 ;
   double*  fd1  = ss0 + 1131 ;
   double*  ff0  = ss0 + 1251 ;
   double*  ff1  = ss0 + 1351 ;
   double*  hp0  = ss0 + 1451 ;
   double*  hp1  = ss0 + 1514 ;
   double*  gd0  = ss0 + 1703 ;
   double*  gd1  = ss0 + 1793 ;
   double*  fg0  = ss0 + 1973 ;
   double*  ip0  = ss0 + 2123 ;
   double*  gf0  = ss0 + 2459 ;
   double*  gf1  = ss0 + 2609 ;
   double*  hd0  = ss0 + 2759 ;
   double*  hd1  = ss0 + 2885 ;
   double*  gg0  = ss0 + 3137 ;
   double*  hf0  = ss0 + 3362 ;
   double*  hf1  = ss0 + 3572 ;
   double*  id0  = ss0 + 3782 ;
   double*  hg0  = ss0 + 4286 ;
   double*  if0  = ss0 + 4601 ;
   double*  ig0  = ss0 + 5161 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 5 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 4 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 3 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 2 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 1 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffds(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ps0  = ss0 + 9 ;
   double*  ps2  = ss0 + 15 ;
   double*  ds0  = ss0 + 33 ;
   double*  ds1  = ss0 + 39 ;
   double*  fs0  = ss0 + 75 ;
   double*  fs1  = ss0 + 85 ;
   double*  gs0  = ss0 + 135 ;
   double*  gs1  = ss0 + 150 ;
   double*  hs0  = ss0 + 210 ;
   double*  hs1  = ss0 + 231 ;
   double*  is0  = ss0 + 294 ;
   double*  dp1  = ss0 + 378 ;
   double*  fp0  = ss0 + 396 ;
   double*  fp1  = ss0 + 426 ;
   double*  fd0  = ss0 + 456 ;
   double*  gp0  = ss0 + 516 ;
   double*  gp1  = ss0 + 561 ;
   double*  gd0  = ss0 + 606 ;
   double*  hp0  = ss0 + 696 ;
   double*  hp1  = ss0 + 759 ;
   double*  hd0  = ss0 + 822 ;
   double*  ip0  = ss0 + 948 ;
   double*  id0  = ss0 + 1116 ;
   execute_VRR1_gpu< VTS, 1, 0, 8 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 7 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 6 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 5 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 4 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 3 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 1 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 2 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 2, 1 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 2 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 2, 1 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 2 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 1 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 2 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 1 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffdp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  gp0  = ss0 + 633 ;
   double*  gp1  = ss0 + 678 ;
   double*  fd0  = ss0 + 768 ;
   double*  fd1  = ss0 + 828 ;
   double*  ff0  = ss0 + 888 ;
   double*  hp0  = ss0 + 988 ;
   double*  hp1  = ss0 + 1051 ;
   double*  gd0  = ss0 + 1177 ;
   double*  gd1  = ss0 + 1267 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffdd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  hs1  = ss0 + 301 ;
   double*  is0  = ss0 + 406 ;
   double*  sp3  = ss0 + 546 ;
   double*  pp2  = ss0 + 549 ;
   double*  pd2  = ss0 + 567 ;
   double*  dp1  = ss0 + 585 ;
   double*  dd1  = ss0 + 639 ;
   double*  fp0  = ss0 + 711 ;
   double*  fp1  = ss0 + 741 ;
   double*  df1  = ss0 + 831 ;
   double*  gp0  = ss0 + 891 ;
   double*  gp1  = ss0 + 936 ;
   double*  fd0  = ss0 + 1071 ;
   double*  fd1  = ss0 + 1131 ;
   double*  ff0  = ss0 + 1251 ;
   double*  ff1  = ss0 + 1351 ;
   double*  hp0  = ss0 + 1451 ;
   double*  hp1  = ss0 + 1514 ;
   double*  gd0  = ss0 + 1703 ;
   double*  gd1  = ss0 + 1793 ;
   double*  fg0  = ss0 + 1973 ;
   double*  ip0  = ss0 + 2123 ;
   double*  gf0  = ss0 + 2459 ;
   double*  gf1  = ss0 + 2609 ;
   double*  hd0  = ss0 + 2759 ;
   double*  hd1  = ss0 + 2885 ;
   double*  gg0  = ss0 + 3137 ;
   double*  hf0  = ss0 + 3362 ;
   double*  hf1  = ss0 + 3572 ;
   double*  id0  = ss0 + 3782 ;
   double*  hg0  = ss0 + 4286 ;
   double*  if0  = ss0 + 4601 ;
   double*  ig0  = ss0 + 5161 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 5 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 4 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 3 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 2 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 1 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffdf(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps2  = ss0 + 18 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  hs1  = ss0 + 336 ;
   double*  is0  = ss0 + 462 ;
   double*  sp3  = ss0 + 630 ;
   double*  sd3  = ss0 + 636 ;
   double*  pp2  = ss0 + 642 ;
   double*  pd2  = ss0 + 669 ;
   double*  dp1  = ss0 + 705 ;
   double*  pf2  = ss0 + 777 ;
   double*  dd1  = ss0 + 807 ;
   double*  fp0  = ss0 + 915 ;
   double*  fp1  = ss0 + 945 ;
   double*  df1  = ss0 + 1065 ;
   double*  gp0  = ss0 + 1185 ;
   double*  gp1  = ss0 + 1230 ;
   double*  fd0  = ss0 + 1410 ;
   double*  fd1  = ss0 + 1470 ;
   double*  dg1  = ss0 + 1650 ;
   double*  gd0  = ss0 + 1740 ;
   double*  gd1  = ss0 + 1830 ;
   double*  ff0  = ss0 + 2100 ;
   double*  ff1  = ss0 + 2200 ;
   double*  hp0  = ss0 + 2400 ;
   double*  hp1  = ss0 + 2463 ;
   double*  hd0  = ss0 + 2715 ;
   double*  hd1  = ss0 + 2841 ;
   double*  gf0  = ss0 + 3219 ;
   double*  gf1  = ss0 + 3369 ;
   double*  fg0  = ss0 + 3669 ;
   double*  fg1  = ss0 + 3819 ;
   double*  ip0  = ss0 + 3969 ;
   double*  id0  = ss0 + 4389 ;
   double*  gg0  = ss0 + 5061 ;
   double*  gg1  = ss0 + 5286 ;
   double*  hf0  = ss0 + 5511 ;
   double*  hf1  = ss0 + 5721 ;
   double*  fh0  = ss0 + 6141 ;
   double*  if0  = ss0 + 6351 ;
   double*  hg0  = ss0 + 7191 ;
   double*  hg1  = ss0 + 7506 ;
   double*  gh0  = ss0 + 7821 ;
   double*  ig0  = ss0 + 8136 ;
   double*  hh0  = ss0 + 8976 ;
   double*  ih0  = ss0 + 9417 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 6 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 5 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 4 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 3 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 2 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 5, 1 >( ih0 , ig0 , if0 , hg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fffs(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 10 ;
   double*  ps2  = ss0 + 16 ;
   double*  ds0  = ss0 + 37 ;
   double*  ds1  = ss0 + 43 ;
   double*  fs0  = ss0 + 85 ;
   double*  fs1  = ss0 + 95 ;
   double*  gs0  = ss0 + 155 ;
   double*  gs1  = ss0 + 170 ;
   double*  hs0  = ss0 + 245 ;
   double*  hs1  = ss0 + 266 ;
   double*  is0  = ss0 + 350 ;
   double*  pp2  = ss0 + 462 ;
   double*  dp1  = ss0 + 471 ;
   double*  dd1  = ss0 + 507 ;
   double*  fp0  = ss0 + 543 ;
   double*  fp1  = ss0 + 573 ;
   double*  gp0  = ss0 + 633 ;
   double*  gp1  = ss0 + 678 ;
   double*  fd0  = ss0 + 768 ;
   double*  fd1  = ss0 + 828 ;
   double*  ff0  = ss0 + 888 ;
   double*  hp0  = ss0 + 988 ;
   double*  hp1  = ss0 + 1051 ;
   double*  gd0  = ss0 + 1177 ;
   double*  gd1  = ss0 + 1267 ;
   double*  gf0  = ss0 + 1357 ;
   double*  ip0  = ss0 + 1507 ;
   double*  hd0  = ss0 + 1759 ;
   double*  hd1  = ss0 + 1885 ;
   double*  hf0  = ss0 + 2011 ;
   double*  id0  = ss0 + 2221 ;
   double*  if0  = ss0 + 2557 ;
   execute_VRR1_gpu< VTS, 1, 0, 9 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 8 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 7 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 6 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 5 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 4 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 1 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 2, 1, 2 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 2, 1 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 3 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 4, 1, 3 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 2 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 1 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 3 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 2 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 3, 1 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 3 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 5, 2, 2 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 3, 1 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 2 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 1 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fffp(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 11 ;
   double*  ps2  = ss0 + 17 ;
   double*  ds0  = ss0 + 41 ;
   double*  ds1  = ss0 + 47 ;
   double*  fs0  = ss0 + 95 ;
   double*  fs1  = ss0 + 105 ;
   double*  gs0  = ss0 + 175 ;
   double*  gs1  = ss0 + 190 ;
   double*  hs0  = ss0 + 280 ;
   double*  hs1  = ss0 + 301 ;
   double*  is0  = ss0 + 406 ;
   double*  sp3  = ss0 + 546 ;
   double*  pp2  = ss0 + 549 ;
   double*  pd2  = ss0 + 567 ;
   double*  dp1  = ss0 + 585 ;
   double*  fp0  = ss0 + 639 ;
   double*  fp1  = ss0 + 669 ;
   double*  dd1  = ss0 + 759 ;
   double*  df1  = ss0 + 831 ;
   double*  gp0  = ss0 + 891 ;
   double*  gp1  = ss0 + 936 ;
   double*  fd0  = ss0 + 1071 ;
   double*  fd1  = ss0 + 1131 ;
   double*  ff0  = ss0 + 1251 ;
   double*  ff1  = ss0 + 1351 ;
   double*  hp0  = ss0 + 1451 ;
   double*  hp1  = ss0 + 1514 ;
   double*  gd0  = ss0 + 1703 ;
   double*  gd1  = ss0 + 1793 ;
   double*  fg0  = ss0 + 1973 ;
   double*  ip0  = ss0 + 2123 ;
   double*  gf0  = ss0 + 2459 ;
   double*  gf1  = ss0 + 2609 ;
   double*  hd0  = ss0 + 2759 ;
   double*  hd1  = ss0 + 2885 ;
   double*  gg0  = ss0 + 3137 ;
   double*  hf0  = ss0 + 3362 ;
   double*  hf1  = ss0 + 3572 ;
   double*  id0  = ss0 + 3782 ;
   double*  hg0  = ss0 + 4286 ;
   double*  if0  = ss0 + 4601 ;
   double*  ig0  = ss0 + 5161 ;
   execute_VRR1_gpu< VTS, 1, 0, 10 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 9 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 8 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 7 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 6 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 5 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 1 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 1, 1, 2 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 1 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 3 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR6_gpu< VTS, 3, 1, 4 >( fp0 , fs0 , ds1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 2, 2, 2 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 1 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 4 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 3 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 3, 2 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 4 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 3 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 4, 1 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 4 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 2 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 3 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 4, 1 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 2 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 3 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 4, 1 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 2 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 1 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void fffd(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 12 ;
   double*  ps2  = ss0 + 18 ;
   double*  ds0  = ss0 + 45 ;
   double*  ds1  = ss0 + 51 ;
   double*  fs0  = ss0 + 105 ;
   double*  fs1  = ss0 + 115 ;
   double*  gs0  = ss0 + 195 ;
   double*  gs1  = ss0 + 210 ;
   double*  hs0  = ss0 + 315 ;
   double*  hs1  = ss0 + 336 ;
   double*  is0  = ss0 + 462 ;
   double*  sp3  = ss0 + 630 ;
   double*  sd3  = ss0 + 636 ;
   double*  pp2  = ss0 + 642 ;
   double*  pd2  = ss0 + 669 ;
   double*  dp1  = ss0 + 705 ;
   double*  pf2  = ss0 + 777 ;
   double*  dd1  = ss0 + 807 ;
   double*  fp0  = ss0 + 915 ;
   double*  fp1  = ss0 + 945 ;
   double*  df1  = ss0 + 1065 ;
   double*  gp0  = ss0 + 1185 ;
   double*  gp1  = ss0 + 1230 ;
   double*  fd0  = ss0 + 1410 ;
   double*  fd1  = ss0 + 1470 ;
   double*  dg1  = ss0 + 1650 ;
   double*  gd0  = ss0 + 1740 ;
   double*  gd1  = ss0 + 1830 ;
   double*  ff0  = ss0 + 2100 ;
   double*  ff1  = ss0 + 2200 ;
   double*  hp0  = ss0 + 2400 ;
   double*  hp1  = ss0 + 2463 ;
   double*  hd0  = ss0 + 2715 ;
   double*  hd1  = ss0 + 2841 ;
   double*  ip0  = ss0 + 3219 ;
   double*  gf0  = ss0 + 3639 ;
   double*  gf1  = ss0 + 3789 ;
   double*  fg0  = ss0 + 4089 ;
   double*  fg1  = ss0 + 4239 ;
   double*  id0  = ss0 + 4389 ;
   double*  hf0  = ss0 + 5061 ;
   double*  hf1  = ss0 + 5271 ;
   double*  gg0  = ss0 + 5691 ;
   double*  gg1  = ss0 + 5916 ;
   double*  fh0  = ss0 + 6141 ;
   double*  if0  = ss0 + 6351 ;
   double*  hg0  = ss0 + 7191 ;
   double*  hg1  = ss0 + 7506 ;
   double*  gh0  = ss0 + 7821 ;
   double*  ig0  = ss0 + 8136 ;
   double*  hh0  = ss0 + 8976 ;
   double*  ih0  = ss0 + 9417 ;
   execute_VRR1_gpu< VTS, 1, 0, 11 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 10 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 9 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 8 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 7 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 6 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 2 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 1 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 3 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 2, 2 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 4 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 1 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 3 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 5 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 3, 2 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 5 >( gp0 , gs0 , fs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 3, 2, 4 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 1 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 2, 4 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 3 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 5 >( hp0 , hs0 , gs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 5, 2, 4 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 5 >( ip0 , is0 , hs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 3, 3 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 2 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 2, 4 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 3 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 2 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 5, 1 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 3, 3 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 2 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 5, 1 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 4, 2 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 5, 1 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 5, 1 >( ih0 , ig0 , if0 , hg1 , QC_WQ, zcd );
}

template< int VTS >
__device__ void ffff(double * ss0 , double para[4*3+5] ){
   double * PA_WP = para;
   double * QC_WQ = para + 6;
   double * zab   = para + 12;
   double * zcd   = para + 14;
   double   z     = para[16];
   double*  ss3  = ss0 + 3 ;
   double*  ps0  = ss0 + 13 ;
   double*  ps2  = ss0 + 19 ;
   double*  ds0  = ss0 + 49 ;
   double*  ds1  = ss0 + 55 ;
   double*  fs0  = ss0 + 115 ;
   double*  fs1  = ss0 + 125 ;
   double*  gs0  = ss0 + 215 ;
   double*  gs1  = ss0 + 230 ;
   double*  hs0  = ss0 + 350 ;
   double*  hs1  = ss0 + 371 ;
   double*  is0  = ss0 + 518 ;
   double*  sp3  = ss0 + 714 ;
   double*  sd3  = ss0 + 723 ;
   double*  pp2  = ss0 + 735 ;
   double*  sf3  = ss0 + 771 ;
   double*  pd2  = ss0 + 781 ;
   double*  dp1  = ss0 + 835 ;
   double*  pf2  = ss0 + 925 ;
   double*  dd1  = ss0 + 985 ;
   double*  fp0  = ss0 + 1129 ;
   double*  fp1  = ss0 + 1159 ;
   double*  pg2  = ss0 + 1309 ;
   double*  df1  = ss0 + 1354 ;
   double*  fd0  = ss0 + 1534 ;
   double*  fd1  = ss0 + 1594 ;
   double*  gp0  = ss0 + 1834 ;
   double*  gp1  = ss0 + 1879 ;
   double*  dg1  = ss0 + 2104 ;
   double*  ff0  = ss0 + 2284 ;
   double*  ff1  = ss0 + 2384 ;
   double*  hp0  = ss0 + 2684 ;
   double*  hp1  = ss0 + 2747 ;
   double*  gd0  = ss0 + 3062 ;
   double*  gd1  = ss0 + 3152 ;
   double*  dh1  = ss0 + 3512 ;
   double*  hd0  = ss0 + 3638 ;
   double*  hd1  = ss0 + 3764 ;
   double*  gf0  = ss0 + 4268 ;
   double*  gf1  = ss0 + 4418 ;
   double*  fg0  = ss0 + 4868 ;
   double*  fg1  = ss0 + 5018 ;
   double*  ip0  = ss0 + 5318 ;
   double*  fh0  = ss0 + 5822 ;
   double*  fh1  = ss0 + 6032 ;
   double*  id0  = ss0 + 6242 ;
   double*  hf0  = ss0 + 7082 ;
   double*  hf1  = ss0 + 7292 ;
   double*  gg0  = ss0 + 7922 ;
   double*  gg1  = ss0 + 8147 ;
   double*  gh0  = ss0 + 8597 ;
   double*  gh1  = ss0 + 8912 ;
   double*  fi0  = ss0 + 9227 ;
   double*  if0  = ss0 + 9507 ;
   double*  hg0  = ss0 + 10627 ;
   double*  hg1  = ss0 + 10942 ;
   double*  gi0  = ss0 + 11572 ;
   double*  ig0  = ss0 + 11992 ;
   double*  hh0  = ss0 + 13252 ;
   double*  hh1  = ss0 + 13693 ;
   double*  ih0  = ss0 + 14134 ;
   double*  hi0  = ss0 + 15310 ;
   double*  ii0  = ss0 + 15898 ;
   execute_VRR1_gpu< VTS, 1, 0, 12 >( ps0 , ss0 , PA_WP);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 11 >( ds0 , ps0 , ss0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 10 >( fs0 , ds0 , ps0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 4, 0, 9 >( gs0 , fs0 , ds0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 5, 0, 8 >( hs0 , gs0 , fs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 6, 0, 7 >( is0 , hs0 , gs0 , PA_WP, zab );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR1_gpu< VTS, 1, 0, 3 >( sp3 , ss3 , QC_WQ);
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 2, 0, 2 >( sd3 , sp3 , ss3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 1, 1, 4 >( pp2 , ps2 , ss3 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR2_gpu< VTS, 3, 0, 1 >( sf3 , sd3 , sp3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 1, 2, 3 >( pd2 , pp2 , ps2 , sp3 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 2, 1, 5 >( dp1 , ds1 , ps2 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 3, 2 >( pf2 , pd2 , pp2 , sd3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 2, 4 >( dd1 , dp1 , ds1 , pp2 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 3, 1, 6 >( fp0 , fs0 , ds1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 1, 4, 1 >( pg2 , pf2 , pd2 , sf3 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 2, 3, 3 >( df1 , dd1 , dp1 , pd2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 2, 5 >( fd0 , fp0 , fs0 , dp1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 4, 1, 6 >( gp0 , gs0 , fs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 4, 2 >( dg1 , df1 , dd1 , pf2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 3, 4 >( ff0 , fd0 , fp0 , dd1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 5, 1, 6 >( hp0 , hs0 , gs1 , QC_WQ, z );
   execute_VRR5_gpu< VTS, 4, 2, 5 >( gd0 , gp0 , gs0 , fp1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 2, 5, 1 >( dh1 , dg1 , df1 , pg2 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 2, 5 >( hd0 , hp0 , hs0 , gp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 3, 4 >( gf0 , gd0 , gp0 , fd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 4, 3 >( fg0 , ff0 , fd0 , df1 , QC_WQ, zcd );
   execute_VRR6_gpu< VTS, 6, 1, 6 >( ip0 , is0 , hs1 , QC_WQ, z );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 3, 5, 2 >( fh0 , fg0 , ff0 , dg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 2, 5 >( id0 , ip0 , is0 , hp1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 3, 4 >( hf0 , hd0 , hp0 , gd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 4, 4, 3 >( gg0 , gf0 , gd0 , ff1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 5, 2 >( gh0 , gg0 , gf0 , fg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 3, 6, 1 >( fi0 , fh0 , fg0 , dh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 3, 4 >( if0 , id0 , ip0 , hd1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 4, 3 >( hg0 , hf0 , hd0 , gf1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 4, 6, 1 >( gi0 , gh0 , gg0 , fh1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 6, 4, 3 >( ig0 , if0 , id0 , hf1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 5, 2 >( hh0 , hg0 , hf0 , gg1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 5, 2 >( ih0 , ig0 , if0 , hg1 , QC_WQ, zcd );
   execute_VRR5_gpu< VTS, 5, 6, 1 >( hi0 , hh0 , hg0 , gh1 , QC_WQ, zcd );
  if constexpr ( VTS > CUDA_WARPSIZE ){ __syncthreads(); }
   execute_VRR5_gpu< VTS, 6, 6, 1 >( ii0 , ih0 , ig0 , hh1 , QC_WQ, zcd );
}
















































#include <vector>

template < int VTS >
__device__ constexpr auto get_vrr_kernel(int Lindex ){
   void (*(vrr_kernels[4*4*4*4]))(double *, double*) = { 
      ssss<VTS>,sssp<VTS>,sssd<VTS>,sssf<VTS>,ssps<VTS>,sspp<VTS>,sspd<VTS>,sspf<VTS>,ssds<VTS>,ssdp<VTS>,ssdd<VTS>,ssdf<VTS>,ssfs<VTS>,ssfp<VTS>,ssfd<VTS>,ssff<VTS>,
      spss<VTS>,spsp<VTS>,spsd<VTS>,spsf<VTS>,spps<VTS>,sppp<VTS>,sppd<VTS>,sppf<VTS>,spds<VTS>,spdp<VTS>,spdd<VTS>,spdf<VTS>,spfs<VTS>,spfp<VTS>,spfd<VTS>,spff<VTS>,
      sdss<VTS>,sdsp<VTS>,sdsd<VTS>,sdsf<VTS>,sdps<VTS>,sdpp<VTS>,sdpd<VTS>,sdpf<VTS>,sdds<VTS>,sddp<VTS>,sddd<VTS>,sddf<VTS>,sdfs<VTS>,sdfp<VTS>,sdfd<VTS>,sdff<VTS>,
      sfss<VTS>,sfsp<VTS>,sfsd<VTS>,sfsf<VTS>,sfps<VTS>,sfpp<VTS>,sfpd<VTS>,sfpf<VTS>,sfds<VTS>,sfdp<VTS>,sfdd<VTS>,sfdf<VTS>,sffs<VTS>,sffp<VTS>,sffd<VTS>,sfff<VTS>,
      psss<VTS>,pssp<VTS>,pssd<VTS>,pssf<VTS>,psps<VTS>,pspp<VTS>,pspd<VTS>,pspf<VTS>,psds<VTS>,psdp<VTS>,psdd<VTS>,psdf<VTS>,psfs<VTS>,psfp<VTS>,psfd<VTS>,psff<VTS>,
      ppss<VTS>,ppsp<VTS>,ppsd<VTS>,ppsf<VTS>,ppps<VTS>,pppp<VTS>,pppd<VTS>,pppf<VTS>,ppds<VTS>,ppdp<VTS>,ppdd<VTS>,ppdf<VTS>,ppfs<VTS>,ppfp<VTS>,ppfd<VTS>,ppff<VTS>,
      pdss<VTS>,pdsp<VTS>,pdsd<VTS>,pdsf<VTS>,pdps<VTS>,pdpp<VTS>,pdpd<VTS>,pdpf<VTS>,pdds<VTS>,pddp<VTS>,pddd<VTS>,pddf<VTS>,pdfs<VTS>,pdfp<VTS>,pdfd<VTS>,pdff<VTS>,
      pfss<VTS>,pfsp<VTS>,pfsd<VTS>,pfsf<VTS>,pfps<VTS>,pfpp<VTS>,pfpd<VTS>,pfpf<VTS>,pfds<VTS>,pfdp<VTS>,pfdd<VTS>,pfdf<VTS>,pffs<VTS>,pffp<VTS>,pffd<VTS>,pfff<VTS>,
      dsss<VTS>,dssp<VTS>,dssd<VTS>,dssf<VTS>,dsps<VTS>,dspp<VTS>,dspd<VTS>,dspf<VTS>,dsds<VTS>,dsdp<VTS>,dsdd<VTS>,dsdf<VTS>,dsfs<VTS>,dsfp<VTS>,dsfd<VTS>,dsff<VTS>,
      dpss<VTS>,dpsp<VTS>,dpsd<VTS>,dpsf<VTS>,dpps<VTS>,dppp<VTS>,dppd<VTS>,dppf<VTS>,dpds<VTS>,dpdp<VTS>,dpdd<VTS>,dpdf<VTS>,dpfs<VTS>,dpfp<VTS>,dpfd<VTS>,dpff<VTS>,
      ddss<VTS>,ddsp<VTS>,ddsd<VTS>,ddsf<VTS>,ddps<VTS>,ddpp<VTS>,ddpd<VTS>,ddpf<VTS>,ddds<VTS>,dddp<VTS>,dddd<VTS>,dddf<VTS>,ddfs<VTS>,ddfp<VTS>,ddfd<VTS>,ddff<VTS>,
      dfss<VTS>,dfsp<VTS>,dfsd<VTS>,dfsf<VTS>,dfps<VTS>,dfpp<VTS>,dfpd<VTS>,dfpf<VTS>,dfds<VTS>,dfdp<VTS>,dfdd<VTS>,dfdf<VTS>,dffs<VTS>,dffp<VTS>,dffd<VTS>,dfff<VTS>,
      fsss<VTS>,fssp<VTS>,fssd<VTS>,fssf<VTS>,fsps<VTS>,fspp<VTS>,fspd<VTS>,fspf<VTS>,fsds<VTS>,fsdp<VTS>,fsdd<VTS>,fsdf<VTS>,fsfs<VTS>,fsfp<VTS>,fsfd<VTS>,fsff<VTS>,
      fpss<VTS>,fpsp<VTS>,fpsd<VTS>,fpsf<VTS>,fpps<VTS>,fppp<VTS>,fppd<VTS>,fppf<VTS>,fpds<VTS>,fpdp<VTS>,fpdd<VTS>,fpdf<VTS>,fpfs<VTS>,fpfp<VTS>,fpfd<VTS>,fpff<VTS>,
      fdss<VTS>,fdsp<VTS>,fdsd<VTS>,fdsf<VTS>,fdps<VTS>,fdpp<VTS>,fdpd<VTS>,fdpf<VTS>,fdds<VTS>,fddp<VTS>,fddd<VTS>,fddf<VTS>,fdfs<VTS>,fdfp<VTS>,fdfd<VTS>,fdff<VTS>,
      ffss<VTS>,ffsp<VTS>,ffsd<VTS>,ffsf<VTS>,ffps<VTS>,ffpp<VTS>,ffpd<VTS>,ffpf<VTS>,ffds<VTS>,ffdp<VTS>,ffdd<VTS>,ffdf<VTS>,fffs<VTS>,fffp<VTS>,fffd<VTS>,ffff<VTS>,
   };
   return vrr_kernels[Lindex];
}

// protoype of a v3 with templated global call

template< int NVT, int VTS, int la, int lb, int lc, int ld >
__global__ void compute_VRR_v3_batched_gpu_low(
      const int Ncells, 
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int numV, int numVC, const int Ng ){

   constexpr int L = la+lb+lc+ld;
   constexpr int Li = 64*la+16*lb+4*lc+ld;
   constexpr int F_size = Fsize_but_constexpr(L);
                         
   constexpr auto vrr_kernel = get_vrr_kernel<VTS>(Li);

   int my_vrr_rank = threadIdx.x % VTS ;
   int my_vrr_team = threadIdx.x / VTS ;

   for( int block=blockIdx.x; block < Ncells*Ng ; block += gridDim.x ){

      int p = block / Ng; 
      int n3 = block % Ng;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];

      for ( unsigned i = my_vrr_team; i < n_prm ;  i += NVT ){

//         // Screening on the (ab.n1|cd.n2@n3) fondamental integrals
         bool found = false;
         unsigned int Of = 0;

         double * pr_mem, * pqz;
         __shared__ double PQZ[ NVT * 17 ];

         while ( not found and i < n_prm ){
            Of = ((Ov+i) * Ng + n3 ) * F_size;
            // copy Fm[0] ( the ssss(0) integral ) to AC for later screening in ECO
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            pr_mem[0] = Fm[Of];
            // Immediate screening
            if (Fm[Of] > -1+1.e-30 ){ found = true ; }
            else { i += NVT; }
         }

         if ( found and i < n_prm ){ 

            // Copy the sss(m) integrals
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            for( int il=0; il < L+1; il++ ){ pr_mem[il] = Fm[Of+il]; }
   
            // Copy PA WP QC WQ z1-5 to shared memory for each team
            for( int ii = my_vrr_rank; ii < 17 ; ii += VTS ){ PQZ[my_vrr_team*17+ii] = Fm[Of+L+1+ii]; }
            pqz = &PQZ[my_vrr_team*17];
         }

         __syncthreads();


         if ( found and i < n_prm ){ 
//            if ( my_vrr_rank == 0 ){ 
//               printf(" Tx %d Bx %d | Computing p %d %p \n", threadIdx.x, blockIdx.x, i, pr_mem );
//            }


            vrr_kernel( pr_mem, pqz );
         }
      }
   }   
}







// defines a new type called vrr_global_t
// this type is actually a (global) function, that takes that list 
// as arguments and returns void
typedef void (*(vrr_global_t))( const int , 
      const unsigned int* const __restrict__ ,
      const unsigned int* const __restrict__ ,
      const double* const __restrict__ ,
      const double* const __restrict__ ,
      double* const __restrict__ ,
      double* const __restrict__ ,
      int , int , int , int , const int  ) ;

vrr_global_t get_vrr( int vrr_index ){
   vrr_global_t vrr[4*4*4*4] = {
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 0 , 3 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 1 , 3 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 2 , 3 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 0 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 1 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 2 , 3 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 0 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 1 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 2 , 3 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 0 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 1 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 2 >,
compute_VRR_v3_batched_gpu_low<4,16, 3 , 3 , 3 , 3 >
      };
   return vrr[vrr_index];
}

void compute_VRR_v3(
      const int Ncells, 
      const int vrr_index,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int numV, int numVC, const int Ng, hipStream_t hip_stream ){
   get_vrr( vrr_index )<<<Ncells*Ng, 64, 0, hip_stream >>>( Ncells, PMX,FVH,Fm,data,AC,ABCD,vrr_blocksize,hrr_blocksize,numV,numVC,Ng );
}





__global__ void compute_VRR_v2_batched_gpu_low(
      const int Ncells, const int vrr_kernel_index,
      const unsigned int* const __restrict__ PMX,
      const unsigned int* const __restrict__ FVH,
      const double* const __restrict__ Fm,
      const double* const __restrict__ data,
      double* const __restrict__ AC,
      double* const __restrict__ ABCD,
      int vrr_blocksize, int hrr_blocksize, int L, int numV, int numVC, const int Ng ){
   
   int F_size = Fsize(L);

   constexpr int VTS = 16;
   constexpr int NVT =  4;

   auto vrr_kernel = get_vrr_kernel<VTS>(vrr_kernel_index);

   int my_vrr_rank = threadIdx.x % VTS ;
   int my_vrr_team = threadIdx.x / VTS ;
 
   for( int block=blockIdx.x; block < Ncells*Ng ; block += gridDim.x ){

      unsigned int p = block / Ng; 
      int n3 = block % Ng;

      unsigned int Ov     = FVH[p*FVH_SIZE+FVH_OFFSET_OV];
      unsigned int n_prm  = FVH[p*FVH_SIZE+FVH_OFFSET_NPRM];

      for ( unsigned i = my_vrr_team; i < n_prm ;  i += NVT ){

         // Screening on the (ab.n1|cd.n2@n3) fondamental integrals
         bool found = false;
         unsigned int Of = 0;

         double * pr_mem, * pqz;
         __shared__ double PQZ[ NVT * 17 ];

         while ( not found and i < n_prm ){
            Of = ((Ov+i) * Ng + n3 ) * F_size;
            // copy Fm[0] ( the ssss(0) integral ) to AC for later screening in ECO
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            pr_mem[0] = Fm[Of];
            // Immediate screening
            if (Fm[Of] > 1.e-24 ){ found = true ; }
            else { i += NVT; }
         }

         if ( found and i < n_prm ){ 

            // Copy the sss(m) integrals
            pr_mem = &AC[ ((Ov+i) * Ng + n3) * vrr_blocksize ];
            for( int il=0; il < L+1; il++ ){ pr_mem[il] = Fm[Of+il]; }
   
            // Copy PA WP QC WQ z1-5 to shared memory for each team
            for( int ii = my_vrr_rank; ii < 17 ; ii += VTS ){ PQZ[my_vrr_team*17+ii] = Fm[Of+L+1+ii]; }
            pqz = &PQZ[my_vrr_team*17];
         }

         __syncthreads();

         if ( found and i < n_prm ){

            vrr_kernel( pr_mem, pqz );
         }
//         __syncthreads();
      }
   }
}

#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include "libGint.h"
#include <vector>

extern "C" {

void * libgint_create_handle () {
   libGint * handle;
   handle = new libGint() ;
//#pragma omp critical
//   cout << "Thr " << omp_get_thread_num() << " creating handle " << handle << endl;
   return (void*) handle ;
}

void libgint_init( void * handle ){ 
   libGint * g_handle = ( libGint * ) handle ;
   g_handle -> init();
}

void libgint_set_Potential_Truncated( void * handle, double R_cut_, double * C0_, int ld_C0_, int C0_size_ ){
   libGint * g_handle = ( libGint * ) handle ;
   g_handle -> set_Potential_Truncated( R_cut_, C0_, ld_C0_, C0_size_ );
}

void libgint_set_hf_fac( void * handle, double fac ){
   libGint * g_handle = ( libGint * ) handle ;
//#pragma omp critical
//   cout << " Set hf Thr " << omp_get_thread_num() << " using handle " << handle << endl;
   g_handle -> set_hf_fac( fac );
}

void libgint_set_max_mem( void * handle, int max_mem ){
   libGint * g_handle = ( libGint * ) handle ;
   g_handle -> set_max_mem( max_mem );
}


void libgint_set_P( void * handle, double * P, int P_size ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> set_P( P, P_size );
   g_handle -> zero_K( P_size );
}

void libgint_set_P_polarized( void * handle, double * Pa, double * Pb, int P_size ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> set_P( Pa, Pb, P_size );
   g_handle -> zero_K( P_size, P_size );
}

void libgint_set_K( void * handle, double * K, int K_size ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> set_K( K, K_size ); 
}

void libgint_set_K_polarized( void * handle, double * Ka, double * Kb, int K_size ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> set_K( Ka, Kb, K_size ); 
}

void libgint_get_K( void * handle, double * K ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> get_K( K ); 
}

void libgint_get_K_polarized( void * handle, double * Ka, double * Kb ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> get_K( Ka, Kb ); 
}

void libgint_set_cell( void * handle, bool periodic, double * cell_h, double * cell_i ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> set_cell( periodic, cell_h, cell_i );
}

void libgint_set_neighs( void * handle, double * neighs_, int nneighs ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> set_neighs( neighs_, nneighs );
}

void libgint_set_Atom( void * handle, int i, double * R, double * Z, int np){
   libGint * g_handle = (libGint *) handle ;
//   cout << " Setting Atom(set) " << i << " at " << R[0] << " " << R[1] << " " << R[2] << " w Z: | " ;
//   for ( int i = 0 ; i < np ; i++ ){ cout << Z[i] << " " ; }
//   cout << " | " << np << endl;
   g_handle -> set_Atom( i, R, Z, np );
}



double compute_norm_psi( double * K, double * z, int np, int l ){
   double ans = 0.0;
   for ( int i=0; i < np; i++ ){
      for ( int j=0; j < np; j++ ){
         ans += K[i] * K[j] / pow(z[i]+z[j],1.5+l);
      }
   }
   ans *= pow(M_PI,1.5) / pow(2.0,l);
//   cout << " norm at l " << l << " of " ;
//   for (int i=0; i<np;i++){
//      if (i!=0){cout << " + " ;}
//      cout << K[i] << " ^ " << z[i] << " " ;
//   }
//   cout << "[" << np << "] = " << sqrt(ans) << endl ;
   return sqrt(ans);

}


void libgint_set_Atom_L( void * handle, int i, int l, int nl, double * K ){
   libGint * g_handle = (libGint *) handle ;

   int np =  g_handle -> np[i];
   double * z = &( g_handle -> ua.internal_buffer[g_handle -> idx_Z[i]] );

   std::vector<double> K_normalized( nl * np );
   for ( int inl=0; inl < nl ; inl++ ){
      double norm = compute_norm_psi( &K[inl * np], z, np, l );
      for ( int ip = 0; ip < np ; ip ++ ){
         K_normalized[inl * np + ip] = K[inl * np + ip] / norm ;
      }
   }

//   cout << " Setting Atom L (set) " << " as " << i << " " << l << " " << nl << "x" << np << " K: " << endl;
//   for ( int inl=0; inl < nl ; inl++ ){
//      for ( int ip = 0; ip < np ; ip ++ ){
//         cout << K[ inl * np + ip ] << " ( " <<  K_normalized[inl * np + ip] << " ) " ;
//      } cout << endl ;
//   } cout << endl;



   g_handle -> set_Atom_L( i, l, nl, K_normalized.data() );
}

void libgint_set_AtomInfo( void * handle, int i, double * R, double * Z, int np, int lmin, int Lmax, int * nl, double * K ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> set_Atom( i, R, Z, np );
   int k_offset = 0 ;
   for ( int l = lmin; l < Lmax; l++ ){
      g_handle -> set_Atom_L( i, l, nl[i-lmin], &K[k_offset] );
      k_offset += nl[i-lmin] * np;
   }
}

void libgint_add_prm( void * handle, int ipa, int ipb, int ipc, int ipd ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> add_prm( ipa, ipb, ipc, ipd );
}

void libgint_add_shell( void * handle, int i, int j ,int k, int l, int n1, int n2 ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> add_shell( i,j,k,l,n1,n2 );
}

void libgint_add_cell( void * handle ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> add_cell();
}

void libgint_add_qrt( void * handle, int la, int lb, int lc, int ld, int nla, int nlb, int nlc, int nld ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> add_qrt( la, lb, lc, ld, nla, nlb, nlc, nld );
}

void libgint_add_qrtt(
      void * handle, double symm_fac, int la, int lb, int lc, int ld,
      int inla, int inlb, int inlc, int inld, 
      int ld_ac, int ld_ad, int ld_bc, int ld_bd, 
      unsigned int offset_ac_L_set, unsigned int offset_ad_L_set,
      unsigned int offset_bc_L_set, unsigned int offset_bd_L_set,
      bool Tac, bool Tad, bool Tbc, bool Tbd ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> add_qrtt( symm_fac, la,lb,lc,ld, inla,inlb,inlc,inld, ld_ac,ld_ad,ld_bc,ld_bd,
                         offset_ac_L_set,offset_ad_L_set,offset_bc_L_set,offset_bd_L_set,
                         Tac, Tad, Tbc, Tbd );
}

void libgint_add_set( void * handle ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> add_set();
}

void libgint_dispatch( void * handle ){
   libGint * g_handle = (libGint *) handle ;
   g_handle -> dispatch( true );
}

} // end of extern "C"

#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include "plan.h"
#include "util.h"
#include "timer.h"
#include "UniqueArray.h"
#include "BW_by_patch.h"
#include "t_c_g0_n.h"
#include "compute_Fm.h"
#include "compute_VRR.h"
#include "compute_VRR2.h"
#include "compute_ECO.h"
#include "compute_HRR.h"
#include "compute_SPH.h"
#include "fgamma.h"
#include "libGint.h"
#include "c2s.h"
#include <omp.h>

#include <iomanip>      // std::setprecision

using std::max;


std::vector<LibGint_shared> libGint::shared_obj_ptr; // static // 

void libGint::show_state(){
   for (unsigned int L : encoded_moments ){
      int la,lb,lc,ld;
      decodeL(L,&la,&lb,&lc,&ld);
      cout << " Moments : " << la << " " << lb << " " << lc << " " << ld << endl;
   }
}


void libGint::init(){
//   PUSH_RANGE("libGint init",1);
   // IF the static (shared over OMP) persistent vector is not enough, (re)create it
   // ELSE read the values (hopefully) already saved there
   if ( shared_obj_ptr.size() < (std::vector<LibGint_shared>::size_type) omp_get_num_threads() ){
   // barrier necessary to avoid thread > 0 coming to check the if after master has already allocated the shared obj memory
#pragma omp barrier
#pragma omp master
      shared_obj_ptr = std::vector<LibGint_shared>(omp_get_num_threads());
#pragma omp barrier
      CUDA_GPU_ERR_CHECK( hipStreamCreate( &hip_stream ));
      CUBLAS_GPU_ERR_CHECK( hipblasCreate(&cublas_handle) );
      CUBLAS_GPU_ERR_CHECK( hipblasSetStream( cublas_handle, hip_stream ));
      shared_obj_ptr[ my_thr ] = { &cublas_handle, &hip_stream };
      potential_type = COULOMB; // default
   } else {
      // Use the persistent vector to populate the class members
      cublas_handle = * ( shared_obj_ptr[my_thr].cublas_handle );
      hip_stream   = * ( shared_obj_ptr[my_thr].hip_stream   );
      // TODO may want to check if we can save other stuff other than just the stram and hipblas.h
      potential_type = COULOMB; // default
   }
//   POP_RANGE; // libGint init
}

void libGint::set_Potential_Truncated( double R_cut_, double * C0_, int ld_C0_, int C0_size_ ){
   C0_size = C0_size_;
   ld_C0 = ld_C0_;
   R_cut = R_cut_;
   C0 = C0_;
   potential_type = TRUNCATED;

#pragma omp single copyprivate(C0_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&C0_dev, C0_size * sizeof(double) ) );
   CUDA_GPU_ERR_CHECK( hipMemcpy( C0_dev, C0, C0_size * sizeof(double), hipMemcpyHostToDevice ));   
   }
   size_t to_patch_size = POT_TRUNC_N1 * POT_TRUNC_N2 * sizeof(int);

#pragma omp single copyprivate(x12_to_patch_low_R, x12_to_patch_high_R)
   {
   x12_to_patch_low_R  = (int*) malloc( to_patch_size );
   x12_to_patch_high_R = (int*) malloc( to_patch_size );
   fill_x12_to_patch( POT_TRUNC_N1, POT_TRUNC_N2, x12_to_patch_low_R, x12_to_patch_high_R );
   }

#pragma omp single copyprivate(x12_to_patch_low_R_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&x12_to_patch_low_R_dev, to_patch_size ) );
   CUDA_GPU_ERR_CHECK( hipMemcpy( x12_to_patch_low_R_dev, x12_to_patch_low_R, to_patch_size, hipMemcpyHostToDevice ));   
   }

#pragma omp single copyprivate(x12_to_patch_high_R_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&x12_to_patch_high_R_dev, to_patch_size ) );
   CUDA_GPU_ERR_CHECK( hipMemcpy( x12_to_patch_high_R_dev, x12_to_patch_high_R, to_patch_size, hipMemcpyHostToDevice ));   
   }

#pragma omp single copyprivate(BW_by_patch_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&BW_by_patch_dev, 207*4 * sizeof(double) ) );
   CUDA_GPU_ERR_CHECK( hipMemcpy( BW_by_patch_dev, BW_by_patch, 207*4 * sizeof(double), hipMemcpyHostToDevice ));
   }
}

void libGint::set_hf_fac(double fac){ hf_fac = fac; }

// Set the max gpu memory available to each MPI rank
// this memory will be shared by all omp threads
// TODO add hook to read fraction mem from cp2k input
void libGint::set_max_mem(int max_mem_){
   size_t max_dat_mem = size_t(max_mem_) * 1024 * 1024 * FRACTION_DAT_MEM;
   size_t max_idx_mem = size_t(max_mem_) * 1024 * 1024 * FRACTION_IDX_MEM;
   max_dat_mem_per_thread = max_dat_mem / Nomp;
   max_idx_mem_per_thread = max_idx_mem / Nomp;
}

void libGint::set_Atom( int i, double* R_, double* Z_, int np_ ){
   // why not pushback ?
   if ( i >= (int) idx_R.size() ){ idx_R.resize(i+1); }
   if ( i >= (int) idx_Z.size() ){ idx_Z.resize(i+1); }
   if ( i >= (int) np.size() ){ np.resize(i+1); }

   unsigned int tmp_idx_R = ua.add( R_, 3 );
   unsigned int tmp_idx_Z = ua.add( Z_, np_ );
   idx_R[i] = tmp_idx_R;
   idx_Z[i] = tmp_idx_Z;
   np[i] = np_;
   if ( i < (int) all_l.size() ) {
      all_l[i].clear();
      all_nl[i].clear();
      all_idx_K[i].clear();
   }
   // precompute the max possible number of primitives in a set
   if ( np_ > max_n_prm ){
      max_n_prm = np_;
      prm_tmp_list.resize( max_n_prm*max_n_prm*max_n_prm*max_n_prm*4 );
      new_prm_tmp_list.resize( max_n_prm*max_n_prm*max_n_prm*max_n_prm*PMX_SIZE );
   }
}

void libGint::set_Atom_L( int i, int l_, int nl_, double* K_ ){
   if ( i >= (int) all_l.size()     ){ all_l.resize( i+1 ); }
   if ( i >= (int) all_nl.size()    ){ all_nl.resize( i+1 ); }
   if ( i >= (int) all_idx_K.size() ){ all_idx_K.resize( i+1 ); }
   all_l[i].push_back(l_);
   all_nl[i].push_back(nl_);
   unsigned int idx_K = ua.add( K_, np[i]*nl_);
//   cout << " Idx K of " << i << " " << l_ << " " << nl_ << " " << K_ << " is " << idx_K << endl; cout.flush();

   if ( unique_K_set.count(idx_K) == 0 ){
//      cout << " Adding " << idx_K << " " << nl_ << "." << np[i] << " as # " << unique_K_list.size() << endl;
      unique_K_list.push_back( idx_K );
      l_from_K_list.push_back( nl_);
      p_from_K_list.push_back( np[i] );
      unique_K_set.insert( idx_K );
   }
   
   all_idx_K[i].push_back( idx_K );
   data_size = ua.internal_buffer.size()*sizeof(double);
   
}

// Add the index of the gaussian exponenets, encoded into a single uint
void libGint::add_prm( const int ipa, const int ipb, const int ipc, const int ipd ){
//   unsigned int piabcd = encode4(ipa,ipb,ipc,ipd);
//   prm_tmp_list[ n_prm ] = piabcd ;
   prm_tmp_list[ n_prm*4+0 ] = ipa;
   prm_tmp_list[ n_prm*4+1 ] = ipb;
   prm_tmp_list[ n_prm*4+2 ] = ipc;
   prm_tmp_list[ n_prm*4+3 ] = ipd;
   n_prm++;
}

void libGint::add_shell ( int i, int j, int k, int l, int n1, int n2 ){

   if (n_prm == 0){ return; }

   int npa = np[i];
   int npb = np[j];
   int npc = np[k];
   int npd = np[l];
   int npcd = npc*npd;
   int npbcd = npb*npcd;
   for ( unsigned int i_prm = 0 ; i_prm < n_prm ; i_prm++ ){
      int ipa = prm_tmp_list[ i_prm*4+0 ];
      int ipb = prm_tmp_list[ i_prm*4+1 ];
      int ipc = prm_tmp_list[ i_prm*4+2 ];
      int ipd = prm_tmp_list[ i_prm*4+3 ];
      new_prm_tmp_list[ i_prm * PMX_SIZE + PMX_OFFSET_IPZN] = encode4(ipa,ipb,ipc,ipd);
      new_prm_tmp_list[ i_prm * PMX_SIZE + PMX_OFFSET_OGP ] = ipa*npbcd + ipb*npcd + ipc*npd + ipd;
   }


   // number of angular moments in the i,j,k,l set
   int nnla = all_l[i].size();
   int nnlb = all_l[j].size();
   int nnlc = all_l[k].size();
   int nnld = all_l[l].size();
   // For each combination of angular moment, add work to be done
   // to the candidate vectors this_set_* 
   // TODO transpose all_idx_K[i][idx_la],all_nl[i][idx_la],all_l[i][idx_la] to all_idx_K_nl_l[i][idx_la]
   for( int idx_la=0; idx_la < nnla; idx_la++ ){
      unsigned int idx_Ka = all_idx_K[i][idx_la];
      int nla = all_nl[i][idx_la];
      int la = all_l[i][idx_la];
      for( int idx_lb=0; idx_lb < nnlb; idx_lb++ ){
         unsigned int idx_Kb = all_idx_K[j][idx_lb];
         int nlb = all_nl[j][idx_lb];
         int lb = all_l[j][idx_lb];
         for( int idx_lc=0; idx_lc < nnlc; idx_lc++ ){
            unsigned int idx_Kc = all_idx_K[k][idx_lc];
            int nlc = all_nl[k][idx_lc];
            int lc = all_l[k][idx_lc];
            for( int idx_ld=0; idx_ld < nnld; idx_ld++ ){
               unsigned int idx_Kd = all_idx_K[l][idx_ld];
               int nld = all_nl[l][idx_ld];
               int ld = all_l[l][idx_ld];

               const unsigned int L = encodeL(la,lb,lc,ld);
               const int labcd = la+lb+lc+ld;
               int F_size = Fsize(labcd);

               const unsigned int Of = this_set_offset_F[L];
               const unsigned int Ov = this_set_offset_V[L];
               const unsigned int Og = this_set_offset_G[L];
               const unsigned int Oq = this_set_offset_Q[L];

               const unsigned int elabcd = encode_shell(nla,nlb,nlc,nld,n1,n2);
               const unsigned int epabcd = encode4(np[i],np[j],np[k],np[l]);
               const unsigned int nlabcd = nla*nlb*nlc*nld;
               const unsigned int npabcd = np[i]*np[j]*np[k]*np[l];

               unsigned int idx_Kabcd = all_idx_Kabcd[ four_uint_tuple(idx_Ka, idx_Kb, idx_Kc, idx_Kd) ];
//               unsigned int idx_Kr = offset_K[L];
               unsigned int idx_Kr = 0;
//               cout << " K " << i << " " << j << " " << k << " " << l << " [" << idx_Kabcd << "] = " << ua.internal_buffer[idx_Kabcd] << endl;

//               TODO FUTURE add ij and kl detailed screening coeff
//               const unsigned int bra_Radii_ptr  = MAX_N_PRM * MAX_N_PRM * ( kind_set[i] * number_of_sets + kind_set[j] );
//               const unsigned int ket_Radii_ptr  = MAX_N_PRM * MAX_N_PRM * ( kind_set[k] * number_of_sets + kind_set[l] );
               // if this is the first time we see this combination
               // of angular moments, prepare and save the plan
               if ( all_moments[L] == false ){
                  std::vector<int> * plan = NULL ;
                  unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
                  plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );
                  all_vrr_blocksize[L] = vrr_blocksize;
                  all_hrr_blocksize[L] = hrr_blocksize;
                  all_moments[L] = true;
                  max_plan_size  = max(max_plan_size,  plan->size());     // cell Gvectors     gamma   C0
                  AUX_size = sizeof(int)*(max_plan_size) + sizeof(double)*(2*9+3*max_ncells+3388+245+C0_size/Nomp);
               }
               // many, many idxs for the calculation of the integrals
               // later they will be copied to the gpu memory
               const unsigned int tmp[FVH_SIZE] = {
                  Ov, Og, Oq, n_prm, idx_R[i], idx_R[j], idx_R[k], idx_R[l],
                  idx_Z[i], idx_Z[j], idx_Z[k], idx_Z[l], idx_Kabcd, idx_Kr,
                  elabcd, epabcd, nlabcd, npabcd };

               this_set_L.insert(L);
               this_set_OF[L].insert(  this_set_OF[L].end(), n_prm, Of );
               this_set_PMX[L].insert( this_set_PMX[L].end(), &new_prm_tmp_list[0], &new_prm_tmp_list[n_prm*PMX_SIZE] );
               this_set_FVH[L].insert( this_set_FVH[L].end(), tmp, tmp+FVH_SIZE );

               this_set_Fm_size[L] += (F_size * n_prm * max_ncells)*sizeof(double);
               this_set_AC_size[L] += all_vrr_blocksize[L] * n_prm * max_ncells * sizeof(double);
               this_set_ABCD_size[L] += all_hrr_blocksize[L] * nlabcd * sizeof(double);
//               this_set_KRL_size[L] += n_prm * nlabcd * sizeof(double) ;

               this_set_idx_mem_needed[L] += (n_prm + n_prm*PMX_SIZE + FVH_SIZE) * sizeof(unsigned int);
               this_set_offset_G[L] += nlabcd;
               this_set_offset_V[L] += n_prm;
               this_set_offset_F[L] ++ ;
//               this_set_offset_K[L] += nlabcd*n_prm ;
               encoded_moments.insert(L);
   }}}}

   cell_in_set++;
   n_prm = 0;
}

void libGint::add_cell() {}

void libGint::add_qrt( int la, int lb, int lc, int ld, int nla, int nlb, int nlc, int nld ){
   unsigned int nlabcd = nla*nlb*nlc*nld;
   unsigned int L = encodeL(la,lb,lc,ld);
   this_set_ABCD0_size[L] = this_set_ABCD0_size[L] + compute_Nc(la,lb,lc,ld) * nlabcd * sizeof(double) ;
   this_set_SPHER_size[L] = this_set_SPHER_size[L] + compute_Ns(la,lb,lc,ld) * nlabcd * sizeof(double) ;

//   this_set_dat_mem_needed[L] += compute_Nc(la,lb,lc,ld) * nlabcd * sizeof(double);
//   this_set_dat_mem_needed[L] += compute_Ns(la,lb,lc,ld) * nlabcd * sizeof(double);
   this_set_offset_Q[L] += nlabcd ;
}

void libGint::add_qrtt(
      double symm_fac, int la, int lb, int lc, int ld, 
      int inla, int inlb, int inlc, int inld,
      int ld_ac, int ld_ad, int ld_bc, int ld_bd, 
      unsigned int offset_ac_L_set, unsigned int offset_ad_L_set, 
      unsigned int offset_bc_L_set, unsigned int offset_bd_L_set, 
      bool Tac, bool Tad, bool Tbc, bool Tbd ){ 

   // The symm_fac can be only 2,1,0.5 or 0.25 (or 0 in dev version). 
   // We assign them to 0,1,2 and 3 (and 4) respectively
   // We encode it here, decode it inside compute_KS

   unsigned int idx_fac = 0 ;
   if ( symm_fac < 0.1  ){ idx_fac = 4 ; }
   else if ( symm_fac < 0.3  ){ idx_fac = 3 ; }
   else if ( symm_fac < 0.75 ){ idx_fac = 2 ; }
   else if ( symm_fac < 1.5  ){ idx_fac = 1 ; }
   else { idx_fac = 0 ; }

   unsigned int L = encodeL(la,lb,lc,ld);
   unsigned int KS_idxs[KS_SIZE] = {0};
   KS_idxs[KS_OFFSET_IDXFAC ] = idx_fac;
   KS_idxs[KS_OFFSET_INLABCD] = encode4(  inla, inlb, inlc, inld );
   KS_idxs[KS_OFFSET_LDABCD ] = encode4( ld_ac,ld_ad,ld_bc,ld_bd );
   KS_idxs[KS_OFFSET_OFFAC  ] = offset_ac_L_set;
   KS_idxs[KS_OFFSET_OFFAD  ] = offset_ad_L_set;
   KS_idxs[KS_OFFSET_OFFBC  ] = offset_bc_L_set;
   KS_idxs[KS_OFFSET_OFFBD  ] = offset_bd_L_set;
   KS_idxs[KS_OFFSET_TALL   ] = encode4( (int)Tac, (int)Tad, (int)Tbc, (int)Tbd );

   this_set_KS[L].insert( this_set_KS[L].end(), KS_idxs, KS_idxs+KS_SIZE );
   this_set_idx_mem_needed[L] += KS_SIZE * sizeof(unsigned int);
//   if ( la == 3 and lb == 3 and lc == 3 and ld == 3 ){
//      cout << la<<lb<<lc<<ld<<" " << offset_ac_L_set << " " << offset_ad_L_set << " " << offset_bc_L_set << " " << offset_bd_L_set << endl;
//   }

}

void libGint::add_set(){
   
//   set_timer.start();
   for ( unsigned int L : this_set_L ){

#ifdef LIBGINT_INTERNAL_DEBUG
      // Check if this set is even possible
      size_t this_set_dat_mem_needed_1 = max( this_set_Fm_size[L], max( this_set_ABCD_size[L], this_set_SPHER_size[L] ));
      size_t this_set_dat_mem_needed_2 = max( this_set_AC_size[L], 2 * this_set_ABCD0_size[L] );
      this_set_dat_mem_needed[L] = this_set_dat_mem_needed_1 + this_set_dat_mem_needed_2;
      if ( this_set_dat_mem_needed[L] >= max_dat_mem_per_thread ){
         cout << " Calculation may fail: this set L: " << L << " requires at least " <<  this_set_dat_mem_needed[L]/1024/1024 << " MiB for dat " 
              << this_set_Fm_size[L] << " " <<  this_set_AC_size[L] << " " <<  this_set_ABCD_size[L] << " " << this_set_ABCD0_size[L] << endl;
      }
      if ( this_set_idx_mem_needed[L] >= max_idx_mem_per_thread ){
         cout << " Calculation may fail: this set L: " << L << " requires at least " <<  this_set_idx_mem_needed[L]/1024/1024 << " MiB for idx" << endl;
      }
#endif // LIBGINT_INTERNAL_DEBUG

      // Check if we can fit this set in the list of sets to do
      size_t Fm_size_L_p = Fm_size[L]+this_set_Fm_size[L];
      size_t AC_size_L_p = AC_size[L]+this_set_AC_size[L];
      size_t ABCD_size_L_p = ABCD_size[L]+this_set_ABCD_size[L];
      size_t ABCD0_size_L_p = ABCD0_size[L]+this_set_ABCD0_size[L];
      size_t SPHER_size_L_p = SPHER_size[L]+this_set_SPHER_size[L];
      size_t dat_mem_needed_L_1 = max( Fm_size_L_p, max( ABCD_size_L_p, SPHER_size_L_p ));
      size_t dat_mem_needed_L_2 = max( AC_size_L_p, 2*ABCD0_size_L_p );
      size_t dat_mem_req_L = dat_mem_needed_L_1 + dat_mem_needed_L_2 ; 
      size_t idx_mem_req_L = idx_mem_needed[L] + this_set_idx_mem_needed[L];
      bool enough_dat_mem_L = (dat_mem_req_L <= max_dat_mem_per_thread);
      bool enough_idx_mem_L = (idx_mem_req_L <= max_idx_mem_per_thread);
      bool enough_mem_L = enough_dat_mem_L and enough_idx_mem_L;

      if ( enough_mem_L ){
         // Move the indices for this set forward, then include them in the larger lists
         for ( unsigned int idx1 = 0; idx1 < this_set_OF[L].size(); idx1++ ){
            this_set_OF[L][idx1] += offset_F[L];
         }
         for ( unsigned int idx2 = 0; idx2 < this_set_FVH[L].size()/FVH_SIZE; idx2++ ){
            this_set_FVH[L][idx2*FVH_SIZE+0] += offset_V[L];
            this_set_FVH[L][idx2*FVH_SIZE+1] += offset_G[L];
            this_set_FVH[L][idx2*FVH_SIZE+2] += offset_Q[L];
         }
         // TODO move to ::add_set_L(L);
         // Gotta love the stl
          OF[L].insert(  OF[L].end(),  this_set_OF[L].begin(),  this_set_OF[L].end() );
         PMX[L].insert( PMX[L].end(), this_set_PMX[L].begin(), this_set_PMX[L].end() );
         FVH[L].insert( FVH[L].end(), this_set_FVH[L].begin(), this_set_FVH[L].end() );
          KS[L].insert(  KS[L].end(),  this_set_KS[L].begin(),  this_set_KS[L].end() );
         // TODO transpose L from fast to slow variable
            Fm_size[L] +=    this_set_Fm_size[L];
            AC_size[L] +=    this_set_AC_size[L];
          ABCD_size[L] +=  this_set_ABCD_size[L];
         ABCD0_size[L] += this_set_ABCD0_size[L];
         SPHER_size[L] += this_set_SPHER_size[L];

         dat_mem_needed_L_1 = max( Fm_size[L], max( ABCD_size[L], SPHER_size[L] ));
         dat_mem_needed_L_2 = max( AC_size[L], 2*ABCD0_size[L] );
         dat_mem_needed[L] = dat_mem_needed_L_1 + dat_mem_needed_L_2;
         assert( dat_mem_needed[L] <= max_dat_mem_per_thread && " Strange. Pre check dat mem failed "); 
//         dat_mem_needed[L] += this_set_dat_mem_needed[L];
         idx_mem_needed[L] += this_set_idx_mem_needed[L];
         this_set_dat_mem_needed[L] = 0;
         this_set_idx_mem_needed[L] = 0;
         offset_V[L] += this_set_offset_V[L];
         offset_F[L] += this_set_offset_F[L];
         offset_G[L] += this_set_offset_G[L];
         offset_Q[L] += this_set_offset_Q[L];
         //
      } else {
//         cout << " L: " << L << " added to after list " << dat_mem_req_L << " " << idx_mem_req_L << endl;
         add_L_after_dispatch.insert(L);
      }
      // we still need to max check to make sure we dispatch early correctly
      max_dat_mem_needed = max(max_dat_mem_needed, dat_mem_req_L);
      max_idx_mem_needed = max(max_idx_mem_needed, idx_mem_req_L);
   }

   bool out_of_dat_mem = max_dat_mem_needed > max_dat_mem_per_thread;
   bool out_of_idx_mem = max_idx_mem_needed > max_idx_mem_per_thread;
   bool out_of_mem = out_of_dat_mem or out_of_idx_mem;

   if ( out_of_mem ){
         dispatch(false);
   }

   for ( unsigned int L : add_L_after_dispatch ){
      // TODO move to ::add_set_L(L);
       OF[L].insert(  OF[L].end(),  this_set_OF[L].begin(),  this_set_OF[L].end() );
      PMX[L].insert( PMX[L].end(), this_set_PMX[L].begin(), this_set_PMX[L].end() );
      FVH[L].insert( FVH[L].end(), this_set_FVH[L].begin(), this_set_FVH[L].end() );
       KS[L].insert(  KS[L].end(),  this_set_KS[L].begin(),  this_set_KS[L].end() );
         Fm_size[L] +=    this_set_Fm_size[L];
         AC_size[L] +=    this_set_AC_size[L];
       ABCD_size[L] +=  this_set_ABCD_size[L];
      ABCD0_size[L] += this_set_ABCD0_size[L];
      SPHER_size[L] += this_set_SPHER_size[L];
      dat_mem_needed[L] += this_set_dat_mem_needed[L];
      idx_mem_needed[L] += this_set_idx_mem_needed[L];
      this_set_dat_mem_needed[L] = 0;
      this_set_idx_mem_needed[L] = 0;
      offset_V[L] += this_set_offset_V[L];
      offset_F[L] += this_set_offset_F[L];
      offset_G[L] += this_set_offset_G[L];
      offset_Q[L] += this_set_offset_Q[L];

      max_dat_mem_needed = max(max_dat_mem_needed, dat_mem_needed[L]);
      max_idx_mem_needed = max(max_idx_mem_needed, idx_mem_needed[L]);
   }
   // TODO move to reset_set
   for ( unsigned int L : this_set_L ){
       this_set_OF[L].clear();
      this_set_PMX[L].clear();
      this_set_FVH[L].clear();
       this_set_KS[L].clear();
         this_set_Fm_size[L] = 0;
         this_set_AC_size[L] = 0;
       this_set_ABCD_size[L] = 0;
      this_set_ABCD0_size[L] = 0;
      this_set_SPHER_size[L] = 0;
      this_set_offset_V[L] = 0;
      this_set_offset_F[L] = 0;
      this_set_offset_G[L] = 0;
      this_set_offset_Q[L] = 0;

      idx_mem_needed[L] = 0;

   }

   add_L_after_dispatch.clear();
   this_set_L.clear();

}


void libGint::set_cell( bool periodic_, double * cell_h_,  double * cell_i_ ){
//#pragma omp critical
//   {cout << " Setting cell " << periodic_ << " " << cell_h[0] << " " << cell_i_[8] << endl; cout.flush();}
   periodic = periodic_;
   for (int i=0;i<9;i++){ cell_h[CELL_HMAT_OFF+i] = cell_h_[i]; }
   for (int i=0;i<9;i++){ cell_h[CELL_HINV_OFF+i] = cell_i_[i]; }
}

void libGint::set_neighs( double * neighs_, int nneighs_ ){
//#pragma omp critical
//   {cout << " Setting neigh " << nneighs_ << " " << neighs_[0] << " -  " << neighs_[3*nneighs_-1] << endl; cout.flush();}
   max_ncells = nneighs_;
   neighs.resize(3*nneighs_);
   for (int i=0;i<3*nneighs_;i++){ neighs[i] = neighs_[i]; }
}


// TODO ? get zero and set could be async ?
void libGint::set_P( double * P_, int P_size ){
   nspin = 1;
   P_a = P_ ;
   FP_size = P_size;

#pragma omp single copyprivate(P_a_dev)
   {
   // if this is the first call, P_a_dev is zero and no operation is performed.
   CUDA_GPU_ERR_CHECK( hipFree(P_a_dev));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&P_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemcpy( P_a_dev, P_a, sizeof(double)*FP_size, hipMemcpyHostToDevice ));  
   }

   FP_size_omp = 2 * nspin * sizeof(double)*(FP_size) / Nomp ;
//#pragma omp critical
//   cout << " Setting P from " << P_ << " to " << P_a_dev << " x " << FP_size << endl;

   allocate_on_GPU();
  
}
void libGint::set_P( std::vector<double> & P_ ){ set_P(P_.data(), P_.size()); }

void libGint::set_P( double * P_a_, double * P_b_, int P_size ){
//   cout << " Setting P Polarized with size " << P_a_.size() << " " << P_b_.size() << endl ;
   nspin = 2;
   P_a = P_a_;
   P_b = P_b_;
   FP_size = P_size;

#pragma omp single copyprivate(P_a_dev) // nowait
   {
   // if this is the first call, P_a_dev is zero and no operation is performed.
   CUDA_GPU_ERR_CHECK( hipFree(P_a_dev));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&P_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemcpy( P_a_dev, P_a_, sizeof(double)*FP_size, hipMemcpyHostToDevice ));
   }
#pragma omp single copyprivate(P_b_dev)
   {
   CUDA_GPU_ERR_CHECK( hipFree(P_b_dev));     
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&P_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemcpy( P_b_dev, P_b_, sizeof(double)*FP_size, hipMemcpyHostToDevice ));
   }

   FP_size_omp = 2 * nspin * sizeof(double)*(FP_size) / Nomp ;
   allocate_on_GPU();
}
void libGint::set_P( std::vector<double> & P_a_, std::vector<double> & P_b_ ){ set_P( P_a_.data(), P_b_.data(), P_a_.size()); }

void libGint::allocate_on_GPU(){
//   PUSH_RANGE("dispatch malloc",1);

   // TODO move at init
   ftable_ld = 0; // ld of table for fgamma === n+1 === nmax + 6 + 1 === 21 + 6 + 1 === 28
   int nmax = 21;
   double tdelta = 0.1;  // hardcoded in source code
   double tmin = 0.; // hardcoded in source code
   double tmax = 12.;  // hardcoded in source code
   int n = nmax + 6; // 6 is the max taylor expansion. Also hardcoded
   int itabmin = floor(tmin/tdelta); // === 0
   int itabmax = int((tmax - tmin)/tdelta); // === 120
   int nelem = (itabmax - itabmin + 1 ) * (n+1); // === 121*(n+1) == 121*ftable_ld === 121*28 === 3388
   double* ftable = create_md_ftable( nmax, tmin, tmax, tdelta, &ftable_ld);

//   cout << " Starting at " << ua.internal_buffer.size() << endl;
   if ( not all_idx_Kabcd_ready ) {
      for ( unsigned int ia = 0; ia < unique_K_list.size(); ia++ ){
      unsigned int iKa = unique_K_list[ia];
      unsigned int nla = l_from_K_list[ia];
      unsigned int npa = p_from_K_list[ia];
      for ( unsigned int ib = 0; ib < unique_K_list.size(); ib++ ){
      unsigned int iKb = unique_K_list[ib];
      unsigned int nlb = l_from_K_list[ib];
      unsigned int npb = p_from_K_list[ib];
      for ( unsigned int ic = 0; ic < unique_K_list.size(); ic++ ){
      unsigned int iKc = unique_K_list[ic];
      unsigned int nlc = l_from_K_list[ic];
      unsigned int npc = p_from_K_list[ic];
      for ( unsigned int id = 0; id < unique_K_list.size(); id++ ){
      unsigned int iKd = unique_K_list[id];
      unsigned int nld = l_from_K_list[id];
      unsigned int npd = p_from_K_list[id];

         double * Ka = &ua.internal_buffer[iKa];
         double * Kb = &ua.internal_buffer[iKb];
         double * Kc = &ua.internal_buffer[iKc];
         double * Kd = &ua.internal_buffer[iKd];

         int nl = nla*nlb*nlc*nld;
         int np = npa*npb*npc*npd;
         std::vector<double> tmp_K( nl * np );

         int ilabcd = 0;
         for ( unsigned int ila=0; ila < nla; ila++ ){
         for ( unsigned int ilb=0; ilb < nlb; ilb++ ){
         for ( unsigned int ilc=0; ilc < nlc; ilc++ ){
         for ( unsigned int ild=0; ild < nld; ild++ ){
            int ipabcd = 0;
            for ( unsigned int ipa=0; ipa < npa ; ipa++){
            for ( unsigned int ipb=0; ipb < npb ; ipb++){
            for ( unsigned int ipc=0; ipc < npc ; ipc++){
            for ( unsigned int ipd=0; ipd < npd ; ipd++){
               tmp_K[ ilabcd * np + ipabcd ]  = Ka[ ila * npa + ipa ];
               tmp_K[ ilabcd * np + ipabcd ] *= Kb[ ilb * npb + ipb ];
               tmp_K[ ilabcd * np + ipabcd ] *= Kc[ ilc * npc + ipc ];
               tmp_K[ ilabcd * np + ipabcd ] *= Kd[ ild * npd + ipd ];
//               cout << " Precomputing K " << ila<<ilb<<ilc<<ild<<"."<<ipa<<ipb<<ipc<<ipd<< " " << tmp_K[ ilabcd * np + ipabcd ] << endl;
               ipabcd++;
            }}}}
            ilabcd++;
         }}}}
      unsigned int iKabcd = ua.add( tmp_K.data(), nl*np );
//      cout << " Saving precomputed K " << nl << "." << np << " from " << iKa << " " << iKb << " " << iKc << " " << iKd << " @ " << iKabcd << endl;
      all_idx_Kabcd[ four_uint_tuple(iKa, iKb, iKc, iKd) ] = iKabcd;
      }}}}
      all_idx_Kabcd_ready = true;
   }
//   cout << " Ending at " << ua.internal_buffer.size() << endl;

   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&data_dev, sizeof(double)*(ua.internal_buffer.size()) ));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&dat_mem_dev, max_dat_mem_per_thread ));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&idx_mem_dev , max_idx_mem_per_thread ));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&cell_h_dev, sizeof(double)*(2*9) ));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&neighs_dev, sizeof(double)*(3*max_ncells) ));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&ftable_dev, sizeof(double)*(nelem) ));
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&C2S_dev, sizeof(double)*245 ));
   // TODO do
   #define max_plan_size_possible 10000000
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&plan_dev,sizeof(int)*max_plan_size_possible ));
   CUDA_GPU_ERR_CHECK( hipHostAlloc( (void**)&idx_mem_stg, max_idx_mem_per_thread, hipHostMallocPortable ));
   CUDA_GPU_ERR_CHECK( hipHostAlloc( (void**)&plan_stg, sizeof(int)*max_plan_size_possible, hipHostMallocPortable ));
   CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//   POP_RANGE; // dispatch malloc

//   PUSH_RANGE("dispatch memcpy",1);
   CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
      data_dev, ua.internal_buffer.data(), sizeof(double)*(ua.internal_buffer.size()), hipMemcpyHostToDevice, hip_stream ));
   CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
      cell_h_dev, cell_h, sizeof(double)*(2*9), hipMemcpyHostToDevice, hip_stream ));
   CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
      neighs_dev, neighs.data(), sizeof(double)*(3*max_ncells), hipMemcpyHostToDevice, hip_stream )); 
   CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
      ftable_dev, ftable, sizeof(double)*(nelem), hipMemcpyHostToDevice, hip_stream ));
   CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
      C2S_dev, c2s, sizeof(double)*245, hipMemcpyHostToDevice, hip_stream ));
//   POP_RANGE; // dispatch memcpy
//   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
   // ! needed after async memcpy TODO move to dispatch
//   CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
   CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
}

void libGint::free_on_GPU(){
   // TODO move to some resize / delete function at get_K time
   CUDA_GPU_ERR_CHECK( hipFree(data_dev) );
   CUDA_GPU_ERR_CHECK( hipFree(dat_mem_dev) );
   CUDA_GPU_ERR_CHECK( hipFree(idx_mem_dev) ); 
   CUDA_GPU_ERR_CHECK( hipFree(cell_h_dev) );
   CUDA_GPU_ERR_CHECK( hipFree(neighs_dev) );
   CUDA_GPU_ERR_CHECK( hipFree(ftable_dev) );
   CUDA_GPU_ERR_CHECK( hipFree(C2S_dev) );
   CUDA_GPU_ERR_CHECK( hipFree(plan_dev) );
   CUDA_GPU_ERR_CHECK( hipHostFree(idx_mem_stg) );
   CUDA_GPU_ERR_CHECK( hipHostFree(plan_stg) );
#pragma omp single
   CUDA_GPU_ERR_CHECK( hipFree(K_a_dev));
#pragma omp single
   CUDA_GPU_ERR_CHECK( hipFree(P_a_dev));
   if ( nspin == 2 ){
#pragma omp single
      CUDA_GPU_ERR_CHECK( hipFree(K_b_dev));
#pragma omp single
      CUDA_GPU_ERR_CHECK( hipFree(P_b_dev));
   }
}

void libGint::zero_K( int K_size ){
   assert( nspin == 1 );
   FP_size = K_size;
#pragma omp single copyprivate(K_a_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemset( K_a_dev, 0, sizeof(double)*FP_size ));
   }
}

void libGint::zero_K( int K_size, int K_size_ ){
   assert( nspin == 2 );
   assert( K_size == K_size_ );
   FP_size = K_size;
#pragma omp single copyprivate(K_a_dev) // nowait
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemset( K_a_dev, 0, sizeof(double)*FP_size ));
   }
#pragma omp single copyprivate(K_b_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&K_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemset( K_b_dev, 0, sizeof(double)*FP_size ));
   }
}

void libGint::set_K( double * K_ , int K_size ){
   assert( nspin == 1 );
   K_a = K_;
   FP_size = K_size;

#pragma omp single copyprivate(K_a_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemcpy( K_a_dev, K_, sizeof(double)*FP_size, hipMemcpyHostToDevice ));
   }
}

void libGint::set_K( std::vector<double> & K_ ){ set_K( K_.data(), K_.size()); }

void libGint::set_K( double * K_a_, double * K_b_, int K_size ){
   assert( nspin == 2 );
   K_a = K_a_;
   K_b = K_b_;
   FP_size = K_size;
#pragma omp single copyprivate(K_a_dev) // nowait
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&K_a_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemcpy( K_a_dev, K_a_, sizeof(double)*FP_size, hipMemcpyHostToDevice ));
   }
#pragma omp single copyprivate(K_b_dev)
   {
   CUDA_GPU_ERR_CHECK( hipMalloc( (void**)&K_b_dev, sizeof(double)*FP_size ));
   CUDA_GPU_ERR_CHECK( hipMemcpy( K_b_dev, K_b_, sizeof(double)*FP_size, hipMemcpyHostToDevice ));
   }
}
void libGint::set_K( std::vector<double> & K_a_ , std::vector<double> & K_b_ ){ set_K( K_a_.data(), K_b_.data(), K_a_.size()); }

void libGint::get_K( double * K_ ){
   assert( nspin == 1 );
   // make sure every thread is done with its calculations
//   CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//   CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
   dispatch(true);
//   CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//   CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
#pragma omp barrier

#pragma omp single
   CUDA_GPU_ERR_CHECK( hipMemcpy( K_, K_a_dev, sizeof(double)*FP_size, hipMemcpyDeviceToHost ));

   free_on_GPU();

//#pragma omp critical
//   cout << " Getting K from " << K_a_dev << " to " << K_ << " x " << FP_size << endl;

}

void libGint::get_K( std::vector<double> & K_a_ ){ get_K( K_a_.data()); }

void libGint::get_K( double * K_a_,  double * K_b_ ){
   assert( nspin == 2 );
   // make sure every thread is done with its calculations
   dispatch(true);
#pragma omp barrier
 
#pragma omp single nowait
   CUDA_GPU_ERR_CHECK( hipMemcpy( K_a_, K_a_dev, sizeof(double)*FP_size, hipMemcpyDeviceToHost ));
#pragma omp single
   CUDA_GPU_ERR_CHECK( hipMemcpy( K_b_, K_b_dev, sizeof(double)*FP_size, hipMemcpyDeviceToHost ));

   free_on_GPU();

}
void libGint::get_K( std::vector<double> & K_a_,  std::vector<double> & K_b_ ){ get_K( K_a_.data(), K_b_.data()); }

void libGint::reset_indices(unsigned int L){

    OF[L].clear();
   PMX[L].clear();
   FVH[L].clear();
    KS[L].clear();

   offset_F[L] = 0;
   offset_V[L] = 0;
   offset_G[L] = 0;
   offset_Q[L] = 0;

   Fm_size[L] = 0;
   AC_size[L] = 0;
   ABCD_size[L] = 0;
   ABCD0_size[L] = 0;
   SPHER_size[L] = 0;

//   n_set = 0;
//   prm_in_set = 0;
//   n_prm = 0;
//   p0 = 0;
//   cell_in_set = 0;

//   encoded_moments.erase(L);

}

void libGint::dispatch( bool dispatch_all ){

//   dis_timer.start();
//   #pragma omp critical
//   { cout << "Dispatch on stream " << hip_stream << " @ " << &hip_stream << " F: " << dispatch_all << endl; cout.flush(); }

//   CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//   CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//   CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );


//   #pragma omp critical
//   { cout << "Dispatch: at max dat " << max_dat_mem_needed/1024/1024 << " MiB values at thr " << omp_get_thread_num() << endl; cout.flush(); }
//   { cout << "Dispatch: at max idx " << max_idx_mem_needed << " B values at thr " << omp_get_thread_num() << endl; cout.flush(); }
   
//   OUT.resize(out_size);

   // Main cycle.
   // 0) Decide if we are to run this L
   // 1) Get the plan
   // 2) Copy the input vectors to device memory
   // 3) Run

   // There is a (small) chance we get to this point before the previous dispatch on the same stream
   // has finished. To avoid problems, we wait
   CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );

   // Flag to sync before we overwrite the RAM side idx
   bool first_loop_this_dispatch = true;

//   PUSH_RANGE("dispatch all L",3);
   for ( unsigned int L : encoded_moments ){
 
//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );

  
      // Do we have to compute this batch of integrals:
      // 2 possible reasons: 
      // (1) We have asked to dispatch all, likely because we got a get_K request
      // (2) adding the next set would mean the the next time we try the batch would require
      //       too much memory
      bool must_be_computed = dispatch_all or (add_L_after_dispatch.count(L) > 0);
      bool can_be_skipped = not must_be_computed;

      // Early exit moments with a small number of integrals
      // No worry, they are guaranteed to be computed before get_K returns
      bool is_too_small = ABCD_size[L] < MIN_INT_BATCH_SIZE;
      if ( can_be_skipped and is_too_small ) { continue; }
      if ( SPHER_size[L] == 0 ){ continue; }

      size_t dat_mem_needed_L_1 = max( Fm_size[L], max( ABCD_size[L], SPHER_size[L] ));
      size_t idx_mem_needed_L = OF[L].size() + PMX[L].size() + FVH[L].size() + KS[L].size();
#ifdef LIBGINT_INTERNAL_DEBUG
      size_t dat_mem_needed_L_2 = max( AC_size[L], 2*ABCD0_size[L] );
      size_t dat_mem_needed_L = dat_mem_needed_L_1 + dat_mem_needed_L_2; 
//      cout << L << " " << dat_mem_needed_L << " . " << dat_mem_needed[L] << " . " << max_dat_mem_per_thread << endl;
      assert( dat_mem_needed_L <= max_dat_mem_per_thread && " Uff. Check dat mem failed "); 
      assert( idx_mem_needed_L <= max_idx_mem_per_thread && " Uff. Check idx mem failed "); 
#endif // LIBGINT_INTERNAL_DEBUG

      int la,lb,lc,ld,labcd;
      decodeL(L,&la,&lb,&lc,&ld);
      labcd = la+lb+lc+ld;
      int Nc = compute_Nc(la,lb,lc,ld);
      int Ns = compute_Ns(la,lb,lc,ld);

//      std::string Lname = std::to_string(la) + "_" + std::to_string(lb) + "_" + std::to_string(lc) + "_" + std::to_string(ld);
//      cout << "Running " << Lname << " " << SPHER_size[L] << endl;

      std::vector<int> * plan = NULL ;
      unsigned int vrr_blocksize, hrr_blocksize, numV, numVC, numVCH;
      plans.get( la, lb, lc, ld, &plan, &vrr_blocksize, &hrr_blocksize, &numV, &numVC, &numVCH );
      unsigned int Nop = numVC - numV + 1;

      unsigned int Nprm   = offset_V[L];
      unsigned int Ncells = offset_F[L];
      unsigned int Nqrtt  = offset_Q[L];

      double* Fm_dev    = &dat_mem_dev[0];
      double* AC_dev    = &dat_mem_dev[0] + dat_mem_needed_L_1 / sizeof(double);
      double* ABCD_dev  = &dat_mem_dev[0];
      double* ABCD0_dev = &dat_mem_dev[0] + dat_mem_needed_L_1 / sizeof(double);
      double* SPTMP_dev = &dat_mem_dev[0] + dat_mem_needed_L_1 / sizeof(double) + ABCD0_size[L] / sizeof(double);
      double* SPHER_dev = &dat_mem_dev[0]; 

      unsigned int*  OF_dev = &idx_mem_dev[0];
      unsigned int* PMX_dev =  OF_dev +  OF[L].size();
      unsigned int* FVH_dev = PMX_dev + PMX[L].size();
      unsigned int*  KS_dev = FVH_dev + FVH[L].size();

      unsigned int*  OF_stg = &idx_mem_stg[0];
      unsigned int* PMX_stg =  OF_stg +  OF[L].size();
      unsigned int* FVH_stg = PMX_stg + PMX[L].size();
      unsigned int*  KS_stg = FVH_stg + FVH[L].size();


//#pragma omp critical
//      {
//      size_t tot_mem_L = Fm_size[L] + AC_size[L] + ABCD_size[L] + ABCD0_size[L] + SPHER_size[L];
      if ( false ){
      cout << " L " << la << "" << lb << "" << lc << "" << ld << " | ";
      cout << Nprm << " prms " << Ncells << " cells " << Nqrtt << " qrtts " << max_ncells << " Ng " ;
      cout <<    Fm_size[L]/1024/1024 << " " ;
      cout <<    AC_size[L]/1024/1024 << " " ;
      cout <<  ABCD_size[L]/1024/1024 << " ";
      cout << ABCD0_size[L]/1024/1024 << " ";
      cout << SPHER_size[L]/1024/1024 << " | ";
      cout << dispatch_all << " " << must_be_computed << " " << is_too_small << " " << first_loop_this_dispatch << " | ";
      cout << dat_mem_needed_L_1/1024/1024 << " MB | " << dat_mem_needed_L_2/1024/1024 << " MB | ";
      cout << endl;
      }
//      }

//      PUSH_RANGE(Lname.c_str(),3);

//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );

      // it is (very) possible that we reach this point before the previous loop completed, so we sync
      // before overwriting index arrays on device
      // TODO ? not necessary ?
      if ( not first_loop_this_dispatch ){
//         cout << " SYNCING " << endl;
         CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
      } else {
         first_loop_this_dispatch = false;
      }

      // Stage the idx array for async copy on device
      memcpy(  OF_stg,  OF[L].data(), sizeof(unsigned int)*( OF[L].size()) );
      memcpy( PMX_stg, PMX[L].data(), sizeof(unsigned int)*(PMX[L].size()) );
      memcpy( FVH_stg, FVH[L].data(), sizeof(unsigned int)*(FVH[L].size()) );
      memcpy(  KS_stg,  KS[L].data(), sizeof(unsigned int)*( KS[L].size()) );
      memcpy(plan_stg,  plan->data(), sizeof(int)*( plan->size()) );

      CUDA_GPU_ERR_CHECK( hipMemcpyAsync( 
         idx_mem_dev, idx_mem_stg, sizeof(unsigned int)*idx_mem_needed_L, hipMemcpyHostToDevice, hip_stream));
//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );

     
//      PUSH_RANGE("transfer indeces",4);
      CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
         plan_dev, plan_stg, sizeof(int)*(plan->size()), hipMemcpyHostToDevice, hip_stream));
//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );


//      CUDA_GPU_ERR_CHECK( hipMemcpyAsync( 
//          OF_dev,  OF[L].data(), sizeof(unsigned int)*(Nprm), hipMemcpyHostToDevice, hip_stream ));  
//      CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
//         PMX_dev, PMX[L].data(), sizeof(unsigned int)*(Nprm), hipMemcpyHostToDevice, hip_stream )); 
//      CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
//         FVH_dev, FVH[L].data(), sizeof(unsigned int)*(FVH[L].size()), hipMemcpyHostToDevice, hip_stream ));
//      CUDA_GPU_ERR_CHECK( hipMemcpyAsync(
//          KS_dev,  KS[L].data(), sizeof(unsigned int)*( KS[L].size()), hipMemcpyHostToDevice, hip_stream )); 

//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );

      // (nvidia?) GPUs adhere to IEEE-754, so a pattern of all 0s represents a floating-point zero.
      CUDA_GPU_ERR_CHECK( hipMemsetAsync( dat_mem_dev, 0, dat_mem_needed_L , hip_stream ) );

//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );     
//      POP_RANGE; // transfer indeces

//      PUSH_RANGE("compute",5);
      int prep_Fm_blocksize = 128;
      int prep_Fm_numblocks = (Nprm+prep_Fm_blocksize-1)/prep_Fm_blocksize;

      CUDA_GPU_ERR_CHECK( hipMemsetAsync( Fm_dev, 0, Fm_size[L] , hip_stream ) );
      prepare_Fm_batched_gpu_low_private<<<prep_Fm_numblocks,prep_Fm_blocksize,0,hip_stream>>>(
         FVH_dev, OF_dev, PMX_dev, data_dev, Fm_dev, Nprm, labcd,
         periodic, cell_h_dev, neighs_dev, max_ncells );

//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      std::vector<double> FM0_on_cpu(Fm_size[L]);
//      CUDA_GPU_ERR_CHECK( hipMemcpy( FM0_on_cpu.data(),  Fm_dev, sizeof(double)*(Fm_size[L]), hipMemcpyDeviceToHost) );
//      cout << " FM " << endl;
//      for( unsigned int ifm=0; ifm < Fm_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << FM0_on_cpu[ifm] << endl;
//      } cout << endl;

      int Fm_blocksize = 32;
      int Fm_numblocks = Nprm; // (Nprm+Fm_blocksize-1)/Fm_blocksize;

      compute_Fm_batched_gpu_low_private<<<Fm_numblocks,Fm_blocksize,0,hip_stream>>>(
         Fm_dev, Nprm, labcd, periodic, neighs_dev, 
         ftable_dev, ftable_ld,R_cut,C0_dev,ld_C0,
         x12_to_patch_low_R_dev, x12_to_patch_high_R_dev, BW_by_patch_dev,
         potential_type, max_ncells );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );

      int Vm_blocksize = 64;
      int Vm_numblocks = Nprm;
      compute_Vm_batched_gpu_low_private<<<Vm_numblocks,Vm_blocksize,0,hip_stream>>>(
         Fm_dev, Nprm, labcd, periodic, neighs_dev, 
         ftable_dev, ftable_ld,R_cut,C0_dev,ld_C0,
         x12_to_patch_low_R_dev, x12_to_patch_high_R_dev, BW_by_patch_dev,
         potential_type, max_ncells );

//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      std::vector<double> FM_on_cpu(Fm_size[L]);
//      CUDA_GPU_ERR_CHECK( hipMemcpy( FM_on_cpu.data(),  Fm_dev, sizeof(double)*(Fm_size[L]), hipMemcpyDeviceToHost) );
//      cout << " FM " << endl;
//      for( unsigned int ifm=0; ifm < Fm_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << FM_on_cpu[ifm] << endl;
//      } cout << endl;

//      std::vector<unsigned int> FVH_on_cpu(FVH[L].size());
//      CUDA_GPU_ERR_CHECK( hipMemcpy( FVH_on_cpu.data(),  FVH_dev, sizeof(unsigned int )*(FVH[L].size()), hipMemcpyDeviceToHost) );
//      cout << " FVH <- " << la << " " << lb << " " << lc << " " << ld << endl;
//      for( int ifm=0; ifm < FVH[L].size(); ifm++ ){
//         cout << " " << FVH_on_cpu[ifm] ;
//         if (ifm % FVH_SIZE == FVH_SIZE-1 ){ cout << endl ; }
//      } cout << endl;
//
//      // (nvidia?) GPUs adhere to IEEE-754, so a pattern of all 0s represents a floating-point zero.
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      cout << " Dev is " << dat_mem_dev << " of size " << max_dat_mem_per_thread/1024/1024 << " AC is " << AC_dev << " of size " << AC_size[L]/1024/1024 << " L: " << L << endl;

      CUDA_GPU_ERR_CHECK( hipMemsetAsync( AC_dev, 0, AC_size[L] , hip_stream ) );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

      int vrr_index = 64*la+16*lb+4*lc+ld;
//      compute_VRR_v2_batched_gpu_low<<<Ncells*max_ncells,64,0,hip_stream>>>(
//        Ncells, vrr_index, PMX_dev, FVH_dev, Fm_dev, data_dev,
//        AC_dev, nullptr, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 

      compute_VRR_v3(
        Ncells, vrr_index, PMX_dev, FVH_dev, Fm_dev, data_dev,
        AC_dev, nullptr, vrr_blocksize, hrr_blocksize, numV, numVC, max_ncells, hip_stream ); 


//         compute_VRR_batched_gpu_low<<<Ncells*max_ncells,64,0,hip_stream>>>(
//            Ncells, plan_dev, PMX_dev, FVH_dev, Fm_dev, data_dev,
//            AC_dev, nullptr, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 


//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      std::vector<double> AC0_on_cpu(AC_size[L]);
//      CUDA_GPU_ERR_CHECK( hipMemcpy( AC0_on_cpu.data(),  AC_dev, AC_size[L], hipMemcpyDeviceToHost) );
//      cout << " AC " << la << " " << lb << " " << lc << " " << ld << " " << AC_size[L] << endl;
//      for( unsigned int ifm=0; ifm < AC_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << AC0_on_cpu[ifm] << endl;
//      } cout << endl;    

      CUDA_GPU_ERR_CHECK( hipMemsetAsync( ABCD_dev, 0, ABCD_size[L] , hip_stream ) );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

      compute_SFT_batched_gpu_low<<<Ncells*Nop,128,0,hip_stream>>>(
         Ncells, plan_dev, PMX_dev, FVH_dev, nullptr, data_dev,
         AC_dev, ABCD_dev, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

      compute_ECO_batched_gpu_low<<<Ncells*Nop,128,0,hip_stream>>>(
         Ncells, plan_dev, PMX_dev, FVH_dev, nullptr, data_dev,
         AC_dev, ABCD_dev, vrr_blocksize, hrr_blocksize, labcd, numV, numVC, max_ncells ); 
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      std::vector<double> AC_on_cpu(AC_size[L]);
//      CUDA_GPU_ERR_CHECK( hipMemcpy( AC_on_cpu.data(),  AC_dev, sizeof(double)*(AC_size[L]), hipMemcpyDeviceToHost) );
//      cout << " AC " << AC_size[L] << endl;
//      for( unsigned int ifm=0; ifm < AC_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << AC_on_cpu[ifm] << endl;
//      } cout << endl;

//      std::vector<double> ABCD_on_cpu(ABCD_size[L]);
//      CUDA_GPU_ERR_CHECK( hipMemcpy( ABCD_on_cpu.data(),  ABCD_dev, sizeof(double)*(ABCD_size[L]), hipMemcpyDeviceToHost) );
//      cout << " ABCD " << ABCD_size[L] << endl;
//      for( unsigned int ifm=0; ifm < ABCD_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << ABCD_on_cpu[ifm] << endl;
//      } cout << endl;

//      std::vector<unsigned int> FVH2_on_cpu(FVH[L].size());
//      CUDA_GPU_ERR_CHECK( hipMemcpy( FVH2_on_cpu.data(),  FVH_dev, sizeof(unsigned int )*(FVH[L].size()), hipMemcpyDeviceToHost) );
//      cout << " FVH 2 <- " << endl;
//      for( int ifm=0; ifm < FVH[L].size(); ifm++ ){
//         cout << " " << FVH2_on_cpu[ifm] ;
//         if (ifm % FVH_SIZE == FVH_SIZE-1 ){ cout << endl ; }
//      } cout << endl;

//      cout << "FVH" << endl;
//      for ( int ii = 0 ; ii < FVH[L].size() ; ii++ ){
//         cout << FVH[L][ii] << " " ;
//         if ( ii % FVH_SIZE == FVH_SIZE-1 ){ cout << endl ; }
//      } cout << endl;

      CUDA_GPU_ERR_CHECK( hipMemsetAsync( ABCD0_dev, 0, ABCD0_size[L] , hip_stream ) );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

      compute_HRR_batched_gpu_low<<<Ncells,128,0,hip_stream>>>(
         Ncells, plan_dev, FVH_dev, data_dev, ABCD_dev, ABCD0_dev,
         periodic, cell_h_dev, neighs_dev,
         hrr_blocksize, Nc, numVC, numVCH );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      std::vector<double> ABCD0_on_cpu(ABCD0_size[L]);
//      CUDA_GPU_ERR_CHECK( hipMemcpy( ABCD0_on_cpu.data(),  ABCD0_dev, sizeof(double)*(ABCD0_size[L]), hipMemcpyDeviceToHost) );
//      cout << " ABCD0 " << endl;
//      for( int ifm=0; ifm < ABCD0_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << ABCD0_on_cpu[ifm] << endl;
//      } cout << endl;


      // Note: we need to DeviceSynchronize before going from kernels to cublas. TODO actually check it is true
      // TODO it should not be necessary since this cublas handle has been assigned to this stream
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
      // note: uses ABCD as a scratch space

//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
      // (nvidia?) GPUs adhere to IEEE-754, so a pattern of all 0s represents a floating-point zero.
//      CUDA_GPU_ERR_CHECK( hipMemsetAsync( SPHER_dev, 0, SPHER_size[L]*sizeof(double) , hip_stream ) );

      compute_SPH_batched_gpu_alt ( Nqrtt, la, lb, lc, ld, ABCD0_dev, SPHER_dev, SPTMP_dev, C2S_dev, cublas_handle );

//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      std::vector<double> SPHER_on_cpu(SPHER_size[L]);
//      CUDA_GPU_ERR_CHECK( hipMemcpy( SPHER_on_cpu.data(),  SPHER_dev, sizeof(double)*(SPHER_size[L]), hipMemcpyDeviceToHost) );
//      cout << " SPHER " << endl;
//      for( int ifm=0; ifm < SPHER_size[L]; ifm++ ){
//         cout << ifm << " " << std::setprecision(16) << SPHER_on_cpu[ifm] << endl;
//      } cout << endl;

//      #pragma omp critical
//      {
//         cout << " Preparing to compute KS " << Nqrtt << " " << KS_dev << " [" << KS[L].size() << "] "
//              << P_a_dev << " [" << FP_size      << " ]" << SPHER_dev << " [" << SPHER_size[L] << "]"
//              << K_a_dev << " " << hf_fac << " on stream " << hip_stream << endl;
//
//         for (int q=0; q < KS[L].size() ; q++ ){
//            cout << KS[L][q] << "  ";
//            if ( q%KS_SIZE==KS_SIZE-1) { cout << endl; }
//         }
//      }
      compute_KS_gpu<<<Nqrtt,128,0,hip_stream>>>( Nqrtt, KS_dev, la,lb,lc,ld, P_a_dev, SPHER_dev, K_a_dev, data_dev, hf_fac );
      if ( nspin == 2 ){
         compute_KS_gpu<<<Nqrtt,128,0,hip_stream>>>( Nqrtt, KS_dev, la,lb,lc,ld, P_b_dev, SPHER_dev, K_b_dev, data_dev, hf_fac );
      }

//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );

//      cout << endl;

//      CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//      CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//      CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );

//      POP_RANGE; // compute
//      POP_RANGE; // Lname
      reset_indices(L);
   }
//   POP_RANGE; // compute all L

// #pragma omp barrier

//   std::vector<double> OUT_from_gpu( OUT.size() );
//   CUDA_GPU_ERR_CHECK( hipMemcpy( OUT_from_gpu.data(), OUT_dev, sizeof(double)*(OUT.size()), hipMemcpyDeviceToHost ));

//   std::vector<double> F_a_from_gpu( FP_size );
//   cout << endl;
//   cout << " Copying K back from gpu to cpu. " << endl;
//   cout << " Gpu address is " << K_a_dev << endl;
//   cout << " Cpu address is " << F_a_from_gpu.data() << endl;
//   cout << " Copying " << FP_size << " elements " << endl;
//   cout << endl;
//   CUDA_GPU_ERR_CHECK( hipMemcpy( F_a_from_gpu.data(), K_a_dev, sizeof(double)*(FP_size), hipMemcpyDeviceToHost ));
//   if ( nspin == 2 ){
//      std::vector<double> F_b_from_gpu( FP_size );
//      CUDA_GPU_ERR_CHECK( hipMemcpy( F_b_from_gpu.data(), K_b_dev, sizeof(double)*(FP_size), hipMemcpyDeviceToHost ));
//   }
//   for ( int ipf=0; ipf < FP_size; ipf++ ){ cout <<  ipf << " " << F_a_from_gpu[ipf] << endl ; } cout << endl;

   // Wait for all kernels to finish before returning control to caller
//   CUDA_GPU_ERR_CHECK( hipStreamSynchronize(hip_stream) );
//   CUDA_GPU_ERR_CHECK( hipDeviceSynchronize() );
//   CUDA_GPU_ERR_CHECK( hipPeekAtLastError() );


//   dis_timer.stop();
//   dis_ms += dis_timer.elapsedMilliseconds(); 
//   dis_cnt++;
}


template < class T > 
T sum( std::vector< T > x ){
   T ret = (T)0;
   for( auto i : x ){
      ret += i;
   }
   return ret;
}

void libGint::report_througput(bool skip_cpu){
#pragma omp single
   cout << " report_througput not implemented  ";
}

#include "hip/hip_runtime.h"
#include "hipblas/hipblas.h"
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

#include <array>
#include "plan.h"
#include "define.h"

bool ArrayOpLess( const OpArray& a, const std::array<int,OP_SIZE>& b ){
   return a[H__OFFSET] < b[H__OFFSET];
}

inline bool ArrayOpGreater( const OpArray& a, const std::array<int,OP_SIZE>& b ){
   if (a[H__OFFSET] > b[H__OFFSET]) return true;
   if (a[H__OFFSET] < b[H__OFFSET]) return false;
   if (a[M__OFFSET] < b[M__OFFSET]) return true;
   return false;
}

//inline int NLco( const int L ){ return ((L+1)*(L+2))/2; }


int mem( const int la, const int lb, const int lc, const int ld){
   return NLco(la)*NLco(lb)*NLco(lc)*NLco(ld);
}
int mem( const Op& sh ){
   return NLco(sh.la)*NLco(sh.lb)*NLco(sh.lc)*NLco(sh.ld);
}

bool contains(const OpDict* G, Op op ) { return G->count(op) > 0; }
bool contains(const OpDict& G, Op op ) { return G.count(op) > 0; }


bool contains(const std::unordered_map< int, Plan_t >& PC , const int la, const int lb, const int lc, const int ld ){
  return PC.count(SERIAL_L4(la,lb,lc,ld)) > 0;
}

void rec_add( int h, int la, int lb, int lc, int ld, int m, OpDict *G ){

   if ( la+lb+lc+ld == 0 ) return;

   Op tmp; tmp.la=la; tmp.lb=lb; tmp.lc=lc; tmp.ld=ld; tmp.m=m; 

   // Add Second Horizontal Recursive Relation (a,b|c,d) = (a+1,b-1,c,d) + AB (a,b-1,c,d)
   if ( lb > 0 ){
      tmp.t = HRR2;
      (*G)[tmp] = h;
      rec_add( h+1, la+1,lb-1,lc  ,ld  ,0  ,G);
      rec_add( h+1, la  ,lb-1,lc  ,ld  ,0  ,G);
      return;
   }
   // Add First Horizontal Recursive Relation (a,b|c,d) = (a,b,c+1,d-1) + CD (a,b,c,d-1)
   if ( ld > 0 ){
      tmp.t = HRR1;
      (*G)[tmp] = h;
      rec_add( h+1, la  ,lb  ,lc+1,ld-1,0  ,G);
      rec_add( h+1, la  ,lb  ,lc  ,ld-1,0  ,G);
      return;
   }
   // find which VRR should be used
   if (lc == 0){
      // VRR of the first and second kind:
      // (a,0,m) = PA # X_(a,a-1) # (a-1,0,m) + WP # X_(a,a-1) (a-1,0,m)
      //         + e2 * ( 1/2zab * X_(a,c;a-2,c) # (a-2,c,m) + (-r/zab2) * X_(a,c,a-2,c) # (a-2,c,m+1) )
      tmp.t = VRR1;
      if (la > 1) tmp.t = VRR2;
   } else {
      // VRR of the first kind but with c and a swapped
      tmp.t = VRR3;
      if (lc > 1){
         tmp.t = VRR4;
         // General VRR
         // (a,c,m) = QC # X_(a,c;a,c-1) # (a,c-1,m) + WQ # X_(a,c;a,c-1) # (a,c-1,m+1) 
         //         + f2  *( 1/2zcd * X_(a,c;a,c-2) # (a,c-2,m) + (-r/zcd2) #  X_(a,c;a,c-2) # (a,c-2,m+1) )
         //         + e2 * 1/2z * X_(a,c,a-1,c-1) # (a-1,c-1,m+1)
         if (la > 0) tmp.t = VRR5;
      } else {
         if (la > 0) tmp.t = VRR6;
      }
   }
   // update the urgency of the op
   if ( not contains(G,tmp) ){
      (*G)[tmp] = h;
   } else {
      (*G)[tmp] = std::max( (*G)[tmp], h );
      return;
   }
   int t = tmp.t;
   // Apply the correct VRR
   if (t == VRR1){
      rec_add( h+1, la-1,lb,lc-0,ld, m+0, G );
      rec_add( h+1, la-1,lb,lc-0,ld, m+1, G );
      return;
   }
   if (t == VRR2){
      rec_add( h+1, la-1,lb,lc-0,ld, m+0, G );
      rec_add( h+1, la-1,lb,lc-0,ld, m+1, G );
      rec_add( h+1, la-2,lb,lc-0,ld, m+0, G );
      rec_add( h+1, la-2,lb,lc-0,ld, m+1, G );
      return;
   }
   if (t == VRR3){
      rec_add( h+1, la-0,lb,lc-1,ld, m+0, G );
      rec_add( h+1, la-0,lb,lc-1,ld, m+1, G );
       return;
   }
  if (t == VRR4){
      rec_add( h+1, la-0,lb,lc-1,ld, m+0, G );
      rec_add( h+1, la-0,lb,lc-1,ld, m+1, G );
      rec_add( h+1, la-0,lb,lc-2,ld, m+0, G );
      rec_add( h+1, la-0,lb,lc-2,ld, m+1, G );
      return;
   }
   if (t == VRR5){
      rec_add( h+1, la-0,lb,lc-1,ld, m+0, G );
      rec_add( h+1, la-0,lb,lc-1,ld, m+1, G );
      rec_add( h+1, la-0,lb,lc-2,ld, m+0, G );
      rec_add( h+1, la-0,lb,lc-2,ld, m+1, G );
      rec_add( h+1, la-1,lb,lc-1,ld, m+1, G );
      return;
   }
   if (t == VRR6){
      rec_add( h+1, la-0,lb,lc-1,ld, m+0, G );
      rec_add( h+1, la-0,lb,lc-1,ld, m+1, G );
      rec_add( h+1, la-1,lb,lc-1,ld, m+1, G );
      return;
   }
}


bool is_vrr(const int t){
   return t==VRR1 or t==VRR2 or t==VRR3 or t==VRR4 or t==VRR5 or t==VRR6;
}



bool CompareOp( const OpArray& op1, const OpArray& op2 ){
   // returns true if op1 goes before op2, false otherwise.
   int t1 = op1[T__OFFSET];
   int a1 = op1[LA_OFFSET]; 
   int b1 = op1[LB_OFFSET]; 
   int c1 = op1[LC_OFFSET]; 
   int d1 = op1[LD_OFFSET];
//   int m1 = op1[M__OFFSET];
   int h1 = op1[H__OFFSET];

   int t2 = op2[T__OFFSET];
   int a2 = op2[LA_OFFSET]; 
   int b2 = op2[LB_OFFSET]; 
   int c2 = op2[LC_OFFSET]; 
   int d2 = op2[LD_OFFSET];
//   int m2 = op2[M__OFFSET];
   int h2 = op2[H__OFFSET];

   int l1 = a1+b1+c1+d1;
   int l2 = a2+b2+c2+d2;

   const bool op1_first = true  ;
   const bool op2_first = false ;

   // vrr1 first, then vrr2 and vrr3. Any other op is always after them
   if (t1 == VRR1 and t2 != VRR1 ){ return op1_first; }
   if (t1 != VRR1 and t2 == VRR1 ){ return op2_first; }
   if (t1 == VRR1 and t2 == VRR1 ){ return  l1 < l2 ; }
   // no more vrr1. 
   if (t1 == VRR2 and t2 != VRR2 ){ return op1_first; }
   if (t1 != VRR2 and t2 == VRR2 ){ return op2_first; }
   if (t1 == VRR2 and t2 == VRR2 ){ return  l1 < l2 ; }

   if (t1 == VRR3 and t2 != VRR3 ){ return op1_first; }
   if (t1 != VRR3 and t2 == VRR3 ){ return op2_first; }
   if (t1 == VRR3 and t2 == VRR3 ){ return  l1 < l2 ; }

   bool is1_vrr = is_vrr(t1);
   bool is2_vrr = is_vrr(t2);

   // prefer vrrs over hrrs
   if ( is1_vrr and not is2_vrr) { return op1_first; }
   if ( is2_vrr and not is1_vrr) { return op2_first; }

   // if they are both hrr, return order by h. Higher h first
   if ( ( not is1_vrr ) and ( not is2_vrr ) ){ return h1 > h2; }

   // now we know we only have vrr4,vrr5,vrr6
   // order by moment, lower moment first
   return l1 < l2;
}

bool contains ( const std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, int> > > >& shared_addr, int la, int lb, int lc, int ld ){
   return shared_addr.count(la) == 1 and shared_addr.at(la).count(lb) == 1 and shared_addr.at(la).at(lb).count(lc) == 1 and shared_addr.at(la).at(lb).at(lc).count(ld) == 1;
}


bool contains ( const std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, int> > >& priv_addr, int la, int lc, int m ){
   return priv_addr.count(la) == 1 and priv_addr.at(la).count(lc) == 1 and priv_addr.at(la).at(lc).count(m) == 1;
}


std::vector<int> plan( const int LA, const int LB, const int LC, const int LD, int* pr_size, int* sh_size, int* numV, int* numVC, int* numVCH ){
   int L = LA + LB + LC + LD;

//   cout << "Planning " << LA << " " << LB << " " << LC << " " << LD << endl; cout.flush();

   // given the target angular moments la,lb,lc,ld prepare a plan of execution
   // the plan at the end will be a vector of ( array of int of size OP_SIZE )
   // each element describe how to compute one of the intermidiate product (or the final element)
   // each element of the plan contains 
   // 0    -> type of operation. Can be HRR, VRR or (Contract from Private To Shared memory) CP2D
   // 1-4  -> l of the target
   // 5    -> m of the target. Or maximum m of the target
   // 6    -> (order of priority) h. Or minimum m of the target
   // 7-12 -> memory offsets.
   // For example, if op = [6, 2 0 1 0 2, 7, 405 45 51 18 0 0]
   // We need to perform a VRR of type 6, with the target AL=2, CL=1, m=2 (BL and DL are always 0 for a vrr6 op)
   // this means dsps(2) = QC dsss(2) + WQ dsss(3) + 1/2z psss(3)
   // the output is at pr_mem+405, the inputs are at pr_mem+45,51 and 18

   // We use 3 main memory structure for the plan:
   // unordered map so that we can easily add and check if on op already exist 
   OpDict G = OpDict();
   // a list of OpDict so that we can easily insert new ops in any place 
   std::list<OpArray> G_list;
   std::list<OpArray> G_list2;
   // a vector of int (declared later, once we knwow its size)
   // std::vector<int> G_vec;
   //
   // We also use temporary object to store operations and memory addresses
   // temporary holders for an operation. Also used for memory addresses lookup
   Op tmp;
   OpArray tmp_op;
   // max and min m associated to a given la,lc
   std::unordered_map<int, std::unordered_map<int, int> > vrr_min_m;
   std::unordered_map<int, std::unordered_map<int, int> > vrr_max_m;
   // global/register/per thread memory
   std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, int> > > priv_addr;
   // Addresses in shared memory for the HRR and output of CP2S
   std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, int> > > > shared_addr;

   int curr_pr_mem_loc = 0;
   int curr_sh_mem_loc = 0;

   // always insert the final results in the zero address
   shared_addr[LA][LB][LC][LD] = 0;
   curr_sh_mem_loc += mem(LA,LB,LC,LD);

   // make it so that the address for ssssm values starts at 0 and are consecutive
   for( int m=0; m < L+1; m++ ){
      priv_addr[0][0][m] = curr_pr_mem_loc;
      curr_pr_mem_loc += 1;
   }

   //
   // create the tree of operations
   //
   rec_add( 0, LA, LB, LC, LD, 0, &G );
   //
   //

   // change from unordered map to list
   for ( const auto& Op_h : G ) {
      tmp = Op_h.first;
      int h = Op_h.second;
      tmp_op[T__OFFSET] = tmp.t;
      tmp_op[LA_OFFSET] = tmp.la;
      tmp_op[LB_OFFSET] = tmp.lb;
      tmp_op[LC_OFFSET] = tmp.lc;
      tmp_op[LD_OFFSET] = tmp.ld;
      tmp_op[M__OFFSET] = tmp.m;
      tmp_op[H__OFFSET] = h;
      G_list.push_back(tmp_op);
   }

   // sort the order of operations by the value of h
   G_list.sort( CompareOp );

//   printf(" L %d%d%d%d: la %d lc %d min %d max %d \n", LA,LB,LC,LD, 0, 0, vrr_min_m[0][0], vrr_max_m[0][0] );
   vrr_min_m[0][0] = 0;
   vrr_max_m[0][0] = L;
//   printf(" L %d%d%d%d: la %d lc %d min %d max %d \n", LA,LB,LC,LD, 0, 0, vrr_min_m[0][0], vrr_max_m[0][0] );

   // find the max and min m for all a0c0 operations
   int n_ops = int( G_list.size() );
   for ( std::list<OpArray>::const_iterator it = G_list.begin(); it != G_list.end(); it++ ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
      int lc = tmp_op[LC_OFFSET];
      int m  = tmp_op[M__OFFSET];

      if ( is_vrr(t) ){
          if ( vrr_min_m.count(la) == 1 and vrr_min_m[la].count(lc) == 1 ){
             vrr_min_m[la][lc] = std::min(vrr_min_m[la][lc], m);
             vrr_max_m[la][lc] = std::max(vrr_max_m[la][lc], m);
          } else {
             vrr_min_m[la][lc] = m;
             vrr_max_m[la][lc] = m;
          }
//          printf(" L %d%d%d%d: la %d lc %d min %d max %d \n", LA,LB,LC,LD, la, lc, vrr_min_m[la][lc], vrr_max_m[la][lc] );
      }
   }

   // compress G by collapsing vrr operations with the same la and lc, but different m
   n_ops = int( G_list.size() );
   for ( std::list<OpArray>::const_iterator it = G_list.begin(); it != G_list.end(); it++ ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
      int lc = tmp_op[LC_OFFSET];
      int m  = tmp_op[M__OFFSET];
      if ( is_vrr(t) ){
         if ( m == vrr_max_m[la][lc] ){
            tmp_op[H__OFFSET] = vrr_min_m[la][lc]; // note, we reuse h to store minimum m
            G_list2.push_back( tmp_op );
         } // else pass and DO NOT ADD THE OP
      } else {
         G_list2.push_back( tmp_op );
      }
   }

   G_list = G_list2 ;

   // assign memory locations to the operations
   // In the case of the HRRs, we add both input and output to the 
   // memory locations in shared memory. In the case of the VRR
   // this is not necessary, we only need to add the output to the memory
   n_ops = int( G_list.size() );
   for ( std::list<OpArray>::const_iterator it = G_list.begin(); it != G_list.end(); it++ ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
      int lb = tmp_op[LB_OFFSET];
      int lc = tmp_op[LC_OFFSET];
      int ld = tmp_op[LD_OFFSET];
      int m  = tmp_op[M__OFFSET];
      int h  = tmp_op[H__OFFSET];
      if (t == HRR1){
         if ( !contains(shared_addr, la,lb,lc  ,ld  ) ){ shared_addr[la][lb][lc  ][ld  ] = curr_sh_mem_loc; curr_sh_mem_loc += mem(la,lb,lc  ,ld  ); }
         if ( !contains(shared_addr, la,lb,lc+1,ld-1) ){ shared_addr[la][lb][lc+1][ld-1] = curr_sh_mem_loc; curr_sh_mem_loc += mem(la,lb,lc+1,ld-1); }
         if ( !contains(shared_addr, la,lb,lc  ,ld-1) ){ shared_addr[la][lb][lc  ][ld-1] = curr_sh_mem_loc; curr_sh_mem_loc += mem(la,lb,lc  ,ld-1); }
      } else if (t == HRR2) {
         if ( !contains(shared_addr, la  ,lb  ,lc,ld) ){ shared_addr[la  ][lb  ][lc][ld] = curr_sh_mem_loc; curr_sh_mem_loc += mem(la  ,lb  ,lc,ld); }
         if ( !contains(shared_addr, la+1,lb-1,lc,ld) ){ shared_addr[la+1][lb-1][lc][ld] = curr_sh_mem_loc; curr_sh_mem_loc += mem(la+1,lb-1,lc,ld); }
         if ( !contains(shared_addr, la  ,lb-1,lc,ld) ){ shared_addr[la  ][lb-1][lc][ld] = curr_sh_mem_loc; curr_sh_mem_loc += mem(la  ,lb-1,lc,ld); }
      } else { // if is vrr
         int max_m = m; // added for clarity
         int min_m = h; // note, we reused h to store minimum m
         if ( !contains(priv_addr, la,lc,0) ){
            for ( int mm=min_m; mm <= max_m; mm++){ // long proof, but the m are always continuous from min_m to max_m
               priv_addr[la][lc][mm] = curr_pr_mem_loc;
               curr_pr_mem_loc += mem(la,0,lc,0); // when done this way, pr data with the same a c but different m are countiguous in memory
            }
         }
      }
   }
   // save the amount of memory needed in the /output/ variables pr_size and sh_size
   (*pr_size) = curr_pr_mem_loc;
   (*sh_size) = curr_sh_mem_loc;

   // if we only require the s integrals, the op list is empty.
   // We need to manually add the sum of the K contributions to the final integral
   tmp_op[T__OFFSET] = CP2S;
   tmp_op[LA_OFFSET] = 0;
   tmp_op[LB_OFFSET] = 0;
   tmp_op[LC_OFFSET] = 0;
   tmp_op[LD_OFFSET] = 0;
   tmp_op[M__OFFSET] = 0;
   tmp_op[H__OFFSET] = 0;
   if (n_ops == 0){
      G_list.insert( G_list.begin(), tmp_op );
   } else {
      // in the next step, we are going to add reductions from the VRR.
      // Since 0000 is never computed from a VRR, although it might be needed,
      // we check if it is needed (aka is on shared addr) and manually add the reduction
      if ( contains(shared_addr, 0, 0, 0, 0) ){
         G_list.insert( G_list.begin()++, tmp_op );
      }
   }

   // Add a CP2S operation after an element if ((is computed using a VRR) and (it is needed for a HRR))
   OpArray tmp_op2;
   for ( auto it = G_list.begin() ; it != G_list.end(); it++ ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
      int lb = tmp_op[LB_OFFSET];
      int lc = tmp_op[LC_OFFSET];
      int ld = tmp_op[LD_OFFSET];
//      int max_m = tmp_op[M__OFFSET];
      int min_m = tmp_op[H__OFFSET];
      bool is_op_vrr = is_vrr(t); 
      bool move_pr_to_sh = ( (min_m == 0) and (contains(shared_addr,la,lb,lc,ld)) );
      if ( is_op_vrr and move_pr_to_sh ){
         tmp_op2 = tmp_op;
         tmp_op2[T__OFFSET] = CP2S;
         G_list.insert(++it, tmp_op2 );// TODO ++it, followed by it-- ?
         it--;
      }
   }

   // add a sync team operation before a VRR operation if its total moment is different from the one of the vrr preceding it
   int curr_l = 1;
   for ( std::list<OpArray>::iterator it = G_list.begin() ; it != G_list.end(); ++it ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
      int lb = tmp_op[LB_OFFSET];
      int lc = tmp_op[LC_OFFSET];
      int ld = tmp_op[LD_OFFSET];
//      int m  = tmp_op[M__OFFSET];
//      int h  = tmp_op[H__OFFSET];
      int l  = la + lb + lc + ld ;
      if ( is_vrr(t) and l != curr_l ){
         tmp_op2 = tmp_op;
         tmp_op2[T__OFFSET] = SYTM;
         G_list.insert(it, tmp_op2 );
         curr_l = l;
      }
   }

   // add a sync block operation before a HRR operation if its priority is different from the one of the hrr preceding it
   // note: this is because (1) the hrr are sorted by already priority in G_list (2) hrr with the same priority can always be done in whatever order with no sync
   int curr_h = -1; // special value to mark first h
   for ( std::list<OpArray>::iterator it = G_list.begin() ; it != G_list.end(); ++it ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
//      int la = tmp_op[LA_OFFSET];
//      int lb = tmp_op[LB_OFFSET];
//      int lc = tmp_op[LC_OFFSET];
//      int ld = tmp_op[LD_OFFSET];
//      int m  = tmp_op[M__OFFSET];
      int h  = tmp_op[H__OFFSET];
      if ( t == HRR1 or t == HRR2 ){
         if ( curr_h == -1 ) { curr_h = h; }
         if ( h != curr_h ){
            tmp_op2 = tmp_op;
            tmp_op2[T__OFFSET] = SYBL;
            G_list.insert(it, tmp_op2 );
            curr_h = h;
         }
      }
   }

   // Convert the list to the final output vector
   // Reads the t,a,b,c,d,m,h from the list element.
   // Then does a lookup of the memory addresses for the given operation
   // and add them to the data in the vector
   n_ops = int( G_list.size() );
   std::vector<int> G_vec((n_ops+2)*OP_SIZE);
   int i = 0;
   // part1 : ALL VRRs and sync team
   for ( std::list<OpArray>::iterator it = G_list.begin() ; it != G_list.end(); ++it ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
      int lb = tmp_op[LB_OFFSET];
      int lc = tmp_op[LC_OFFSET];
      int ld = tmp_op[LD_OFFSET];
      int M  = tmp_op[M__OFFSET]; // max_m
      int m  = tmp_op[H__OFFSET]; // min_m
      if ( is_vrr(t) or t == SYTM ){
         G_vec[i*OP_SIZE + T__OFFSET] = t  ; // tmp_op[T__OFFSET];
         G_vec[i*OP_SIZE + LA_OFFSET] = la ; // tmp_op[LA_OFFSET];
         G_vec[i*OP_SIZE + LB_OFFSET] = lb ; // tmp_op[LB_OFFSET];
         G_vec[i*OP_SIZE + LC_OFFSET] = lc ; // tmp_op[LC_OFFSET];
         G_vec[i*OP_SIZE + LD_OFFSET] = ld ; // tmp_op[LD_OFFSET];
         G_vec[i*OP_SIZE + M__OFFSET] = M  ; // tmp_op[M__OFFSET];
         G_vec[i*OP_SIZE + H__OFFSET] = m  ; // tmp_op[H__OFFSET];
         // sorry for the block of code. For each op type, find the memory addresses needed
         // and add them in the G vector in the correct positions
         // Note that some memory in G will remain unused. We do so to keep the size of op uniform
         if (t == VRR1){
            G_vec[i*OP_SIZE + M1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M2_OFFSET] = priv_addr.at(la-1).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M3_OFFSET] = priv_addr.at(la-1).at(lc  ).at(m+1);

            G_vec[i*OP_SIZE + L1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(vrr_max_m[la  ][lc  ]) + mem(la  ,0,lc  ,0);
            G_vec[i*OP_SIZE + L2_OFFSET] = priv_addr.at(la-1).at(lc  ).at(vrr_max_m[la-1][lc  ]) + mem(la-1,0,lc  ,0);
            G_vec[i*OP_SIZE + L3_OFFSET] = priv_addr.at(la-1).at(lc  ).at(vrr_max_m[la-1][lc  ]) + mem(la-1,0,lc  ,0);

         } else if ( t == VRR2 ){
            G_vec[i*OP_SIZE + M1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M2_OFFSET] = priv_addr.at(la-1).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M3_OFFSET] = priv_addr.at(la-1).at(lc  ).at(m+1);
            G_vec[i*OP_SIZE + M4_OFFSET] = priv_addr.at(la-2).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M5_OFFSET] = priv_addr.at(la-2).at(lc  ).at(m+1);

            G_vec[i*OP_SIZE + L1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(vrr_max_m[la  ][lc  ]) + mem(la  ,0,lc  ,0);
            G_vec[i*OP_SIZE + L2_OFFSET] = priv_addr.at(la-1).at(lc  ).at(vrr_max_m[la-1][lc  ]) + mem(la-1,0,lc  ,0);
            G_vec[i*OP_SIZE + L3_OFFSET] = priv_addr.at(la-1).at(lc  ).at(vrr_max_m[la-1][lc  ]) + mem(la-1,0,lc  ,0);
            G_vec[i*OP_SIZE + L4_OFFSET] = priv_addr.at(la-2).at(lc  ).at(vrr_max_m[la-2][lc  ]) + mem(la-2,0,lc  ,0);
            G_vec[i*OP_SIZE + L5_OFFSET] = priv_addr.at(la-2).at(lc  ).at(vrr_max_m[la-2][lc  ]) + mem(la-2,0,lc  ,0);

         } else if ( t == VRR3 ){
            G_vec[i*OP_SIZE + M1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m  );
            G_vec[i*OP_SIZE + M3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m+1);

            G_vec[i*OP_SIZE + L1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(vrr_max_m[la  ][lc  ]) + mem(la  ,0,lc  ,0); 
            G_vec[i*OP_SIZE + L2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0);
            G_vec[i*OP_SIZE + L3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0);

         } else if ( t == VRR4 ){
            G_vec[i*OP_SIZE + M1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m  );
            G_vec[i*OP_SIZE + M3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m+1);
            G_vec[i*OP_SIZE + M4_OFFSET] = priv_addr.at(la  ).at(lc-2).at(m  );
            G_vec[i*OP_SIZE + M5_OFFSET] = priv_addr.at(la  ).at(lc-2).at(m+1);

            G_vec[i*OP_SIZE + L1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(vrr_max_m[la  ][lc  ]) + mem(la  ,0,lc  ,0);
            G_vec[i*OP_SIZE + L2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0);
            G_vec[i*OP_SIZE + L3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0);
            G_vec[i*OP_SIZE + L4_OFFSET] = priv_addr.at(la  ).at(lc-2).at(vrr_max_m[la  ][lc-2]) + mem(la  ,0,lc-2,0);
            G_vec[i*OP_SIZE + L5_OFFSET] = priv_addr.at(la  ).at(lc-2).at(vrr_max_m[la  ][lc-2]) + mem(la  ,0,lc-2,0);

            // VRR5 requires 6 memory addresses. It (currently) sets the value of OP_SIZE
         } else if ( t == VRR5 ){         
            G_vec[i*OP_SIZE + M1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m  );
            G_vec[i*OP_SIZE + M3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m+1);
            G_vec[i*OP_SIZE + M4_OFFSET] = priv_addr.at(la  ).at(lc-2).at(m  );
            G_vec[i*OP_SIZE + M5_OFFSET] = priv_addr.at(la  ).at(lc-2).at(m+1);
            G_vec[i*OP_SIZE + M6_OFFSET] = priv_addr.at(la-1).at(lc-1).at(m+1);

            G_vec[i*OP_SIZE + L1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(vrr_max_m[la  ][lc  ]) + mem(la  ,0,lc  ,0); 
            G_vec[i*OP_SIZE + L2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0); 
            G_vec[i*OP_SIZE + L3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0);
            G_vec[i*OP_SIZE + L4_OFFSET] = priv_addr.at(la  ).at(lc-2).at(vrr_max_m[la  ][lc-2]) + mem(la  ,0,lc-2,0);
            G_vec[i*OP_SIZE + L5_OFFSET] = priv_addr.at(la  ).at(lc-2).at(vrr_max_m[la  ][lc-2]) + mem(la  ,0,lc-2,0);
            G_vec[i*OP_SIZE + L6_OFFSET] = priv_addr.at(la-1).at(lc-1).at(vrr_max_m[la-1][lc-1]) + mem(la-1,0,lc-1,0);

         } else if ( t == VRR6 ){
            G_vec[i*OP_SIZE + M1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(m  );
            G_vec[i*OP_SIZE + M2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m  );
            G_vec[i*OP_SIZE + M3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(m+1);
            G_vec[i*OP_SIZE + M4_OFFSET] = priv_addr.at(la-1).at(lc-1).at(m+1);

            G_vec[i*OP_SIZE + L1_OFFSET] = priv_addr.at(la  ).at(lc  ).at(vrr_max_m[la  ][lc  ]) + mem(la  ,0,lc  ,0); 
            G_vec[i*OP_SIZE + L2_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0);
            G_vec[i*OP_SIZE + L3_OFFSET] = priv_addr.at(la  ).at(lc-1).at(vrr_max_m[la  ][lc-1]) + mem(la  ,0,lc-1,0);
            G_vec[i*OP_SIZE + L4_OFFSET] = priv_addr.at(la-1).at(lc-1).at(vrr_max_m[la-1][lc-1]) + mem(la-1,0,lc-1,0);
         } else if ( t == SYTM ){
            // We don't need any other info
         }
      i++;
      }
   }
   // saves number of vvr and team sync to output variable numV
   (*numV) = i;

   // add a sync team?block? after the vrr, before the cp2s // almost sure is not needed
   G_vec[i*OP_SIZE + T__OFFSET] = SYBL; i++;


   // now all cp2s
   for ( std::list<OpArray>::iterator it = G_list.begin() ; it != G_list.end(); ++it ){
      tmp_op = (*it);
      int t = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
//      int lb = tmp_op[LB_OFFSET];
      int lc = tmp_op[LC_OFFSET];
//      int ld = tmp_op[LD_OFFSET];
//      int m  = tmp_op[M__OFFSET];
//      int h  = tmp_op[H__OFFSET];
      if ( t == CP2S ){
         G_vec[i*OP_SIZE + T__OFFSET] = tmp_op[T__OFFSET];
         G_vec[i*OP_SIZE + LA_OFFSET] = tmp_op[LA_OFFSET];
         G_vec[i*OP_SIZE + LB_OFFSET] = tmp_op[LB_OFFSET];
         G_vec[i*OP_SIZE + LC_OFFSET] = tmp_op[LC_OFFSET];
         G_vec[i*OP_SIZE + LD_OFFSET] = tmp_op[LD_OFFSET];
         G_vec[i*OP_SIZE + M__OFFSET] = tmp_op[M__OFFSET];
         G_vec[i*OP_SIZE + H__OFFSET] = tmp_op[H__OFFSET];
         G_vec[i*OP_SIZE + M1_OFFSET] = priv_addr.at(la).at(lc).at(0);
         G_vec[i*OP_SIZE + M2_OFFSET] = shared_addr.at(la).at(0).at(lc).at(0);
         i++;
      }
   }
   // saves number of vvr, team sync and cp2s to output variable numVC
   (*numVC) = i;

   // add a sync block after the cp2s, before the hrr
   G_vec[i*OP_SIZE + T__OFFSET] = SYBL; i++;

   for ( std::list<OpArray>::iterator it = G_list.begin() ; it != G_list.end(); ++it ){
      tmp_op = (*it);
      int t  = tmp_op[T__OFFSET];
      int la = tmp_op[LA_OFFSET];
      int lb = tmp_op[LB_OFFSET];
      int lc = tmp_op[LC_OFFSET];
      int ld = tmp_op[LD_OFFSET];
//      int m  = tmp_op[M__OFFSET];
//      int h  = tmp_op[H__OFFSET];
      if ( t == HRR1 or t == HRR2 or t == SYBL ){
         G_vec[i*OP_SIZE + T__OFFSET] = tmp_op[T__OFFSET];
         G_vec[i*OP_SIZE + LA_OFFSET] = tmp_op[LA_OFFSET];
         G_vec[i*OP_SIZE + LB_OFFSET] = tmp_op[LB_OFFSET];
         G_vec[i*OP_SIZE + LC_OFFSET] = tmp_op[LC_OFFSET];
         G_vec[i*OP_SIZE + LD_OFFSET] = tmp_op[LD_OFFSET];
         G_vec[i*OP_SIZE + M__OFFSET] = tmp_op[M__OFFSET];
         G_vec[i*OP_SIZE + H__OFFSET] = tmp_op[H__OFFSET];
         if ( t == HRR1 ){
            G_vec[i*OP_SIZE + M1_OFFSET] = shared_addr.at(la).at(lb).at(lc  ).at(ld  );
            G_vec[i*OP_SIZE + M2_OFFSET] = shared_addr.at(la).at(lb).at(lc+1).at(ld-1);
            G_vec[i*OP_SIZE + M3_OFFSET] = shared_addr.at(la).at(lb).at(lc  ).at(ld-1);
         } else if ( t == HRR2 ){
            G_vec[i*OP_SIZE + M1_OFFSET] = shared_addr.at(la  ).at(lb  ).at(lc).at(ld);
            G_vec[i*OP_SIZE + M2_OFFSET] = shared_addr.at(la+1).at(lb-1).at(lc).at(ld);
            G_vec[i*OP_SIZE + M3_OFFSET] = shared_addr.at(la  ).at(lb-1).at(lc).at(ld);
         }
         i++;
      }
   }
   (*numVCH) = i;

//   using std::cout;
//   using std::endl;
//   cout << " Plan for " << LA << LB << LC << LD << endl;
//   for ( i = 0; i < G_vec.size(); i ++ ){
//      cout << G_vec[i] << " ";
//      if ( i % OP_SIZE == OP_SIZE-1 ){ cout << endl; }
//   }
//   cout << endl ;

   return G_vec;
}
