#include <vector>
#include <unordered_map>
#include <iostream>
#include "UniqueArray.h"

using std::cout;
using std::endl;

bool linear_search( const std::vector<double>& data, double* values, int size, int* sp ){
   // I know it can be faster. If data is (indirectly) sorted it will be even faster to rewrite it
//   cout << " Looking for " << values[0] << " of size " << size << " among " << data.size() << endl;
   for( int i=0; i < data.size(); i++ ){
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

UniqueArray::UniqueArray(){
   internal_buffer = std::vector<double>() ;
}

// First horrible inefficient implementation
unsigned int UniqueArray::add( double* values, int size ){

      unsigned int prev_size = (unsigned int) internal_buffer.size();

      int sp=0;
      bool found = linear_search( internal_buffer, values, size, &sp );
      if ( found ){ return sp ; }
      
      internal_buffer.insert(internal_buffer.end(), values, values+size );
      return prev_size;
}

void UniqueArray::show_ia(){
   for ( unsigned int i=0; i < internal_buffer.size(); i++ ){
      std::cout << i << " " << internal_buffer[i] << std::endl ;
   }
}



