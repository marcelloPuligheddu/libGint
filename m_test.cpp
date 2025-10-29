#include <iostream>

template <int b>
int value_in_block(int i) {
   if constexpr ( b == 0 or b == 1 ){ return 0; 
   } else{
      constexpr int Nw=  (b - 1)*(b + 0)/2;
      constexpr int Nm = (b + 0)*(b + 1)/2;
      constexpr int NL = (b + 1)*(b + 2)/2;
      unsigned ui = static_cast<unsigned>(i);
      unsigned w_ness = 1u - (ui / Nm);
      unsigned m_ness = (ui >= Nm && ui < NL - 1) ? 1u : 0u;
      unsigned l_ness = (ui == (unsigned)(NL - 1)) ? 1u : 0u;
      int val_w = i;
      int val_m = (i - Nm) + Nw;
      int val_l = (b - 1) + Nw;
      return val_w * w_ness + val_m * m_ness + val_l * l_ness;
   }
}

// Example usage and test:
int main() {


    for (int i = 0; i < 1; ++i) {
        std::cout << value_in_block<0>(i) << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << value_in_block<1>(i) << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < 6; ++i) {
        std::cout << value_in_block<2>(i) << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << value_in_block<3>(i) << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < 15; ++i) {
        std::cout << value_in_block<4>(i) << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < 21; ++i) {
        std::cout << value_in_block<5>(i) << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < 28; ++i) {
        std::cout << value_in_block<6>(i) << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < 36; ++i) {
        std::cout << value_in_block<7>(i) << " ";
    }
    std::cout << "\n";





    return 0;
}

