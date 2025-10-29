#include <iostream>

template<int L>
int value_in_block(int i) {
    constexpr int nl = (L + 1) * (L + 2) / 2;
    constexpr int nm = (L + 0) * (L + 1) / 2;
    constexpr int nw = (L - 1) * (L + 0) / 2;
    constexpr int nx = (L - 2) * (L - 1) / 2;

    if constexpr (L <= 2) {
        return 0;
    }

    // Compute masks without branching on i, using boolean arithmetic.
    // bool-to-int conversions in C++: true=1, false=0
    int mask_i_lt_nw = int(i < nw);
    int mask_i_ge_nw = 1 - mask_i_lt_nw;
    int mask_i_lt_nw_plus_L = int(i < nw + L);
    int mask_i_eq_nl_1 = int(i == nl - 1);
    int mask_i_eq_nl_2 = int(i == nl - 2);

    // Calculate return values multiplied by masks and sum:
    int res = 0;
    res += mask_i_lt_nw * i;                       // i < nw : return i
    res += (mask_i_ge_nw && mask_i_lt_nw_plus_L) * 0; // nw <= i < nw + L : 0
    res += mask_i_eq_nl_1 * (nw - 1);              // i == nl-1 : nw-1
    res += mask_i_eq_nl_2 * 0;                      // i == nl-2 : 0

    // Else return i - nm + nx for all other cases
    int mask_else = 1 - (mask_i_lt_nw + (mask_i_ge_nw && mask_i_lt_nw_plus_L) + mask_i_eq_nl_1 + mask_i_eq_nl_2);
    res += mask_else * (i - nm + nx);

    return res;
}


int main(){

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

  
}




