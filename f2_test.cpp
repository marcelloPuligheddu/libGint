#include <iostream>

template<int L>
inline int triangular_row(int i) {
    int k = 0;
    for (int j = 1; j <= L; ++j) {
        int tj = j * (j + 1) / 2;
        k += (i >= tj); // Increments k if i >= T(j)
    }
    return k;
}

template<int L>
int value_in_block(int i) {
    if constexpr (L == 0 || L == 1) { return 0; }

    constexpr int nl = (L + 1) * (L + 2) / 2;
    constexpr int nm = (L + 0) * (L + 1) / 2;
    
    int k = triangular_row<L>(i);
    int is_eq_nl_1 = (i == (nl - 1));
    int is_lt_nm   = (i < nm);
    int is_else    = 1 - (is_eq_nl_1 | is_lt_nm);
    int val1 = is_eq_nl_1 * (L - 1);
    int val2 = is_lt_nm   * (L - 1 - k);
    int val3 = is_else    * (L + nm - i - 1);

    return val1 + val2 + val3;
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




