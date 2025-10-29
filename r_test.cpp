#include <iostream>
#include <iomanip>
#include <cstdlib>
// === GPU-friendly triangle value function ===
template <int L>
int tri_value(int i, int r) {
    if constexpr (L == 0) {
        return 0;
    }

    int row = 0;
    int acc = 0;

    #pragma unroll
    for (int k = 1; k <= L + 1; ++k) {
        int row_start = acc;
        acc += k;
        int in_row = (i - row_start) >= 0 && (i - acc) < 0;
        row += in_row * (k - 1);
    }

    int result0 = i;
    int result1 = i - row;
    int result2 = std::max(0,result1 - 1);

    return (r == 0) * result0 +
           (r == 1) * result1 +
           (r == 2) * result2;
}

// === Helper to test a given L ===
template <int L>
void test_level() {
    constexpr int N = (L + 1) * (L + 2) / 2;
    std::cout << "L = " << L << " (" << N << " entries)\n";

    for (int r = 0; r < 3; ++r) {
        std::cout << "r" << r << ": ";
        for (int i = 0; i < N; ++i) {
            std::cout << std::setw(2) << tri_value<L>(i, r) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
}

// === Main Test ===
int main() {
    test_level<0>();
    test_level<1>();
    test_level<2>();
    test_level<3>();
    test_level<4>();
    test_level<5>();
    test_level<6>();
    test_level<7>();
    test_level<8>();
    return 0;
}
