#include <iostream>
#include <iomanip>

// tri_value<L>(i,d):
//   i in [0..(L+1)*(L+2)/2)
//   d ∈ {0,1,2}
// returns the 3 triangular‐matrix rows you specified.
template <int L>
inline int tri_value(int i, int d) {
    if constexpr (L == 0) {
        return 0;
    }

    // 1) find the 'row' containing i, branchless in i
    int row = 0, acc = 0;
    #pragma unroll
    for (int k = 1; k <= L + 1; ++k) {
        int start = acc;
        acc += k;
        int in_row = (i - start) >= 0 && (i - acc) < 0;
        row += in_row * (k - 1);
    }

    // 2) compute 'col' as offset within that row
    int start_of_row = row * (row + 1) / 2;
    int col = i - start_of_row;

    // 3) three outputs
    int r0 = L - row;
    int r1 = row - col;
    int r2 = col;

    // 4) select based on d (only divergence on d, not on i)
    return (d == 0) * r0
         + (d == 1) * r1
         + (d == 2) * r2;
}

// Helper to test one level L
template <int L>
void test_level() {
    constexpr int N = (L + 1) * (L + 2) / 2;
    std::cout << "=== L = " << L << " (N=" << N << ") ===\n";
    for (int d = 0; d < 3; ++d) {
        std::cout << "row " << d << ": ";
        for (int i = 0; i < N; ++i) {
            std::cout << std::setw(2) << tri_value<L>(i, d) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

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
