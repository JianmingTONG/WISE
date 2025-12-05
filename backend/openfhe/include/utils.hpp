#pragma once
#include <vector>
#include <unordered_set>
#include <string>
#include "openfhe.h"

using namespace lbcrypto;
namespace fheproj {

struct Diagonal {
    int bs = 0;
    int gs = 0;
    std::vector<double> data;
    Plaintext ptxt;
};

struct Block {
    int bx = 0;
    int by = 0;
    std::vector<Diagonal> diags;
};

struct LinearTransformReady {
    int T_rows = 0;
    int T_cols = 0;
    int output_rotations = 0;
    std::vector<std::vector<Block>> mat;
    std::vector<std::vector<double>> bias;
    std::vector<lbcrypto::Plaintext> bias_ptxt;
    std::vector<std::vector<int>> v_bs;
    int num_diags = 0;
    std::unordered_set<int> global_rots;
};

struct WinogradReady {
    int T_rows = 0;
    int T_cols = 0;
    int output_rotations = 0;
    int R = 0;
    int S = 0;
    int Ht = 0;
    int Wt = 0;

    std::vector<std::vector<int>> i2c_target;
    std::vector<std::vector<int>> i2c_offset;
    std::vector<std::vector<int>> A_kron;
    std::vector<std::vector<int>> B_kron;

    std::vector<std::vector<std::vector<Block>>> mats;

    std::vector<std::vector<double>> bias;
    std::vector<lbcrypto::Plaintext> bias_ptxt;

    std::vector<std::vector<int>> v_bs;
    int num_diags = 0;
    std::unordered_set<int> global_rots;
};

struct HerPNReady {
    std::vector<std::vector<double>> a0;
    std::vector<std::vector<double>> a1;
};

struct NonlinearReady {
    std::vector<double> coeffs;
    double a;
    double b;
    int degree;
};

} // namespace fheproj

