#include "ciphertext-fwd.h"
#include "constants-defs.h"
#include "context.hpp"
#include "ct_handle.hpp"
#include "lattice/constants-lattice.h"
#include "lattice/stdlatticeparms.h"
#include "openfhe.h"
#include "utils.hpp"

#include <chrono>
#include <random>
#include <stdexcept>

using namespace lbcrypto;

namespace fheproj {

static SecurityLevel map_security_level(const std::string& s) {
    if (s == "HEStd_128_classic") return HEStd_128_classic;
    else if (s == "HEStd_NotSet")
        return HEStd_NotSet;
    else
        return HEStd_NotSet;
}

static SecretKeyDist map_secret_key_dist(const std::string& s) {
    if (s == "GAUSSIAN") return GAUSSIAN;
    else if (s == "UNIFORM_TERNARY")
        return UNIFORM_TERNARY;
    else
        return UNIFORM_TERNARY;
}

static ScalingTechnique map_scaling_technique(const std::string& s) {
    if (s == "FIXEDAUTO") return FIXEDAUTO;
    else if (s == "FLEXIBLEAUTO")
        return FLEXIBLEAUTO;
    else if (s == "FIXEDMANUAL")
        return FIXEDMANUAL;
    else if (s == "FLEXIBLEAUTOEXT")
        return FLEXIBLEAUTOEXT;
    else if (s == "COMPOSITESCALINGAUTO")
        return COMPOSITESCALINGAUTO;
    else if (s == "COMPOSITESCALINGMANUAL")
        return COMPOSITESCALINGMANUAL;
    else
        return FLEXIBLEAUTO;
}

static KeySwitchTechnique map_key_switch_technique(const std::string& s) {
    if (s == "BV") return BV;
    else if (s == "HYBRID")
        return HYBRID;
    else
        return HYBRID;
}

struct FHEContext::Impl {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;
};

FHEContext::~FHEContext() = default;
FHEContext::FHEContext(FHEContext&&) noexcept = default;
FHEContext& FHEContext::operator=(FHEContext&&) noexcept = default;

FHEContext::FHEContext(const FHEContext::Params& p): pimpl_(std::make_unique<Impl>()), slots_(1u << p.log_batch_size), composite_degree_(p.composite_degree) {
    CCParams<CryptoContextCKKSRNS> params;
    auto secret_key_dist = map_secret_key_dist(p.secret_key_dist);
    auto security_level = map_security_level(p.security_level);
    auto scaling_technique = map_scaling_technique(p.scaling_technique);
    auto key_switch_technique = map_key_switch_technique(p.key_switch_technique);
    params.SetSecurityLevel(security_level);
    params.SetSecretKeyDist(secret_key_dist);
    params.SetScalingTechnique(scaling_technique);
    params.SetKeySwitchTechnique(key_switch_technique);
    usint depth
        = p.level_budget.empty() ? p.mult_depth : (p.mult_depth + FHECKKSRNS::GetBootstrapDepth(p.level_budget, secret_key_dist));
    this->depth_ = depth;
    std::cerr << "multiplicative depth: " << depth << std::endl;
    params.SetMultiplicativeDepth(depth);
    params.SetFirstModSize(p.first_mod_size);
    params.SetScalingModSize(p.scale_mod_size);
    params.SetBatchSize(1u << p.log_batch_size);
    params.SetRingDim(1u << p.log_N);
    // For COMPOSITESCALINGAUTO: set register word size, OpenFHE auto-computes composite degree
    // For other techniques with composite_degree > 1: manually set composite degree
    if (scaling_technique == COMPOSITESCALINGAUTO || scaling_technique == COMPOSITESCALINGMANUAL) {
        if (p.register_word_size > 0) {
            params.SetRegisterWordSize(p.register_word_size);
            std::cerr << "SetRegisterWordSize: " << p.register_word_size << std::endl;
        }
        // For COMPOSITESCALINGMANUAL, also set composite degree explicitly
        if (scaling_technique == COMPOSITESCALINGMANUAL && p.composite_degree > 1) {
            params.SetCompositeDegree(p.composite_degree);
            std::cerr << "SetCompositeDegree: " << p.composite_degree << std::endl;
        }
    } else if (p.composite_degree > 1) {
        // Legacy path for non-composite scaling techniques
        params.SetCompositeDegree(p.composite_degree);
        std::cerr << "SetCompositeDegree: " << p.composite_degree << std::endl;
    }

    pimpl_->cc = GenCryptoContext(params);
    auto& cc = pimpl_->cc;

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    if (!p.level_budget.empty()) cc->Enable(FHE);

    pimpl_->keys = cc->KeyGen();

    cc->EvalMultKeyGen(pimpl_->keys.secretKey);
    if (!p.level_budget.empty()) {
        std::cerr << "EvalBootstrapSetup with level_budget: " << p.level_budget << std::endl;
        cc->EvalBootstrapSetup(p.level_budget);
        cc->EvalBootstrapKeyGen(pimpl_->keys.secretKey, 1 << p.log_batch_size);
    }
    cc->EvalRotateKeyGen(pimpl_->keys.secretKey, p.global_rots);

    auto print_moduli_chain = [&](const DCRTPoly& poly) {
        int num_primes = poly.GetNumOfElements();
        double total_bit_len = 0.0;
        for (int i = 0; i < num_primes; i++) {
            auto qi = poly.GetParams()->GetParams()[i]->GetModulus();
            std::cout << "q_" << i << ": " << qi << ",  log q_" << i << ": " << log(qi.ConvertToDouble()) / log(2) << std::endl;
            total_bit_len += log(qi.ConvertToDouble()) / log(2);
        }
        std::cout << "Total bit length: " << total_bit_len << std::endl;
    };
    print_moduli_chain(pimpl_->keys.publicKey->GetPublicElements()[0]);
}

std::string FHEContext::encode_plaintext_bytes(const std::vector<double>& values) const {
    Plaintext pt = pimpl_->cc->MakeCKKSPackedPlaintext(values);

    std::ostringstream os(std::ios::binary);
    Serial::Serialize(pt, os, SerType::BINARY);
    return os.str();
}

std::shared_ptr<CiphertextHandle> FHEContext::encrypt(const std::vector<double>& v, uint32_t level) const {
    if (v.size() != slots_) [[unlikely]] { throw std::invalid_argument("encrypt: len != slots"); }
    auto& cc = pimpl_->cc;
    // With FLEXIBLEAUTO, always encrypt at level 0 (maximum depth)
    // The level parameter is ignored - OpenFHE manages levels automatically
    std::cerr << "encrypt at level: 0 (FLEXIBLEAUTO mode)" << std::endl;
    auto pt = cc->MakeCKKSPackedPlaintext(v, 1, 0);
    auto h = std::make_shared<CiphertextHandle>();
    h->mut() = cc->Encrypt(pimpl_->keys.publicKey, pt);
    return h;
}

std::vector<std::shared_ptr<CiphertextHandle>> FHEContext::encrypt_batch(const std::vector<std::vector<double>>& batch,
                                                                         uint32_t level) const {
    std::vector<std::shared_ptr<CiphertextHandle>> out;
    out.reserve(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        if (batch[i].size() != slots_) [[unlikely]]
            throw std::invalid_argument("encrypt_batch: row " + std::to_string(i) + " len != slots");
        out.emplace_back(encrypt(batch[i], level));
    }
    return out;
}

std::vector<double> FHEContext::decrypt(const std::shared_ptr<CiphertextHandle>& h, size_t length) const {
    if (!h) throw std::invalid_argument("decrypt: null handle");
    auto& cc = pimpl_->cc;

    Plaintext pt;
    cc->Decrypt(pimpl_->keys.secretKey, h->get(), &pt);
    uint32_t L = (length == 0) ? slots_ : static_cast<uint32_t>(length);
    pt->SetLength(L);

    auto vals = pt->GetRealPackedValue();
    if (vals.size() > L) vals.resize(L);
    return vals;
}

std::vector<std::shared_ptr<CiphertextHandle>> FHEContext::deepcopy_ciphertexts(const std::vector<std::shared_ptr<CiphertextHandle>>& x) const {
    std::vector<std::shared_ptr<CiphertextHandle>> y;
    y.reserve(x.size());
    for (const auto& h: x) {
        if (!h) [[unlikely]] throw std::invalid_argument("eval_add_batch: null handle");
        auto nh = std::make_shared<CiphertextHandle>();
        nh->mut() = h->get()->Clone();
        y.emplace_back(std::move(nh));
    }
    return y;
};

std::tuple<size_t, size_t, int128_t, size_t, size_t> FHEContext::stats(const std::shared_ptr<CiphertextHandle>& h) const {
    auto c = h->get();
    size_t level = c->GetLevel();
    size_t noise_scale_deg = c->GetNoiseScaleDeg();
    int128_t scaling_factor = c->GetScalingFactor();
    size_t num_primes = c->GetElements()[0].GetNumOfElements();
    size_t rem = this->depth_ - c->GetLevel() - (c->GetNoiseScaleDeg() - 1);
    return {level, noise_scale_deg, scaling_factor, num_primes, rem};
}

std::shared_ptr<CiphertextHandle> FHEContext::eval_add(const std::shared_ptr<CiphertextHandle>& a,
                                                       const std::shared_ptr<CiphertextHandle>& b) const {
    if (!a || !b) [[unlikely]] { throw std::invalid_argument("eval_add_in_place: null handle"); }
    const auto& ca = a->get();
    const auto& cb = b->get();
    auto& cc = pimpl_->cc;
    auto sum = cc->EvalAdd(ca, cb);
    auto h = std::make_shared<CiphertextHandle>();
    h->mut() = sum;
    return h;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_add_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& as,
                           const std::vector<std::shared_ptr<CiphertextHandle>>& bs) const {
    if (as.size() != bs.size()) [[unlikely]] { throw std::invalid_argument("eval_add_batch: size mismatch"); }
    std::vector<std::shared_ptr<CiphertextHandle>> out;
    out.reserve(as.size());
    auto& cc = pimpl_->cc;
    for (size_t i = 0; i < as.size(); ++i) {
        if (!as[i] || !bs[i]) [[unlikely]] {
            throw std::invalid_argument("eval_add_batch: null handle at index " + std::to_string(i));
        }

        const auto& ca = as[i]->get();
        const auto& cb = bs[i]->get();

        auto sum = cc->EvalAdd(ca, cb);

        auto h = std::make_shared<CiphertextHandle>();
        h->mut() = sum;
        out.emplace_back(std::move(h));
    }
    return out;
}

void FHEContext::eval_add_in_place(const std::shared_ptr<CiphertextHandle>& a, const std::shared_ptr<CiphertextHandle>& b) const {
    if (!a || !b) [[unlikely]] { throw std::invalid_argument("eval_add_in_place: null handle"); }
    auto& ca = a->mut();
    const auto& cb = b->get();
    auto& cc = pimpl_->cc;
    cc->EvalAddInPlace(ca, cb);
}

void FHEContext::eval_add_in_place_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& A,
                                         const std::vector<std::shared_ptr<CiphertextHandle>>& B) const {
    if (A.size() != B.size()) [[unlikely]] { throw std::invalid_argument("eval_add_in_place_batch: size mismatch"); }
    auto& cc = pimpl_->cc;
    for (size_t i = 0; i < A.size(); ++i) {
        if (!A[i] || !B[i]) [[unlikely]] {
            throw std::invalid_argument("eval_add_in_place_batch: null handle at index " + std::to_string(i));
        }
        auto& ca = A[i]->mut();
        const auto& cb = B[i]->get();

        cc->EvalAddInPlace(ca, cb);
    }
}

std::shared_ptr<CiphertextHandle> FHEContext::level_reduce(const std::shared_ptr<CiphertextHandle>& a, size_t level) const {
    if (!a) [[unlikely]]
        throw std::invalid_argument("level_reduce_to_primes_in_place: null handle");
    auto& cc = pimpl_->cc;
    auto& ct = a->get();
    auto b = cc->LevelReduce(ct, nullptr, level);
    auto h = std::make_shared<CiphertextHandle>();
    h->mut() = std::move(b);
    return h;
}

void FHEContext::level_reduce_in_place(const std::shared_ptr<CiphertextHandle>& a, size_t level) const {
    if (!a) [[unlikely]]
        return;
    if (level == 0) [[unlikely]]
        return;
    auto& cc = pimpl_->cc;
    auto& ct = a->mut();
    cc->LevelReduceInPlace(ct, nullptr, level);
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::level_reduce_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& as, size_t level) const {
    if (level == 0) [[unlikely]] {
        auto out = as;
        return out;
    }
    std::vector<std::shared_ptr<CiphertextHandle>> out;
    out.reserve(as.size());
    auto& cc = pimpl_->cc;
    for (size_t i = 0; i < as.size(); ++i) {
        const auto& ah = as[i];
        if (!ah) [[unlikely]] { throw std::invalid_argument("level_reduce_batch: null handle at index " + std::to_string(i)); }
        const auto& ct = ah->get();
        auto ct_new = cc->LevelReduce(ct, nullptr, level);
        auto h = std::make_shared<CiphertextHandle>();
        h->mut() = std::move(ct_new);
        out.emplace_back(std::move(h));
    }
    return out;
}

void FHEContext::level_reduce_in_place_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& as, size_t level) const {
    if (level == 0) [[unlikely]]
        return;
    auto& cc = pimpl_->cc;
    for (size_t i = 0; i < as.size(); ++i) {
        if (!as[i]) [[unlikely]] { throw std::invalid_argument("level_reduce_batch: null handle at index " + std::to_string(i)); }
        cc->LevelReduceInPlace(as[i]->mut(), nullptr, level);
    }
}

std::vector<std::vector<double>> FHEContext::decrypt_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& hs,
                                                           size_t length) const {
    std::vector<std::vector<double>> out;
    out.reserve(hs.size());
    for (size_t i = 0; i < hs.size(); ++i) {
        if (!hs[i]) [[unlikely]] { throw std::invalid_argument("decrypt_batch: null handle at row " + std::to_string(i)); }
        out.emplace_back(decrypt(hs[i], length));
    }
    return out;
}

std::vector<Ciphertext<DCRTPoly>> mat_x_vec(const CryptoContext<DCRTPoly>& cc,
                                            const std::vector<Ciphertext<DCRTPoly>>& x,
                                            const std::vector<std::vector<Block>>& mat,
                                            const std::vector<std::vector<int>>& v_bs,
                                            size_t T_rows,
                                            size_t T_cols,
                                            int level,
                                            int start,
                                            int total,
                                            double& pack_time,
                                            int& num_adds,
                                            int& num_mults,
                                            int& num_rots) {
    uint32_t CyclotomicOrder = cc->GetCyclotomicOrder();
    size_t n = x.size();

    std::vector<std::unordered_map<int, Ciphertext<DCRTPoly>>> x_bs(n);
    for (int i = 0; i < n; ++i) {
        auto ct_precomp = cc->EvalFastRotationPrecompute(x[i]);
        for (int bs: v_bs[i]) {
            if (bs != 0) {
                x_bs[i][bs] = cc->EvalFastRotation(x[i], bs, CyclotomicOrder, ct_precomp);
                ++num_rots;
            } else {
                x_bs[i][bs] = x[i];
            }
        }
    }

    std::vector<Ciphertext<DCRTPoly>> outputs(T_rows);

    for (int bx = 0; bx < T_rows; ++bx) {
        std::unordered_map<int, std::vector<Ciphertext<DCRTPoly>>> mp;
        for (int by = 0; by < T_cols; ++by) {
            std::cerr << start + bx * T_cols + by << "/" << total << "\r";

            // std::vector<Plaintext> ptxts(mat[bx][by].diags.size());
            // auto pack_time_start = std::chrono::high_resolution_clock::now();
            // for (size_t i = 0; i < mat[bx][by].diags.size(); ++i) {
            //     ptxts[i] = cc->MakeCKKSPackedPlaintext(mat[bx][by].diags[i].data, 1, level);
            // }
            // auto pack_time_end = std::chrono::high_resolution_clock::now();
            // pack_time += std::chrono::duration<double>(pack_time_end - pack_time_start).count();

            for (size_t i = 0; i < mat[bx][by].diags.size(); ++i) {
                int bs = mat[bx][by].diags[i].bs;
                int gs = mat[bx][by].diags[i].gs;

                auto pack_time_start = std::chrono::high_resolution_clock::now();
                // With FLEXIBLEAUTO, create plaintexts at depth 0 (level 0)
                // OpenFHE automatically adjusts levels during operations
                Plaintext ptxt = cc->MakeCKKSPackedPlaintext(mat[bx][by].diags[i].data, 1, 0);
                auto pack_time_end = std::chrono::high_resolution_clock::now();
                pack_time += std::chrono::duration<double>(pack_time_end - pack_time_start).count();

                auto out = cc->EvalMult(x_bs[by][bs], ptxt);
                ++num_mults;
                mp[gs].emplace_back(out);
            }
        }
        std::vector<Ciphertext<DCRTPoly>> sums;
        for (auto& [gs, ciphers]: mp) {
            auto sum = cc->EvalAddMany(ciphers);
            num_adds += static_cast<int>(ciphers.size()) - 1;
            sums.emplace_back(gs == 0 ? sum : cc->EvalRotate(sum, gs));
        }
        outputs[bx] = cc->EvalAddMany(sums);
        num_adds += static_cast<int>(sums.size()) - 1;
    }

    return outputs;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_linear_transform(const std::vector<std::shared_ptr<CiphertextHandle>>& x,
                                  const LinearTransformReady& lt,
                                  int level) const {
    auto TIME_START = std::chrono::high_resolution_clock::now();

    auto keys = pimpl_->keys;
    if (x.empty()) [[unlikely]] { throw std::invalid_argument("eval_linear_transform: empty input vector"); }
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        if (!x[i]) [[unlikely]] {
            throw std::invalid_argument("eval_linear_transform: null handle at index " + std::to_string(i));
        }
    }
    auto& cc = pimpl_->cc;
    std::vector<Ciphertext<DCRTPoly>> cs(n);
    for (size_t i = 0; i < n; ++i) { cs[i] = x[i]->get(); }
    double pack_time = 0.0;
    int num_adds = 0;
    int num_mults = 0;
    int num_rots = 0;
    auto outputs = mat_x_vec(cc,
                             cs,
                             lt.mat,
                             lt.v_bs,
                             lt.T_rows,
                             lt.T_cols,
                             level,
                             0,
                             lt.T_rows * lt.T_cols,
                             pack_time,
                             num_adds,
                             num_mults,
                             num_rots);

    std::vector<std::shared_ptr<CiphertextHandle>> outputs_ptr(outputs.size());
    {
        auto time_start = std::chrono::high_resolution_clock::now();
        std::vector<Plaintext> bias_ptxts(lt.T_rows);
        // With FLEXIBLEAUTO, create plaintexts at depth 0 (level 0)
        for (int i = 0; i < lt.T_rows; ++i) { bias_ptxts[i] = cc->MakeCKKSPackedPlaintext(lt.bias[i], 1, 0); }
        auto time_end = std::chrono::high_resolution_clock::now();
        pack_time += std::chrono::duration<double>(time_end - time_start).count();
        if (lt.T_rows != 1 && lt.output_rotations != 0) [[unlikely]] {
            throw std::invalid_argument("eval_linear_transform: non-zero output rotations not supported");
        }
        for (size_t i = 0; i < lt.T_rows; ++i) {
            cc->EvalAddInPlace(outputs[i], bias_ptxts[i]);

            for (size_t r = 0, offset = this->slots_; r < lt.output_rotations; ++r, offset >>= 1) {
                auto t = cc->EvalRotate(outputs[i], offset >> 1);
                cc->EvalAddInPlace(outputs[i], t);
            }

            // With composite scaling (composite_degree > 1), each "level" consists of
            // composite_degree primes. OpenFHE's RescaleInPlace drops the entire composite
            // level when SetCompositeDegree is configured. We only call RescaleInPlace ONCE.
            cc->RescaleInPlace(outputs[i]);

            outputs_ptr[i] = std::make_shared<CiphertextHandle>();
            outputs_ptr[i]->mut() = std::move(outputs[i]);
        }
    }

    auto TIME_END = std::chrono::high_resolution_clock::now();

    std::cerr << "linear transform @ level=" << level << std::endl
              << "  total time: " << std::chrono::duration<double>(TIME_END - TIME_START).count()
              << "s\n  pack time: " << pack_time << "s\n  num_adds: " << num_adds << "\n  num_mults: " << num_mults
              << "\n  num_rots: " << num_rots << std::endl
              << std::endl;
    ;

    return outputs_ptr;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_winograd(const std::vector<std::shared_ptr<CiphertextHandle>>& x, const WinogradReady& wino, int level) const {
    auto TIME_START = std::chrono::high_resolution_clock::now();

    auto keys = pimpl_->keys;
    if (x.empty()) [[unlikely]] { throw std::invalid_argument("eval_winograd: empty input vector"); }
    for (size_t i = 0; i < x.size(); ++i) {
        if (!x[i]) [[unlikely]] { throw std::invalid_argument("eval_winograd: null handle at index " + std::to_string(i)); }
    }
    auto& cc = pimpl_->cc;
    uint32_t CyclotomicOrder = cc->GetCyclotomicOrder();
    int ts_out = wino.Ht * wino.Wt, ts_in = (wino.Ht + wino.R - 1) * (wino.Wt + wino.S - 1);

    double pack_time = 0.0;
    int num_adds = 0;
    int num_mults = 0;
    int num_rots = 0;

    auto linear_sum_tree_01
        = [&cc, &num_adds](std::vector<Ciphertext<DCRTPoly>>& inputs, const std::vector<int>& weights) -> Ciphertext<DCRTPoly> {
        int wsum = 0;
        for (int i: weights) wsum += i;
        int negate = wsum >= 0 ? 0 : 1;

        std::vector<int> pool[2];
        for (int i = 0; i < (int)weights.size(); ++i) {
            if (weights[i] == 1) pool[0 ^ negate].emplace_back(i);
            else if (weights[i] == -1)
                pool[1 ^ negate].emplace_back(i);
        }
        std::vector<Ciphertext<DCRTPoly>> c;
        uint32_t i = 0, j = 0;
        for (; i < pool[0].size() && j < pool[1].size(); ++i, ++j) {
            c.emplace_back(cc->EvalSub(inputs[pool[0][i]], inputs[pool[1][j]]));
            ++num_adds;
        }
        for (; i + 1 < pool[0].size(); i += 2) {
            c.emplace_back(cc->EvalAdd(inputs[pool[0][i]], inputs[pool[0][i + 1]]));
            ++num_adds;
        }
        if (i < pool[0].size()) c.emplace_back(inputs[pool[0][i]]);
        Ciphertext<DCRTPoly> result = cc->EvalAddMany(c);
        num_adds += (int)c.size() - 1;
        if (negate == 1) cc->EvalNegateInPlace(result), ++num_adds;
        return result;
    };

    std::vector<std::vector<Ciphertext<DCRTPoly>>> d_tilde(ts_in, std::vector<Ciphertext<DCRTPoly>>(wino.T_cols));
    {
        std::vector<std::vector<Ciphertext<DCRTPoly>>> xt(ts_out, std::vector<Ciphertext<DCRTPoly>>(wino.T_cols));

        for (size_t i = 0; i < x.size(); ++i) {
            size_t tile_idx = i / wino.T_cols;
            size_t cipher_idx = i % wino.T_cols;
            xt[tile_idx][cipher_idx] = x[i]->get();
        }

        std::vector<std::vector<Ciphertext<DCRTPoly>>> c(wino.T_cols, std::vector<Ciphertext<DCRTPoly>>(ts_in));
        for (int k = 0; k < ts_out; ++k) {
            for (int i = 0; i < wino.T_cols; ++i) {
                auto ct_precomp = cc->EvalFastRotationPrecompute(xt[k][i]);
                for (int j = 0; j < ts_out; ++j) {
                    size_t tar = wino.i2c_target[k][j], off = wino.i2c_offset[k][j];
                    c[i][tar] = cc->EvalFastRotation(xt[k][i], off, CyclotomicOrder, ct_precomp);
                    ++num_rots;
                }
            }
        }
        for (int i = 0; i < ts_in; ++i) {
            for (int j = 0; j < wino.T_cols; ++j) { d_tilde[i][j] = linear_sum_tree_01(c[j], wino.B_kron[i]); }
        }
    }

    std::vector<Ciphertext<DCRTPoly>> outputs(ts_out * wino.T_rows);
    {
        std::vector<std::vector<Ciphertext<DCRTPoly>>> e_tilde(wino.T_rows, std::vector<Ciphertext<DCRTPoly>>(ts_in));
        for (int i = 0; i < ts_in; ++i) {
            auto outputs = mat_x_vec(cc,
                                     d_tilde[i],
                                     wino.mats[i],
                                     wino.v_bs,
                                     wino.T_rows,
                                     wino.T_cols,
                                     level,
                                     i * wino.T_rows * wino.T_cols,
                                     ts_in * wino.T_rows * wino.T_cols,
                                     pack_time,
                                     num_adds,
                                     num_mults,
                                     num_rots);
            for (int j = 0; j < wino.T_rows; ++j) e_tilde[j][i] = outputs[j];
        }
        for (int i = 0; i < wino.T_rows; ++i) {
            for (int j = 0; j < ts_out; ++j) {
                int idx = j * wino.T_rows + i;
                outputs[idx] = linear_sum_tree_01(e_tilde[i], wino.A_kron[j]);
            }
        }
    }

    std::vector<std::shared_ptr<CiphertextHandle>> outputs_ptr(outputs.size());
    {
        auto time_start = std::chrono::high_resolution_clock::now();
        std::vector<Plaintext> bias_ptxts(wino.bias.size());
        // With FLEXIBLEAUTO, create plaintexts at depth 0 (level 0)
        for (int i = 0; i < wino.bias.size(); ++i) { bias_ptxts[i] = cc->MakeCKKSPackedPlaintext(wino.bias[i], 1, 0); }
        auto time_end = std::chrono::high_resolution_clock::now();
        pack_time += std::chrono::duration<double>(time_end - time_start).count();

        if (wino.T_rows != 1 && wino.output_rotations != 0) [[unlikely]] {
            throw std::invalid_argument("eval_linear_transform: non-zero output rotations not supported");
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            cc->EvalAddInPlace(outputs[i], bias_ptxts[i]);

            for (size_t r = 0, offset = this->slots_; r < wino.output_rotations; ++r, offset >>= 1) {
                auto t = cc->EvalRotate(outputs[i], offset >> 1);
                cc->EvalAddInPlace(outputs[i], t);
            }

            // With composite scaling, RescaleInPlace drops the entire composite level
            cc->RescaleInPlace(outputs[i]);

            outputs_ptr[i] = std::make_shared<CiphertextHandle>();
            outputs_ptr[i]->mut() = std::move(outputs[i]);
        }
    }

    auto TIME_END = std::chrono::high_resolution_clock::now();

    std::cerr << "winograd @ level=" << level << std::endl
              << "  total time: " << std::chrono::duration<double>(TIME_END - TIME_START).count()
              << "s\n  pack time: " << pack_time << "s\n  num_adds: " << num_adds << "\n  num_mults: " << num_mults
              << "\n  num_rots: " << num_rots << std::endl
              << std::endl;

    return outputs_ptr;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_winograd_advance(const std::vector<std::shared_ptr<CiphertextHandle>>& x,
                                  const WinogradReady& wino,
                                  int level) const {
    return {};
    // auto TIME_START = std::chrono::high_resolution_clock::now();

    // auto keys = pimpl_->keys;
    // if (x.empty()) [[unlikely]] { throw std::invalid_argument("eval_winograd: empty input vector"); }
    // for (size_t i = 0; i < x.size(); ++i) {
    //     if (!x[i]) [[unlikely]] { throw std::invalid_argument("eval_winograd: null handle at index " + std::to_string(i)); }
    // }
    // auto& cc = pimpl_->cc;
    // uint32_t CyclotomicOrder = cc->GetCyclotomicOrder();
    // int ts_out = wino.Ht * wino.Wt, ts_in = (wino.Ht + wino.R - 1) * (wino.Wt + wino.S - 1);

    // double pack_time = 0.0;
    // int num_adds = 0;
    // int num_mults = 0;
    // int num_rots = 0;

    // auto linear_sum_tree_01
    //     = [&cc, &num_adds](std::vector<Ciphertext<DCRTPoly>>& inputs, const std::vector<int>& weights) -> Ciphertext<DCRTPoly>
    //     { int wsum = 0; for (int i: weights) wsum += i; int negate = wsum >= 0 ? 0 : 1;

    //     std::vector<int> pool[2];
    //     for (int i = 0; i < (int)weights.size(); ++i) {
    //         if (weights[i] == 1) pool[0 ^ negate].emplace_back(i);
    //         else if (weights[i] == -1)
    //             pool[1 ^ negate].emplace_back(i);
    //     }
    //     std::vector<Ciphertext<DCRTPoly>> c;
    //     uint32_t i = 0, j = 0;
    //     for (; i < pool[0].size() && j < pool[1].size(); ++i, ++j) {
    //         c.emplace_back(cc->EvalSub(inputs[pool[0][i]], inputs[pool[1][j]]));
    //         ++num_adds;
    //     }
    //     for (; i + 1 < pool[0].size(); i += 2) {
    //         c.emplace_back(cc->EvalAdd(inputs[pool[0][i]], inputs[pool[0][i + 1]]));
    //         ++num_adds;
    //     }
    //     if (i < pool[0].size()) c.emplace_back(inputs[pool[0][i]]);
    //     Ciphertext<DCRTPoly> result = cc->EvalAddMany(c);
    //     num_adds += (int)c.size() - 1;
    //     if (negate == 1) cc->EvalNegateInPlace(result), ++num_adds;
    //     return result;
    // };

    // std::vector<std::vector<std::unordered_map<int, Ciphertext<DCRTPoly>>>> e_tilde_bs(ts_in,
    // std::vector<Ciphertext<DCRTPoly>>(wino.T_cols));
    // {
    //     std::vector<std::vector<Ciphertext<DCRTPoly>>> xt(ts_out, std::vector<Ciphertext<DCRTPoly>>(wino.T_cols));
    //     for (size_t i = 0; i < x.size(); ++i) {
    //         size_t tile_idx = i / wino.T_cols;
    //         size_t cipher_idx = i % wino.T_cols;
    //         xt[tile_idx][cipher_idx] = x[i]->mut();
    //         size_t target_level = this->depth_ - target_levels_remaining - (xt[tile_idx][cipher_idx]->GetNoiseScaleDeg() - 1);
    //         size_t level = xt[tile_idx][cipher_idx]->GetLevel();
    //         if (level > target_level) [[unlikely]] {
    //             throw std::invalid_argument("eval_winograd: input at index " + std::to_string(i)
    //                                         + " has level less than target_level");
    //         }
    //         if (level < target_level) { cc->LevelReduceInPlace(xt[tile_idx][cipher_idx], nullptr, target_level - level); }
    //     }
    //     std::vector<std::vector<Ciphertext<DCRTPoly>>> c(wino.T_cols, std::vector<Ciphertext<DCRTPoly>>(ts_in));
    //     for (int k = 0; k < ts_out; ++k) {
    //         for (int i = 0; i < wino.T_cols; ++i) {
    //             auto ct_precomp = cc->EvalFastRotationPrecompute(xt[k][i]);
    //             for (int j = 0; j < ts_out; ++j) {
    //                 size_t tar = wino.i2c_target[k][j], off = wino.i2c_offset[k][j];
    //                 c[i][tar] = cc->EvalFastRotation(xt[k][i], off, CyclotomicOrder, ct_precomp);
    //                 ++num_rots;
    //             }
    //         }
    //     }
    //     for (int i = 0; i < ts_in; ++i) {
    //         for (int j = 0; j < wino.T_cols; ++j) { d_tilde[i][j] = linear_sum_tree_01(c[j], wino.B_kron[i]); }
    //     }
    // }

    // std::vector<Ciphertext<DCRTPoly>> outputs(ts_out * wino.T_rows);
    // {
    //     std::vector<std::vector<Ciphertext<DCRTPoly>>> e_tilde(wino.T_rows, std::vector<Ciphertext<DCRTPoly>>(ts_in));
    //     for (int i = 0; i < ts_in; ++i) {
    //         auto outputs = mat_x_vec(cc,
    //                                  d_tilde[i],
    //                                  wino.mats[i],
    //                                  wino.v_bs,
    //                                  wino.T_rows,
    //                                  wino.T_cols,
    //                                  this->depth_ - target_levels_remaining,
    //                                  keys,
    //                                  pack_time,
    //                                  num_adds,
    //                                  num_mults,
    //                                  num_rots);
    //         for (int j = 0; j < wino.T_rows; ++j) e_tilde[j][i] = outputs[j];
    //     }
    //     for (int i = 0; i < wino.T_rows; ++i) {
    //         for (int j = 0; j < ts_out; ++j) {
    //             int idx = j * wino.T_rows + i;
    //             outputs[idx] = linear_sum_tree_01(e_tilde[i], wino.A_kron[j]);
    //         }
    //     }
    // }

    // std::vector<std::shared_ptr<CiphertextHandle>> outputs_ptr(outputs.size());
    // {
    //     auto time_start = std::chrono::high_resolution_clock::now();
    //     std::vector<Plaintext> bias_ptxts(wino.bias.size());
    //     for (int i = 0; i < wino.bias.size(); ++i) {
    //         bias_ptxts[i] = cc->MakeCKKSPackedPlaintext(wino.bias[i], 1, (this->depth_ - target_levels_remaining) + 1);
    //     }
    //     auto time_end = std::chrono::high_resolution_clock::now();
    //     pack_time += std::chrono::duration<double>(time_end - time_start).count();
    //     for (size_t i = 0; i < outputs.size(); ++i) {
    //         cc->EvalAddInPlace(outputs[i], bias_ptxts[i]);
    //         outputs_ptr[i] = std::make_shared<CiphertextHandle>();
    //         outputs_ptr[i]->mut() = std::move(outputs[i]);
    //     }
    // }

    // auto TIME_END = std::chrono::high_resolution_clock::now();

    // std::cerr << "winograd @ level=" << target_levels_remaining << std::endl
    //           << "  total time: " << std::chrono::duration<double>(TIME_END - TIME_START).count()
    //           << "s\n  pack time: " << pack_time << "s\n  num_adds: " << num_adds << "\n  num_mults: " << num_mults
    //           << "\n  num_rots: " << num_rots << std::endl;

    // return outputs_ptr;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_bootstrap_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& x) const {
    auto& cc = pimpl_->cc;
    size_t n = x.size();
    std::vector<std::shared_ptr<CiphertextHandle>> y(n);

    auto time_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < x.size(); ++i) {
        std::cerr << "bootstrapping " << i << "/" << n << "\r";
        if (!x[i]) [[unlikely]] {
            throw std::invalid_argument("eval_bootstrap_in_place_batch: null handle at index " + std::to_string(i));
        }
        auto t = cc->EvalBootstrap(x[i]->get());
        // With SetCompositeDegree, a single RescaleInPlace drops compositeDegree primes
        cc->RescaleInPlace(t);
        y[i] = std::make_shared<CiphertextHandle>();
        y[i]->mut() = t;
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    std::cerr << "bootstrap batch of size " << n << std::endl
              << "  total time: " << std::chrono::duration<double>(time_end - time_start).count() << "s" << std::endl << std::endl;
    return y;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_silu_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& x, double a, double b, uint32_t degree) const {
    for (size_t i = 0; i < x.size(); ++i) {
        if (!x[i]) [[unlikely]] {
            throw std::invalid_argument("eval_silu_in_place_batch: null handle at index " + std::to_string(i));
        }
    }
    std::vector<std::shared_ptr<CiphertextHandle>> outputs(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        auto& cc = pimpl_->cc;
        auto sigmoid = cc->EvalLogistic(x[i]->get(), a, b, degree);
        outputs[i] = std::make_shared<CiphertextHandle>();
        outputs[i]->mut() = cc->EvalMult(x[i]->get(), sigmoid);
    }
    return outputs;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_herpn(const std::vector<std::shared_ptr<CiphertextHandle>>& x, const HerPNReady& hp, int level) const {
    auto TIME_START = std::chrono::high_resolution_clock::now();

    auto keys = pimpl_->keys;
    if (x.empty()) [[unlikely]] { throw std::invalid_argument("eval_herpn: empty input vector"); }
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        if (!x[i]) [[unlikely]] {
            throw std::invalid_argument("eval_herpn: null handle at index " + std::to_string(i));
        }
    }
    auto& cc = pimpl_->cc;
    std::vector<Ciphertext<DCRTPoly>> cs(n);
    for (size_t i = 0; i < n; ++i) { cs[i] = x[i]->get(); }

    // Debug: print input ciphertext state
    std::cerr << "herpn input state:" << std::endl;
    for (size_t i = 0; i < std::min(n, size_t(2)); ++i) {
        std::cerr << "  x[" << i << "]: level=" << cs[i]->GetLevel()
                  << ", noiseScaleDeg=" << cs[i]->GetNoiseScaleDeg()
                  << ", numTowers=" << cs[i]->GetElements()[0].GetNumOfElements() << std::endl;
    }
    std::cerr << "  creating a0 at level=" << (level + 1) << ", a1 at level=" << level << std::endl;
    std::cerr << "  mult_depth=" << this->depth_ << std::endl;

    double pack_time = 0.0;
    int num_adds = 0;
    int num_mults_cc = 0;
    int num_mults_cp = 0;
    int num_rots = 0;

    std::vector<Plaintext> a0_ptxts(n), a1_ptxts(n);
    {
        auto time_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) {
            // With FLEXIBLEAUTO, create plaintexts at depth 0 (level 0)
            // OpenFHE will automatically adjust levels during operations
            // a1 is multiplied with x (noiseScaleDeg=1)
            // a0 is added after rescaling
            a0_ptxts[i] = cc->MakeCKKSPackedPlaintext(hp.a0[i], 1, 0);
            a1_ptxts[i] = cc->MakeCKKSPackedPlaintext(hp.a1[i], 1, 0);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        pack_time += std::chrono::duration<double>(time_end - time_start).count();
    }

    std::vector<std::shared_ptr<CiphertextHandle>> outputs_ptr(n);
    for (size_t i = 0; i < n; ++i) {
        // Polynomial evaluation: x^2 + a1*x + a0
        // Step 1: x1 = x * a1 (noiseScaleDeg becomes 2)
        auto x1 = cc->EvalMult(cs[i], a1_ptxts[i]);
        // Step 2: x^2 (noiseScaleDeg becomes 2)
        cc->EvalSquareInPlace(cs[i]);
        // Step 3: x^2 + x1 (both at noiseScaleDeg=2)
        cc->EvalAddInPlace(cs[i], x1);
        // Step 4: Rescale - with composite scaling, this drops the entire composite level
        cc->RescaleInPlace(cs[i]);
        // Step 5: Add a0
        cc->EvalAddInPlace(cs[i], a0_ptxts[i]);
        outputs_ptr[i] = std::make_shared<CiphertextHandle>();
        outputs_ptr[i]->mut() = std::move(cs[i]);
    }

    num_adds += 2 * n;
    num_mults_cc += n;
    num_mults_cp += n;

    auto TIME_END = std::chrono::high_resolution_clock::now();

    std::cerr << "herpn @ level=" << level << std::endl
              << "  total time: " << std::chrono::duration<double>(TIME_END - TIME_START).count()
              << "s\n  pack time: " << pack_time << "s\n  num_adds: " << num_adds << "\n  num_mults_cc: " << num_mults_cc
              << "\n  num_mults_cp: " << num_mults_cp << "\n  num_rots: " << num_rots << std::endl
              << std::endl;

    return outputs_ptr;
}

std::vector<std::shared_ptr<CiphertextHandle>>
FHEContext::eval_nonlinear(const std::vector<std::shared_ptr<CiphertextHandle>>& x, const NonlinearReady& nonlinear, int level) const {
    auto TIME_START = std::chrono::high_resolution_clock::now();

    auto keys = pimpl_->keys;
    if (x.empty()) [[unlikely]] { throw std::invalid_argument("eval_nonlinear: empty input vector"); }
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        if (!x[i]) [[unlikely]] {
            throw std::invalid_argument("eval_nonlinear: null handle at index " + std::to_string(i));
        }
    }
    auto& cc = pimpl_->cc;
    std::vector<Ciphertext<DCRTPoly>> cs(n);
    for (size_t i = 0; i < n; ++i) { cs[i] = x[i]->get(); }
    double pack_time = 0.0;
    int num_adds = 0;
    int num_mults_cc = 0;
    int num_mults_cp = 0;
    int num_rots = 0;

    std::vector<std::shared_ptr<CiphertextHandle>> outputs_ptr(n);
    for (size_t i = 0; i < n; ++i) {
        std::cerr << "Eval chebyshev " << i << "/" << n << "\r";
        cs[i] = cc->EvalChebyshevSeries(cs[i], nonlinear.coeffs, nonlinear.a, nonlinear.b);
        outputs_ptr[i] = std::make_shared<CiphertextHandle>();
        outputs_ptr[i]->mut() = std::move(cs[i]);
    }

    auto TIME_END = std::chrono::high_resolution_clock::now();

    std::cerr << "nonlinear @ level=" << level << std::endl
              << "  total time: " << std::chrono::duration<double>(TIME_END - TIME_START).count()
              << "s\n  pack time: " << pack_time << "s\n  num_adds: " << num_adds << "\n  num_mults_cc: " << num_mults_cc
              << "\n  num_mults_cp: " << num_mults_cp << "\n  num_rots: " << num_rots << std::endl
              << std::endl;

    return outputs_ptr;
}


void FHEContext::test() const {
    using namespace std;
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    auto& cc = pimpl_->cc;
    auto& keys = pimpl_->keys;

    int N = 2048;

    auto gen_plain = [&](int level, double a = -1, double b = 1) {
        vector<complex<double>> v(N);
        auto dist = uniform_real_distribution<double>(a, b);
        for (int i = 0; i < N; ++i) { v[i] = complex<double>(dist(rng), dist(rng)); }
        return make_pair(cc->MakeCKKSPackedPlaintext(v, 1, level), v);
    };

    auto gen_cipher = [&](int level, double a = -1, double b = 1) {
        auto [pt, v] = gen_plain(level, a, b);
        return make_pair(cc->Encrypt(keys.publicKey, pt), v);
    };

    auto ones = [&](int level) {
        vector<double> v(N, 1.0);
        auto pt = cc->MakeCKKSPackedPlaintext(v, 1, level);
        return cc->Encrypt(keys.publicKey, pt);
    };

    auto stats = [&](Ciphertext<DCRTPoly>& c) {
        size_t level = c->GetLevel();
        size_t noise_scale_deg = c->GetNoiseScaleDeg();
        int64_t scaling_factor = c->GetScalingFactor();
        size_t num_primes = c->GetElements()[0].GetNumOfElements();
        size_t chain_len = c->GetElements()[0].GetAllElements().size();

        size_t rem = 30 - c->GetLevel() - (c->GetNoiseScaleDeg() - 1);
        cerr << "level: " << level << ", noise_scale_deg: " << noise_scale_deg << ", scaling_factor: " << scaling_factor
             << ", num_primes: " << num_primes << ", rem: " << rem << endl;
    };

    auto decrypt = [&](Ciphertext<DCRTPoly>& c) {
        Plaintext pt;
        cc->Decrypt(keys.secretKey, c, &pt);
        vector<double> vals = pt->GetRealPackedValue();
        return vals;
    };

    auto cipherprint = [&](Ciphertext<DCRTPoly>& c, int size = 10) {
        stats(c);
        Plaintext pt;
        cc->Decrypt(keys.secretKey, c, &pt);
        vector<double> vals = pt->GetRealPackedValue();
        for (int i = 0; i < size; ++i) { cerr << i << ": " << vals[i] << setprecision(10) << endl; }
    };

    auto silu = [&](double x) -> double {
        return x / (1 + exp(-x));
    };

    int size = 10;

    int depth = this->depth_;
    auto [c, _] = gen_cipher(depth - 1, -1, this->depth_ - 2);

    cipherprint(c, 10);
    //    c = cc->EvalMult(c, p);
    // c = cc->EvalBootstrap(c, 1, 2000);
    c = cc->EvalBootstrap(c);
    //    c = cc->EvalBootstrap(c, 1, 0);
    cipherprint(c, 10);
    return;
    // cerr << "orig:" << endl;
    // for (int i = 0; i < size; ++i) { cerr << i << ": " << setprecision(10) << v[i] << endl; }
    // cipherprint(c);
    // cerr << endl;
    // vector<double> act(size);
    // cerr << "silu truth:" << endl;
    // for (int i = 0; i < size; ++i) { act[i] = silu(v[i]); }
    // for (int i = 0; i < size; ++i) { cerr << i << ": " << setprecision(10) << act[i] << endl; }

    // {
    //     auto t = cc->EvalLogistic(c, -4, 4, 13);
    //     auto el = cc->EvalMult(c, t);

    //     cerr << "eval logistic:" << endl;
    //     cipherprint(el, size);

    //     cerr << "eval logistic diff:" << endl;
    //     auto p = decrypt(el);
    //     for (int i = 0; i < size; ++i) { cerr << i << ": " << setprecision(10) << fabs(p[i] - act[i]) << endl; }
    // }

    // // {
    // //     cerr << "eval chebyshev:" << endl;
    // //     vector<double> coeffs = {1.117453430890045, 1.9993146499511156, 0.9288595192070166, -2.8348506154163985e-16,
    // -0.14564055291152417, 8.093945121246964e-17, 0.032862690405340525, -6.470571301951818e-16, -0.007709018229261363,
    // -1.4952317296618657e-15, 0.0018190821719230033, -1.0978660748260763e-16, -0.00042966633094909814, -5.447204525340349e-16,
    // 0.00010150408266844222, -9.474391770687972e-16, -2.397995822240662e-05,
    // -1.2285920945280185e-15, 5.665204026378215e-06, 1.7104051080514094e-16,
    // -1.3383912161391515e-06, 2.5128512817573504e-17, 3.1619184425945074e-07, -1.6065511747748297e-16, -7.469960058126936e-08,
    // -7.572455763210446e-16, 1.76476069244068e-08, -3.44984900704727e-16,
    // -4.169208653338277e-09, 3.203084277872185e-16, 9.849637114519183e-10, -3.122115682912209e-16,
    // -2.3269827324991317e-10, 4.788313355778748e-16, 5.497209285350798e-11, 2.489821235077242e-17, -1.2990061928700272e-11,
    // -4.77194012213134e-16, 3.0654871954989642e-12, -4.713087841060109e-17,
    // -7.271284744934405e-13, 2.524853601805286e-16, 1.6853016772854863e-13, -4.210786618793355e-16,
    // -4.293942830735163e-14, 2.217150335309485e-16, 6.927973103285438e-15, -2.407659573418734e-16, -5.5737335044655965e-15,
    // -5.878820490044264e-17, -2.347806273946698e-15, -1.0742997567951387e-16, -3.3538326964824363e-15, 1.1678346935201756e-16,
    // -2.9529526638829886e-15, -1.9389196307728387e-16, -3.560304672569212e-15, -3.47546117378402e-16, -2.620045835054193e-15,
    // -1.249312743114138e-16};
    // //     auto cb = cc->EvalChebyshevSeries(c, coeffs, -4, 4);
    // //     cipherprint(cb);
    // // }

    // {
    //     cerr << "eval chebyshev function:" << endl;
    //     auto cb = cc->EvalChebyshevFunction(silu, c, -4, 4, 20);
    //     cipherprint(cb);
    //     cerr << "eval chebyshev function diff:" << endl;
    //     auto p = decrypt(cb);
    //     for (int i = 0; i < size; ++i) { cerr << i << ": " << setprecision(10) << fabs(p[i] - act[i]) << endl; }
    // }
}

} // namespace fheproj
