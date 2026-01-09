#pragma once
#include "ct_handle.hpp"
#include "math/hal/basicint.h"
#include "utils.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace fheproj {

struct LinearTransformReady;
struct WinogradReady;
struct HerPNReady;

class FHEContext {
    public:
    struct Params {
        uint32_t log_N = 0;
        uint32_t scale_mod_size = 0;
        uint32_t first_mod_size = 0;
        uint32_t mult_depth = 0;
        uint32_t log_batch_size = 0;
        std::string security_level = "";
        std::string secret_key_dist = "";
        std::string scaling_technique = "";
        std::string key_switch_technique = "";
        std::vector<uint32_t> level_budget = {};
        std::vector<int32_t> global_rots = {};
        uint32_t composite_degree = 1;
    };

    explicit FHEContext(const Params&);
    ~FHEContext();
    FHEContext(const FHEContext&) = delete;
    FHEContext& operator=(const FHEContext&) = delete;
    FHEContext(FHEContext&&) noexcept;
    FHEContext& operator=(FHEContext&&) noexcept;

    usint depth() const noexcept {
        return depth_;
    }

    std::string encode_plaintext_bytes(const std::vector<double>& values) const;

    std::shared_ptr<CiphertextHandle> encrypt(const std::vector<double>&, uint32_t level) const;

    std::vector<std::shared_ptr<CiphertextHandle>> encrypt_batch(const std::vector<std::vector<double>>&, uint32_t level) const;

    std::vector<double> decrypt(const std::shared_ptr<CiphertextHandle>& h, size_t length = 0) const;

    std::vector<std::vector<double>> decrypt_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& hs,
                                                   size_t length = 0) const;

    std::shared_ptr<CiphertextHandle> eval_add(const std::shared_ptr<CiphertextHandle>& a,
                                               const std::shared_ptr<CiphertextHandle>& b) const;

    std::vector<std::shared_ptr<CiphertextHandle>> eval_add_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& a,
                                                                  const std::vector<std::shared_ptr<CiphertextHandle>>& b) const;

    std::shared_ptr<CiphertextHandle> level_reduce(const std::shared_ptr<CiphertextHandle>&, size_t) const;

    void level_reduce_in_place(const std::shared_ptr<CiphertextHandle>&, size_t) const;

    std::vector<std::shared_ptr<CiphertextHandle>> level_reduce_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& as,
                                                                      size_t level) const;

    void level_reduce_in_place_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& as, size_t level) const;

    void eval_add_in_place(const std::shared_ptr<CiphertextHandle>& a, const std::shared_ptr<CiphertextHandle>& b) const;

    void eval_add_in_place_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& as,
                                 const std::vector<std::shared_ptr<CiphertextHandle>>& bs) const;

    std::tuple<size_t, size_t, int128_t, size_t, size_t> stats(const std::shared_ptr<CiphertextHandle>& h) const;

    std::vector<std::shared_ptr<CiphertextHandle>> eval_linear_transform(const std::vector<std::shared_ptr<CiphertextHandle>>& x,
                                                                         const LinearTransformReady& lt,
                                                                         int target_levels_remaining) const;

    std::vector<std::shared_ptr<CiphertextHandle>> eval_winograd(const std::vector<std::shared_ptr<CiphertextHandle>>& x,
                                                                 const WinogradReady& wino,
                                                                 int target_levels_remaining) const;

    std::vector<std::shared_ptr<CiphertextHandle>> eval_winograd_advance(const std::vector<std::shared_ptr<CiphertextHandle>>& x,
                                                                         const WinogradReady& wino,
                                                                         int target_levels_remaining) const;

    std::vector<std::shared_ptr<CiphertextHandle>>
    eval_herpn(const std::vector<std::shared_ptr<CiphertextHandle>>& x, const HerPNReady& hp, int target_levels_remaining) const;

    std::vector<std::shared_ptr<CiphertextHandle>> eval_nonlinear(const std::vector<std::shared_ptr<CiphertextHandle>>& x,
                                                                  const NonlinearReady& nonlinear,
                                                                  int target_levels_remaining) const;

    std::vector<std::shared_ptr<CiphertextHandle>>
    eval_bootstrap_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& x) const;

    std::vector<std::shared_ptr<CiphertextHandle>>
    eval_silu_batch(const std::vector<std::shared_ptr<CiphertextHandle>>& x, double a, double b, uint32_t degree) const;

    std::vector<std::shared_ptr<CiphertextHandle>>
    deepcopy_ciphertexts(const std::vector<std::shared_ptr<CiphertextHandle>>& x) const;

    void test() const;

    uint32_t slots() const {
        return slots_;
    }

    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    uint32_t slots_ = 0;
    usint depth_ = 0;
    uint32_t composite_degree_ = 1;
};

} // namespace fheproj
