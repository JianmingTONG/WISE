#pragma once
#include "openfhe.h"

#include <memory>

namespace fheproj {

class FHEContext;

class CiphertextHandle {
    public:
    CiphertextHandle();
    ~CiphertextHandle();
    CiphertextHandle(const CiphertextHandle&) = delete;
    CiphertextHandle& operator=(const CiphertextHandle&) = delete;
    CiphertextHandle(CiphertextHandle&&) noexcept;
    CiphertextHandle& operator=(CiphertextHandle&&) noexcept;

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& mut();
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& get() const;

    private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    friend class FHEContext;
};

} // namespace fheproj
