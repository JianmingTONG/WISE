#include "ct_handle.hpp"
using namespace lbcrypto;

namespace fheproj {

struct CiphertextHandle::Impl {
    Ciphertext<DCRTPoly> ct;
};

CiphertextHandle::CiphertextHandle(): p_(new Impl {}) {}
CiphertextHandle::~CiphertextHandle() = default;
CiphertextHandle::CiphertextHandle(CiphertextHandle&&) noexcept = default;
CiphertextHandle& CiphertextHandle::operator=(CiphertextHandle&&) noexcept = default;

Ciphertext<DCRTPoly>& CiphertextHandle::mut() {
    return p_->ct;
}
const Ciphertext<DCRTPoly>& CiphertextHandle::get() const {
    return p_->ct;
}

} // namespace fheproj
