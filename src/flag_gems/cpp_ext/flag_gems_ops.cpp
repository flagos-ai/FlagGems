#include "flag_gems_ops.h"
#include <iostream>

namespace flag_gems {

BackendType get_default_backend() {
#ifdef USE_FLAGTREE
  return BackendType::kFlagTree;
#else
  return BackendType::kATen;
#endif
}

// CPU fallback实现，仅用于校验，性能低；主性能路径是flagtree编译Triton kernel
void mm_fallback(const TensorMeta& a, const TensorMeta& b, TensorMeta& out) {
  std::cerr << "[WARN] mm use cpp fallback, slow path" << std::endl;
}
void addmm_fallback(
    const TensorMeta& a, const TensorMeta& b, const TensorMeta& c, TensorMeta& out, float alpha, float beta) {
  std::cerr << "[WARN] addmm use cpp fallback" << std::endl;
}
void bmm_fallback(const TensorMeta& a, const TensorMeta& b, TensorMeta& out) {
  std::cerr << "[WARN] bmm use cpp fallback" << std::endl;
}
void conv2d_fallback(const TensorMeta& x, const TensorMeta& w, TensorMeta& out, int stride, int padding) {
  std::cerr << "[WARN] conv2d use cpp fallback" << std::endl;
}
void glu_fallback(const TensorMeta& x, TensorMeta& out) {
  std::cerr << "[WARN] glu use cpp fallback" << std::endl;
}

}  // namespace flag_gems
