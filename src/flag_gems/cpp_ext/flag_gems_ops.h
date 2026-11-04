#pragma once
#include <cstdint>
#include <vector>

namespace flag_gems {

enum class BackendType { kATen, kFlagTree };

// 基础设备内存描述符
struct TensorMeta {
  void* ptr;
  int64_t dims[4];
  int64_t strides[4];
  int dtype;
  int ndim;
};

// C++侧算子声明（fallback / 辅助接口，真正高性能计算走flagtree编译Triton）
void mm_fallback(const TensorMeta& a, const TensorMeta& b, TensorMeta& out);
void addmm_fallback(
    const TensorMeta& a, const TensorMeta& b, const TensorMeta& c, TensorMeta& out, float alpha, float beta);
void bmm_fallback(const TensorMeta& a, const TensorMeta& b, TensorMeta& out);
void conv2d_fallback(const TensorMeta& x, const TensorMeta& w, TensorMeta& out, int stride, int padding);
void glu_fallback(const TensorMeta& x, TensorMeta& out);

BackendType get_default_backend();

}  // namespace flag_gems
