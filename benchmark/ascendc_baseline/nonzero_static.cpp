#include <ATen/Functions.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>

namespace {

at::Tensor nonzero_static(
    const at::Tensor& input,
    int64_t size,
    int64_t fill_value) {
  TORCH_CHECK(size >= 0, "nonzero_static: size must be non-negative");

  auto out = at::full(
      {size, input.dim()}, fill_value, input.options().dtype(at::kLong));
  if (size == 0 || input.dim() == 0) {
    return out;
  }

  auto indices = at::nonzero(input);
  auto copy_len = std::min<int64_t>(size, indices.size(0));
  if (copy_len > 0) {
    out.narrow(0, 0, copy_len)
        .copy_(indices.narrow(0, 0, copy_len));
  }
  return out;
}

}

TORCH_LIBRARY(flag_gems_ascendc, m) {
  m.def(
      "nonzero_static(Tensor input, int size, int fill_value=-1) -> Tensor");
}

TORCH_LIBRARY_IMPL(
    flag_gems_ascendc,
    CompositeExplicitAutograd,
    m) {
  m.impl("nonzero_static", TORCH_FN(nonzero_static));
}
