#include <pybind11/pybind11.h>
#include "flag_gems/operators.h"

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(c_operators, m) {
}

namespace flag_gems {
TORCH_LIBRARY(flag_gems, m) {
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
}

TORCH_LIBRARY_IMPL(flag_gems, CUDA, m) {
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("add_tensor", TORCH_FN(add_tensor));
  // Norm
  m.impl("rms_norm", TORCH_FN(rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
}
}  // namespace flag_gems
