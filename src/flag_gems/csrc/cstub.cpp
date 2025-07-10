#include <pybind11/pybind11.h>
#include "torch/python.h"

#include "flag_gems/operators.h"

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(c_operators, m) {
  m.def("sum_dim", &flag_gems::sum_dim);
  m.def("add_tensor", &flag_gems::add_tensor);
  m.def("rms_norm", &flag_gems::rms_norm);
  m.def("fused_add_rms_norm", &flag_gems::fused_add_rms_norm);
  m.def("nonzero", &flag_gems::nonzero);
  // Rotary embedding
  m.def("rotary_embedding", &flag_gems::rotary_embedding);
  m.def("rotary_embedding_inplace", &flag_gems::rotary_embedding_inplace);
  m.def("bmm", &flag_gems::bmm);
}

namespace flag_gems {
TORCH_LIBRARY(flag_gems, m) {
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
  m.def("nonzero(Tensor self) -> Tensor");
  // rotary_embedding
  m.def(
      "rotary_embedding_inplace(Tensor! q, Tensor! k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> ()");
  m.def(
      "rotary_embedding(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> (Tensor, Tensor)");  // q and k may be view to other size
  m.def("cat(Tensor[] tensors, int dim=0) -> Tensor");
  m.def("bmm(Tensor self, Tensor mat2) -> Tensor");
}

TORCH_LIBRARY_IMPL(flag_gems, CUDA, m) {
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("add_tensor", TORCH_FN(add_tensor));
  // Norm
  m.impl("rms_norm", TORCH_FN(rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
  m.impl("nonzero", TORCH_FN(nonzero));
  // Rotary embedding
  m.impl("rotary_embedding", TORCH_FN(rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(rotary_embedding_inplace));
  m.impl("cat", TORCH_FN(cat));
  m.impl("bmm", TORCH_FN(bmm));
}
}  // namespace flag_gems
