#include "aten_patch.h"
#include <pybind11/pybind11.h>
#include "flag_gems/operators.h"
#include "torch/python.h"

std::vector<std::string> registered_ops;

std::vector<std::string> get_registered_ops() {
  return registered_ops;
}

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(aten_patch, m) {
  m.def("get_registered_ops", &get_registered_ops);
}

// NOTE: The custom operator registration below uses TORCH_LIBRARY_IMPL,
// which executes immediately at module import time.
// As a result, it is not currently possible to register ops conditionally,
// e.g., based on a user-defined disabled op list.
// If per-operator control is desired in the future,
// this part should be refactored to delay registration until `init()`
// or use a dynamic dispatch approach.
//
// Contributions are welcome to improve this behavior!
namespace flag_gems {
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  // REGISTER_AND_LOG("addmm", addmm);
  // REGISTER_AND_LOG("addmm.out", addmm_out);
  // REGISTER_AND_LOG("bmm", bmm);
  // REGISTER_AND_LOG("mm", mm_tensor);
  // REGISTER_AND_LOG("mm.out", mm_out_tensor);
  REGISTER_AND_LOG("max.dim_max", max_dim_max);
  REGISTER_AND_LOG("max.dim", max_dim);
  REGISTER_AND_LOG("max", max);
  REGISTER_AND_LOG("sum", sum);
  REGISTER_AND_LOG("zeros", zeros);
  REGISTER_AND_LOG("fill.Scalar", fill_scalar);
  REGISTER_AND_LOG("fill_.Scalar", fill_scalar_);
}

}  // namespace flag_gems
