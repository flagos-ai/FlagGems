#include <pybind11/pybind11.h>
#include "torch/python.h"

#include "flag_gems/operators.h"

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(aten_patch, m) {
}

namespace flag_gems {

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  // blas
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("bmm", TORCH_FN(bmm));
  m.impl("mm", TORCH_FN(mm_tensor));
}
}  // namespace flag_gems
