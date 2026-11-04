#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "flag_gems_ops.h"

namespace py = pybind11;
using namespace flag_gems;

PYBIND11_MODULE(flag_gems_binding, m) {
  py::enum_<BackendType>(m, "BackendType")
      .value("kATen", BackendType::kATen)
      .value("kFlagTree", BackendType::kFlagTree)
      .export_values();

  py::class_<TensorMeta>(m, "TensorMeta")
      .def(py::init<>())
      .def_readwrite("ptr", &TensorMeta::ptr)
      .def_readwrite("ndim", &TensorMeta::ndim);

  m.def("get_default_backend", &get_default_backend);
  m.def("mm_fallback", &mm_fallback);
  m.def("addmm_fallback", &addmm_fallback);
  m.def("bmm_fallback", &bmm_fallback);
  m.def("conv2d_fallback", &conv2d_fallback);
  m.def("glu_fallback", &glu_fallback);
}
