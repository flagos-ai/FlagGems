#pragma once

#include "flag_gems/backend_utils.h"

#if defined(FLAGGEMS_USE_GCU)
#include <pybind11/embed.h>
#endif

namespace flag_gems::test {

#if defined(FLAGGEMS_USE_GCU)
namespace detail {

inline int gcu_init_backend() {
  // Intentionally leaked — avoids segfault from static destruction order
  // conflicts between pybind11 interpreter and PyTorch statics on exit.
  new pybind11::scoped_interpreter();
  pybind11::module_::import("torch");
  pybind11::module_::import("torch_gcu");
  return 0;
}

static int gcu_init_ = gcu_init_backend();

}  // namespace detail
#endif

// Convenience aliases — delegate to backend_utils.h
inline torch::Device default_device(int index = 0) {
  return flag_gems::backend::getDefaultDevice(index);
}

inline bool is_device_available() {
  return flag_gems::backend::isDeviceAvailable();
}

inline void synchronize() {
  flag_gems::backend::synchronize();
}

}  // namespace flag_gems::test
