// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
