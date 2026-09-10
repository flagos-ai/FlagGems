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

#if defined(FLAGGEMS_USE_GCU) || defined(FLAGGEMS_USE_MLU)
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

#if defined(FLAGGEMS_USE_MLU)
namespace detail {

  inline int mlu_init_backend() {
    // Intentionally leaked — avoids segfault from static destruction order
    // conflicts between pybind11 interpreter and PyTorch statics on exit.
    new pybind11::scoped_interpreter();
    // torch_mlu registers the PrivateUse1 device module only when torch reports
    // no active accelerator. Linking libtorch_mlu.so into this process already
    // makes torch.accelerator report "mlu", so the registration branch would be
    // skipped ("Autoload skipped due to environment mismatch"). Mask the query
    // for the duration of the import, then restore it.
    pybind11::exec(R"PY(
import sys
import torch

if not hasattr(torch, "mlu"):
    _orig_current_accelerator = getattr(torch.accelerator, "current_accelerator", None)
    if _orig_current_accelerator is not None:
        torch.accelerator.current_accelerator = lambda *args, **kwargs: None
    try:
        # Drop the whole package: `import torch` may already have auto-loaded
        # torch_mlu (with the registration branch skipped). Popping only the
        # top-level module would leave torch_mlu.* submodules cached, and the
        # re-import would then die with "partially initialized module
        # 'torch_mlu' has no attribute 'mlu'".
        for _name in [n for n in sys.modules if n == "torch_mlu" or n.startswith("torch_mlu.")]:
            del sys.modules[_name]
        import torch_mlu  # noqa: F401
    finally:
        if _orig_current_accelerator is not None:
            torch.accelerator.current_accelerator = _orig_current_accelerator

# MLU matmul defaults to TF32-style reduced precision (fp32 mm is off by ~3e-3),
# which makes the on-device torch reference too coarse for the fp32 tolerances
# used by gems_assert_close. FlagGems' own kernels already pass
# allow_tf32=False, so pin the reference to true fp32 as well.
try:
    from torch_mlu.backends import mlu as _mlu_backends

    _mlu_backends.matmul.allow_tf32 = False
except Exception:  # pragma: no cover - older torch_mlu without the knob
    pass
)PY");
    return 0;
  }

  static int mlu_init_ = mlu_init_backend();

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
