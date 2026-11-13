# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gcu400 masked_select_backward: native fallback for fp64 / complex128.

The generated Triton kernel does not compile for float64 or complex128 on this
backend ("Pipeline run failed: PassManager execution failed").  Both dtypes
have a native composite implementation that is exact, so route them there;
complex64 and the real dtypes stay on the Triton path.
"""

import torch

import flag_gems.ops.masked_select_backward as _generic_module
from flag_gems.ops.masked_select_backward import (
    masked_select_backward as _generic_masked_select_backward,
)
from flag_gems.runtime.backend._enflame.gcu400.ops.copy import copy_ as _gcu_copy_

# The generic implementation restores non-contiguous results through the shared
# copy_, whose strided store mis-addresses the tail on this backend (transposed
# and channels_last destinations).  Point its module global at the gcu400 copy_
# so the Triton path keeps working without editing the shared op.
_generic_module.copy_ = _gcu_copy_

__all__ = ["masked_select_backward"]

_NATIVE_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd)
_NATIVE_DTYPES = (torch.float64, torch.complex128)


def masked_select_backward(grad, input, mask):
    if (
        grad.dtype in _NATIVE_DTYPES
        # Non-contiguous inputs take the op's broadcast/as_strided route, whose
        # mask materialisation goes through the shared contiguous/copy kernels;
        # those mis-address a transposed / channels_last destination on this
        # backend.  The native composite handles it exactly.
        or not input.is_contiguous()
        or not mask.is_contiguous()
    ):
        return torch.ops.aten.masked_select_backward.default.redispatch(
            _NATIVE_KEYSET, grad, input, mask
        )
    return _generic_masked_select_backward(grad, input, mask)
