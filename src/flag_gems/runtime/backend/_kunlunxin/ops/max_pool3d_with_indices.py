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

import logging

import torch

logger = logging.getLogger(__name__)

# Capture the vendor kernels before FlagGems installs its CUDA registrations.
# Calling the ATen operators by name from these wrappers would recurse.
_native_forward = torch.library.get_kernel("aten::max_pool3d_with_indices", "CUDA")
_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(value)


def _requires_cpu_fallback(dilation, ceil_mode):
    return ceil_mode or any(value != 1 for value in _triple(dilation))


def max_pool3d_with_indices(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    logger.debug("GEMS KUNLUNXIN MAX_POOL3D_WITH_INDICES")
    if _requires_cpu_fallback(dilation, ceil_mode):
        # XDNN rejects ceil_mode and non-unit dilation. The generic Triton
        # kernel crashes the current P800 compiler even with a 1-D small tile,
        # so preserve full ATen semantics through the CPU implementation.
        output, indices = torch.ops.aten.max_pool3d_with_indices(
            input.cpu(),
            kernel_size,
            [] if stride is None else stride,
            padding,
            dilation,
            ceil_mode,
        )
        return output.to(input.device), indices.to(input.device)
    return _native_forward.call_boxed(
        _CUDA_KEYSET,
        input,
        kernel_size,
        [] if stride is None else stride,
        padding,
        dilation,
        ceil_mode,
    )
