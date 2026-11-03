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
import triton

import flag_gems

from .i0 import _BLOCK, _i0_poly, i0_kernel, i0_kernel_nomask  # noqa: F401

logger = logging.getLogger(__name__)

_SMALL_BLOCK = 1024
_SMALL_LIMIT = 65536


def i0_(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN I0_")
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        # Try common keyword names
        for k in ("input", "self", "x"):
            if k in kwargs:
                x = kwargs[k]
                break
    if x is None:
        raise ValueError(
            "i0_ expects a tensor as the first positional argument or in keyword 'input'/'self'/'x'."
        )

    if x.device.type != flag_gems.device:
        raise AssertionError(f"Input tensor must be on a {flag_gems.device} device.")
    if not x.is_contiguous():
        raise AssertionError("Input tensor must be contiguous.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise AssertionError(
            "Unsupported dtype for i0_. Supported: float16, bfloat16, float32, float64."
        )

    n_elements = x.numel()
    if n_elements == 0:
        return x

    if n_elements < _SMALL_LIMIT:
        # Small tensors: 1024-lane masked tile (see NOTE above).
        grid = (triton.cdiv(n_elements, _SMALL_BLOCK),)
        i0_kernel[grid](x, x, n_elements, BLOCK_SIZE=_SMALL_BLOCK)
    elif n_elements % _BLOCK == 0:
        grid = (n_elements // _BLOCK,)
        i0_kernel_nomask[grid](x, x, n_elements, BLOCK_SIZE=_BLOCK)
    else:
        grid = (triton.cdiv(n_elements, _BLOCK),)
        i0_kernel[grid](x, x, n_elements, BLOCK_SIZE=_BLOCK)
    return x
