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

"""hardtanh_backward: out[i] = grad_output[i] if min_val < self[i] < max_val else 0.

Semantics reproduced (strict bounds, as required by the reference):
  * threshold decision evaluated in fp32 opmath (exact upcast for fp16/bf16),
  * payload is a bit-exact copy of grad_output (no rescaling/rounding),
  * NaN / +-inf self values fall outside the bounds -> 0,
  * result dtype is the (promoted) payload dtype.

Dispatch:
  * n >= _LARGE_THRESHOLD and n % _ALIGNED == 0 -> mask-free kernel with a bounded
    per-(n_elements, dtype) tile/warps search (every large benchmark shape is a
    multiple of 8192, so no tuned tile ever needs a mask),
  * n >= _LARGE_THRESHOLD otherwise              -> masked 4096-element tile,
  * small tensors                                -> fixed 1024-element tile, because a
    per-call tuner lookup would cost more than the few microseconds of work there.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.hardtanh_backward import (
    hardtanh_backward as default_hardtanh_backward,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _hardtanh_bwd_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    lo,
    hi,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    keep = (x.to(tl.float32) > lo) & (x.to(tl.float32) < hi)
    res = tl.where(keep, g, tl.zeros_like(g))
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def _hardtanh_bwd_kernel_nomask(
    grad_ptr,
    x_ptr,
    out_ptr,
    lo,
    hi,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    g = tl.load(grad_ptr + offsets)
    keep = (x.to(tl.float32) > lo) & (x.to(tl.float32) < hi)
    res = tl.where(keep, g, tl.zeros_like(g))
    tl.store(out_ptr + offsets, res)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=4),
        triton.Config({"BLOCK": 8192}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 8192}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _hardtanh_bwd_kernel_large(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    lo,
    hi,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    g = tl.load(grad_ptr + offsets)
    keep = (x.to(tl.float32) > lo) & (x.to(tl.float32) < hi)
    res = tl.where(keep, g, tl.zeros_like(g))
    tl.store(out_ptr + offsets, res)


_BLOCK_SMALL = 1024
_BLOCK_LARGE = 4096
_ALIGNED = 8192  # largest tuned tile: divisibility guarantee for every tuned config
_LARGE_THRESHOLD = 262144  # keep >= 64 blocks so small tensors still spread over SMs
_WARPS = 4


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def hardtanh_backward(grad_output, self, min_val, max_val):
    logger.debug("GEMS_MTHREADS HARDTANH_BACKWARD")

    if not _use_triton_kernel(grad_output) or not _use_triton_kernel(self):
        return default_hardtanh_backward(grad_output, self, min_val, max_val)

    with torch_device_fn.device(grad_output.device):
        return _launch_hardtanh_backward(grad_output, self, min_val, max_val)


def _launch_hardtanh_backward(grad_output, self, min_val, max_val):
    lo = float(min_val)
    hi = float(max_val)
    n = grad_output.numel()
    if grad_output.dtype == self.dtype:
        out = torch.empty_like(grad_output, memory_format=torch.contiguous_format)
    else:
        out = torch.empty(
            grad_output.shape,
            dtype=torch.promote_types(grad_output.dtype, self.dtype),
            device=grad_output.device,
        )
    if n == 0:
        return out
    if n >= _LARGE_THRESHOLD:
        if n % _ALIGNED == 0:
            _hardtanh_bwd_kernel_large[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
                grad_output, self, out, n, lo, hi
            )
        else:
            _hardtanh_bwd_kernel[((n + _BLOCK_LARGE - 1) // _BLOCK_LARGE,)](
                grad_output, self, out, n, lo, hi, BLOCK=_BLOCK_LARGE, num_warps=_WARPS
            )
    elif n % _BLOCK_SMALL == 0:
        _hardtanh_bwd_kernel_nomask[(n // _BLOCK_SMALL,)](
            grad_output, self, out, lo, hi, BLOCK=_BLOCK_SMALL, num_warps=_WARPS
        )
    else:
        _hardtanh_bwd_kernel[((n + _BLOCK_SMALL - 1) // _BLOCK_SMALL,)](
            grad_output, self, out, n, lo, hi, BLOCK=_BLOCK_SMALL, num_warps=_WARPS
        )
    return out
