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
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from triton.runtime import driver

from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
    unroll_num=8,
)

@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def hardsigmoid_backward_func(grad_output, self):
    # hardsigmoid: y = clamp(x/6 + 0.5, 0, 1)
    # gradient: dy/dx = 1/6 when -3 < x < 3, else 0
    # => grad_input = grad_output * (|self| < 3) / 6
    grad_output_fp32 = grad_output.to(tl.float32)
    self_fp32 = self.to(tl.float32)
    # `0.0 if x <= -3, `+inf (->1.0) if x > -3`; and the mirrored bound.
    p = tl.maximum(0.0, (self_fp32 + 3.0) * 1.0e30)
    q = tl.maximum(0.0, (3.0 - self_fp32) * 1.0e30)
    in_range = tl.minimum(1.0, p) * tl.minimum(1.0, q)
    result = grad_output_fp32 * in_range * (1.0 / 6.0)
    return result.to(grad_output.dtype)

# ---------------------------------------------------------------------------
# Small-shape (host/launch-bound) path: a raw @triton.jit kernel driven by the
# flat launcher. On tiny shapes (e.g. [64,64]=4096 elts) the device work is
# negligible; the cost is host-side -- the pointwise_dynamic python wrapper +
# `fn[grid]`'s JITFunction.run bookkeeping (~20us). flat_launcher binds the
# kernel once and replays it through the flat ABI (~5us), so we route only the
# small tier here and leave the already-at-parity large shapes on
# pointwise_dynamic (variant A of flatLaunchSkill).
_SMALL_NUMEL = 65536
_SMALL_BLOCK = 2048
_SMALL_NUM_WARPS = 4

@triton.jit(
    do_not_specialize=["n_elements"],
    do_not_specialize_on_alignment=["grad_ptr", "self_ptr", "out_ptr"],
)
def _hsb_small_kernel(
    grad_ptr,
    self_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # grad_input = grad_output * (|self| < 3) / 6, branch-free via saturating
    # multiply (no tl.where; XPU vselect is the headline pointwise bottleneck).
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    g = tl.load(grad_ptr + offset, mask=mask, other=0).to(tl.float32)
    x = tl.load(self_ptr + offset, mask=mask, other=0).to(tl.float32)
    p = tl.maximum(0.0, (x + 3.0) * 1.0e30)
    q = tl.maximum(0.0, (3.0 - x) * 1.0e30)
    in_range = tl.minimum(1.0, p) * tl.minimum(1.0, q)
    r = g * in_range * (1.0 / 6.0)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)

_MISS = object()
_FLAT = _MISS

def _flat_launchers():
    """`driver.active.flat_launchers`, resolved once (`driver.active` is a lazy
    proxy, so the attribute walk is not free at a launch's ~12us)."""
    global _FLAT
    if _FLAT is _MISS:
        _FLAT = getattr(driver.active, "flat_launchers", None)
    return _FLAT

def _small_eligible(grad_output, self):
    n = grad_output.numel()
    return (
        0 < n <= _SMALL_NUMEL
        and grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and grad_output.dtype in (torch.float16, torch.float32, torch.bfloat16)
    )

def _hsb_small(grad_output, self):
    out = torch.empty_like(grad_output)
    n = grad_output.numel()
    grid = (triton.cdiv(n, _SMALL_BLOCK),)
    launchers = _flat_launchers()
    if launchers is None:  # triton without the launcher cache: correct, just slower
        _hsb_small_kernel[grid](
            grad_output,
            self,
            out,
            n,
            BLOCK_SIZE=_SMALL_BLOCK,
            num_warps=_SMALL_NUM_WARPS,
        )
        return out
    # key carries everything the compiled kernel depends on: dtypes, block,
    # warps and grid (grid participates in compilation on XPU). Pointers are
    # absent -- sound only because the kernel is compiled without alignment
    # specialization (do_not_specialize_on_alignment above).
    key = (out.dtype, _SMALL_BLOCK, _SMALL_NUM_WARPS, grid[0])
    launch, stream = launchers.acquire(_hsb_small_kernel, key)
    if launch is None:  # first call: compile+launch, then bind for replay
        kernel = _hsb_small_kernel[grid](
            grad_output,
            self,
            out,
            n,
            BLOCK_SIZE=_SMALL_BLOCK,
            num_warps=_SMALL_NUM_WARPS,
        )
        launchers.bind(_hsb_small_kernel, key, kernel, grid)
        return out
    # flat ABI: non-constexpr params in signature order (pointers as data_ptr).
    launch(stream, grad_output.data_ptr(), self.data_ptr(), out.data_ptr(), n)
    return out

def hardsigmoid_backward(grad_output, self):
    logger.debug("GEMS_KUNLUNXIN HARDSIGMOID_BACKWARD")
    if _small_eligible(grad_output, self):
        return _hsb_small(grad_output, self)
    return hardsigmoid_backward_func(grad_output, self)
