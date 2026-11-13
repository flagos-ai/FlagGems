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
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.bf16_fast_store import bf16_fast_store

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# The flagtree compiler shipped in the CI image (flagtree 0.7.0+xpu3.6, an
# older triton fork pinned in backends.yaml) cannot legalize the fused
# triton_xpu.vsigmoidf op that tl.sigmoid(0.0 - x) lowers to
# (ConvertTritonXPUToLLVM marks it illegal), so it fails to compile there. Use
# the explicit 1/(1+exp(x)) = sigmoid(-x) form on flagtree; the newer triton
# build legalizes vsigmoidf and keeps the fused single-op sigmoid.
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)


@triton.jit
def _log_sigmoid_backward_derivative(x_fp32):
    # sigmoid(-x), the log-sigmoid derivative.
    #
    # The XPU backend pattern-matches `1 / (1 + exp(-t))` -- i.e. Triton's
    # `tl.sigmoid(t)` -- and lowers it to the vectorised hardware device call
    # `_ZN3xpu9vsigmoidfEDv16_f`, whereas the mathematically identical
    # `1 / (1 + exp(x))` written with the opposite sign falls back to
    # `llvm.intr.exp2` plus a full fp32 division: 247us vs 393us at 16.7M
    # elements on XPU 5 (IR evidence: vsigmoidf 152 calls vs exp2 88 + divf).
    # So the derivative is computed *as* the sigmoid of -x.
    #
    # Edge cases come out exactly: x = +inf -> 0, x = -inf -> 1, x = -300 -> 1,
    # x = 0 -> 0.5; `tl.sigmoid` does not overflow for any finite input.
    return tl.sigmoid(0.0 - x_fp32)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    # Fallback path (non-contiguous / mixed dtype / non-float tensors): the
    # generic pointwise_dynamic codegen.
    #
    # The derivative is recomputed from `self` instead of consuming the buffer:
    # ATen's forward contract defines buffer == exp(-|self|), but the buffer
    # produced by the vendor log_sigmoid_forward on XPU is uninitialised
    # garbage (measured maxdiff = inf against exp(-|x|)), and the autograd test
    # feeds exactly that full-size garbage buffer.
    self_fp32 = self.to(tl.float32)
    return grad_output * _log_sigmoid_backward_derivative(self_fp32)


@triton.jit
def log_sigmoid_backward_flat_kernel(
    grad_ptr,
    self_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(grad_ptr + offs).to(tl.float32)
    x = tl.load(self_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, g * _log_sigmoid_backward_derivative(x))


@triton.jit
def log_sigmoid_backward_flat_kernel_masked(
    grad_ptr,
    self_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    x = tl.load(self_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, g * _log_sigmoid_backward_derivative(x), mask=mask)


# Per-dtype flat tile for the large regime, tuned on the official benchmark
# matrix (six shapes x fp16/fp32/bf16, all >= 1M elements).  All benchmark
# sizes are powers of two, so the unmasked kernel is what gets measured; the
# masked kernel only covers non-divisible correctness cases.
_TUNED_BLOCK = {
    torch.float16: (16384, 8),
    torch.float32: (32768, 8),
    torch.bfloat16: (16384, 8),
}
# Below 2**21 elements the fp32 tile above would launch only 32 programs; step
# down so small shapes keep >= 64 programs (launch-bound regime).
_SMALL_LIMIT = 1 << 21
_SMALL_BLOCK = 16384
_SMALL_WARPS = 8
# Non-divisible sizes: masked tail path (correctness only, not benchmarked).
_MASK_BLOCK = 16384
_MASK_WARPS = 8

_FLAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _pick_flat_tile(numel, dtype):
    if numel < _SMALL_LIMIT:
        return _SMALL_BLOCK, _SMALL_WARPS
    return _TUNED_BLOCK.get(dtype, (_SMALL_BLOCK, _SMALL_WARPS))


def _can_use_flat(grad_output, self, out):
    return (
        self.dtype in _FLAT_DTYPES
        and grad_output.dtype is self.dtype
        and out.dtype is self.dtype
        and self.numel() == grad_output.numel()
        and self.numel() == out.numel()
        and self.is_contiguous()
        and grad_output.is_contiguous()
        and out.is_contiguous()
    )


def _launch_flat(grad_output, self, out):
    numel = self.numel()
    if numel == 0:
        return out
    block, warps = _pick_flat_tile(numel, self.dtype)
    if numel % block == 0:
        grid = (numel // block,)
        use_mask = False
    else:
        block, warps = _MASK_BLOCK, _MASK_WARPS
        grid = (triton.cdiv(numel, block),)
        use_mask = True

    with bf16_fast_store(out.dtype):
        if use_mask:
            log_sigmoid_backward_flat_kernel_masked[grid](
                grad_output, self, out, numel, BLOCK=block, num_warps=warps
            )
        else:
            log_sigmoid_backward_flat_kernel[grid](
                grad_output, self, out, BLOCK=block, num_warps=warps
            )
    return out


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")

    del buffer
    out = torch.empty_like(self)
    if _can_use_flat(grad_output, self, out):
        return _launch_flat(grad_output, self, out)
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD_OUT")
    del buffer
    if _can_use_flat(grad_output, self, grad_input):
        return _launch_flat(grad_output, self, grad_input)
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
