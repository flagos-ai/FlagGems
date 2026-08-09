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

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Kunlunxin/XPU override for aten::log_sigmoid_backward{,.grad_input}.
#
# 1) CORRECTNESS. The vendor's native aten::log_sigmoid_forward returns an
#    *uninitialized* `buffer` (probed on XPU: bf16 buffer came back with inf /
#    4.8125 while exp(-|x|) was 0.617). The generic implementation trusts that
#    buffer whenever it matches shape/dtype/contiguity and the dtype is not
#    fp32, so autograd through log_sigmoid produced garbage fp16/bf16 gradients
#    (tests/test_log_sigmoid_backward.py::test_log_sigmoid_backward_via_autograd
#    [dtype0]/[dtype2] failed). This override never reads `buffer` and always
#    recomputes the derivative from `self`, which is exactly what the CUDA ATen
#    kernel does (there `buffer` is empty by construction).
#
# 2) PERFORMANCE. The generic functional path is `pointwise_dynamic`, and the
#    Kunlunxin CodeGenConfig caps `max_tile_size` at 512, so 16.7M elements are
#    processed by ~32k tiny programs: 84.5 ms vs 0.214 ms for the vendor kernel
#    (0.003x). The flat kernel below plus the XPU launch knobs measured in
#    /tmp/lsb_probe (BLOCK=32768, num_warps=8, unroll_num=2,
#    buffer_size_limit=8192, unmasked contiguous DMA) brings the same case to
#    0.393 ms (~215x faster kernel, ~0.54x of the vendor kernel).
#
# Probe notes (XPU 1, 2026-08-29, /tmp/lsb_probe/probe_*.json), 16.7M fp16:
#   * cost split at the chosen tile: 2 loads + 1 store floor 0.085 ms,
#     +tl.exp 0.232 ms, +fp32 reciprocal 0.393 ms - exp and the divide each
#     cost ~9 ns/element and add up, so the vendor kernel (0.214 ms, close to
#     the pure-copy floor) stays ahead: the remaining gap is a backend
#     transcendental/divide floor, not a structural problem of this kernel.
#   * `tl.where(x < 0, 1, exp(-|x|)) / (1 + exp(-|x|))` (the generic algebra)
#     costs 0.711 ms vs 0.393 ms for `1 / (1 + exp(x))`: the extra abs+where
#     is ~17 ns/element on XPU. `tl.sigmoid` 0.402 ms, `tl.fdiv` (ieee True or
#     False) 0.431 ms, `tl.exp2`-based 0.724 ms *and* numerically wrong
#     (maxdiff 0.42), libdevice tanh fails to link, and a bitcast reciprocal
#     seed + 3 Newton steps is 2.374 ms (bit ops get scalarized, same dead end
#     the log_sigmoid_forward closure recorded).
#   * special values keep vendor semantics without any branch:
#     x=-inf -> exp=0 -> d=1, x=+inf -> exp=inf -> d=0, x=-300 -> d=1.
#
# 3) BACKEND TRAP (new finding, /tmp/lsb_probe/probe_mask_v8.py). A masked load
#    written as `tl.load(p + off, mask=mask, other=0.0)` silently corrupts a
#    few percent of the *valid* lanes on TritonXPU when the vendor guards
#    TRITONXPU_OTHER_SIM / TRITONXPU_STORE_MASK_SIM are not exported - and it
#    does so even when the mask is entirely true. Measured (fp16, BLOCK=2048):
#    27184/1048576 wrong for [1024,1024], 137586/5924352 for [16,7,57,32,29],
#    40/1517 for [37,41] (wrong lanes read as 0, maxerr 3.5). fp32 corrupts at
#    BLOCK=1024 instead (54977/1048576). Dropping `other=` makes every shape /
#    dtype / block size in the sweep exact, so the masked kernel below relies
#    on the masked store to discard the tail lanes and never passes `other=`.

UNROLL_NUM = 2
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Keep the number of compiled variants small: two unmasked tiles for the
    # divisible (benchmark / large tensor) cases and two masked fallbacks.
    if n_elements >= 32768 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


@triton.jit
def log_sigmoid_backward_flat_kernel(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # NEVER pass `other=` here: on TritonXPU a masked load with an explicit
    # `other` value silently corrupts a few percent of the *valid* lanes (see
    # the module docstring), and the masked store already discards the
    # out-of-range lanes.
    g = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(self_ptr + offsets, mask=mask)
    derivative = 1.0 / (1.0 + tl.exp(x.to(tl.float32)))
    res = g.to(tl.float32) * derivative
    tl.store(
        grad_input_ptr + offsets,
        res.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def log_sigmoid_backward_flat_kernel_unmasked(
    grad_output_ptr,
    self_ptr,
    grad_input_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g = tl.load(grad_output_ptr + offsets)
    x = tl.load(self_ptr + offsets)
    derivative = 1.0 / (1.0 + tl.exp(x.to(tl.float32)))
    res = g.to(tl.float32) * derivative
    tl.store(grad_input_ptr + offsets, res.to(grad_input_ptr.dtype.element_ty))


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_pointwise_kernel(grad_output, self):
    # Strided / broadcast / mixed-dtype fallback with the same algebra as the
    # flat kernel above (still a Triton kernel, no ATen redispatch).
    derivative = 1.0 / (1.0 + tl.exp(self.to(tl.float32)))
    return grad_output * derivative


def _can_use_flat_kernel(grad_output, self, grad_input=None):
    return (
        grad_output.shape == self.shape
        and grad_output.dtype == self.dtype
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and (
            grad_input is None
            or (
                grad_input.shape == self.shape
                and grad_input.dtype == self.dtype
                and grad_input.is_contiguous()
            )
        )
    )


def _launch_flat_kernel(grad_output, self, grad_input):
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        log_sigmoid_backward_flat_kernel[grid](
            grad_output,
            self,
            grad_input,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        log_sigmoid_backward_flat_kernel_unmasked[grid](
            grad_output,
            self,
            grad_input,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    return grad_input


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")
    # `buffer` is intentionally unused: the vendor forward leaves it
    # uninitialized (see the module docstring above).
    if _can_use_flat_kernel(grad_output, self):
        return _launch_flat_kernel(grad_output, self, torch.empty_like(self))
    return log_sigmoid_backward_pointwise_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD OUT")
    if _can_use_flat_kernel(grad_output, self, grad_input):
        return _launch_flat_kernel(grad_output, self, grad_input)
    return log_sigmoid_backward_pointwise_kernel(grad_output, self, out0=grad_input)
