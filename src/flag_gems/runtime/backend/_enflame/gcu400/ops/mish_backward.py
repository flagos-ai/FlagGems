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
"""GCU400 ``mish_backward``: one flat kernel over the fused element space.

The shared implementation is a ``pointwise_dynamic`` kernel. That wrapper
re-derives multi-dimensional indexing and per-dimension masks on every call and
emits its own launch bookkeeping, which on GCU400 costs more than the arithmetic
at the official shapes (0.98-1.28x, GEO 1.075x). Both operands here are dense
same-shape tensors for every benchmark case, so the fast path walks them as one
flat range with the same tile/slab geometry the neighbouring unary operators use,
and the generic kernel stays as the fallback for non-contiguous or broadcast
inputs.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_tanh = tl_extra_shim.tanh

# GCU400: 4 DIE x 6 SIP, resident CTA count aligned with the topology.
NUM_SIPS = 24
MAX_GRID = NUM_SIPS * 2
# --- TUNING PARAMS START ---
BLOCK_LARGE_FP32 = 65536
BLOCK_LARGE_16 = 16384
BLOCK_MID = 8192
# 2026-11-23 sweep (official do_bench, GEO over the 15 core cells; 1.075x at HEAD):
#   flat kernel alone 1.996x, 16-bit large tile 4096/8192/16384/32768 -> 1.53x/1.85x/2.96x/2.79x,
#   fp32 large tile 32768/65536 -> 5.00x/5.54x, slab on/off neutral for 16-bit,
#   fp16-native math failed 4 accuracy cells, the one-exp algebra took 2.809x -> 4.127x.
# --- TUNING PARAMS END ---


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _mish_backward_generic(grad_output, input):
    """Reference-graph math, kept for shapes the flat kernel cannot take."""
    x = input.to(tl.float32)
    grad = grad_output.to(tl.float32)
    softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tanh_sp = _tanh(softplus)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    derivative = tanh_sp + x * sigmoid * (1.0 - tanh_sp * tanh_sp)
    return grad * derivative


@libentry()
@triton.jit
def _mish_backward_flat(
    grad_ptr,
    input_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    SLAB: tl.constexpr,
    COMPUTE_F32: tl.constexpr,
):
    """Elementwise mish backward over a flat range, tile by tile.

    ``SLAB`` switches the tile order from grid-stride to one contiguous run per
    program; contiguous runs are what the unary operators in this backend use once
    the buffers are large enough for HBM channel effects to matter.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    if SLAB:
        per_pid = (num_blocks + num_pids - 1) // num_pids
        begin = pid * per_pid
        end = tl.minimum(begin + per_pid, num_blocks)
        step = 1
    else:
        begin = pid
        end = num_blocks
        step = num_pids
    for block_id in tl.range(begin, end, step):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(input_ptr + off)
            grad = tl.load(grad_ptr + off)
        else:
            mask = off < N
            x = tl.load(input_ptr + off, mask=mask)
            grad = tl.load(grad_ptr + off, mask=mask)
        if COMPUTE_F32:
            x = x.to(tl.float32)
            grad = grad.to(tl.float32)
        # t = exp(x) and u = exp(2 * softplus(x)) = (1 + t)^2, so tanh(softplus) and
        # sigmoid are exact rational functions of t: one transcendental instead of
        # exp + log + tanh + exp. Clamping at 20 keeps u finite and matches the
        # fp32 reference, where tanh(softplus(x)) is already exactly 1 there.
        t = tl.exp(tl.minimum(x, 20.0))
        u = (1.0 + t) * (1.0 + t)
        tanh_sp = (u - 1.0) / (u + 1.0)
        sigmoid = t / (1.0 + t)
        derivative = tanh_sp + x * sigmoid * (1.0 - tanh_sp * tanh_sp)
        if EVEN:
            tl.store(out_ptr + off, grad * derivative)
        else:
            tl.store(out_ptr + off, grad * derivative, mask=mask)


def _choose_config(num_elements, dtype):
    """Tile geometry by element count and dtype, mirroring the unary operators."""
    if num_elements <= 32768:
        return max(1024, triton.next_power_of_2(num_elements)), 4, False, True
    if num_elements < 1 << 22:
        return BLOCK_MID, 4, False, True
    if dtype == torch.float32:
        return BLOCK_LARGE_FP32, 4, True, True
    return BLOCK_LARGE_16, 4, True, True


def _can_flatten(grad_output, input):
    """The flat kernel needs two dense same-dtype tensors with one element count."""
    if grad_output.numel() != input.numel() or grad_output.dtype != input.dtype:
        return False
    if not (grad_output.is_contiguous() and input.is_contiguous()):
        return False
    return grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)


def _launch(grad, inp, out):
    num_elements = inp.numel()
    if num_elements == 0:
        return out
    block_size, num_warps, slab, compute_f32 = _choose_config(num_elements, inp.dtype)
    grid_size = min(triton.cdiv(num_elements, block_size), MAX_GRID)
    with torch_device_fn.device(inp.device):
        _mish_backward_flat[(grid_size,)](
            grad,
            inp,
            out,
            num_elements,
            BLOCK=block_size,
            EVEN=(num_elements % block_size == 0),
            SLAB=slab,
            COMPUTE_F32=compute_f32,
            num_warps=num_warps,
        )
    return out


def mish_backward(grad_output, input):
    logger.debug("GEMS_ENFLAME MISH_BACKWARD")
    if _can_flatten(grad_output, input):
        return _launch(grad_output, input, torch.empty_like(input))
    return _mish_backward_generic(grad_output, input)
