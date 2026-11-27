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

"""gcu400 specialization of silu_and_mul.

The forward/mul path runs as a flat linear-index kernel instead of
@pointwise_dynamic: the codegen kernel spends a large share of its time on
per-element index decomposition plus tile bookkeeping, which caps it at
~1.58 TB/s on 2-byte tensors where a flat kernel with the same math reaches
~1.8 TB/s (fp16 1G elements: 4.075ms -> 1.820ms).  The generic codegen kernel
is kept for the cases the flat kernel does not cover (broadcast, mixed dtypes,
non-contiguous operands, float64) and for the autograd path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from ...gcu400.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Flat-kernel tiles, measured on gcu400 with the persistent 48-program loop
# (min of repeated do_bench runs, contended device):
#   2-byte lanes, 16.7M: 8192 -> 0.0497ms (2026 GB/s), 16384 -> 0.0592ms
#   2-byte lanes, 536M : 8192 -> 1.8197ms (1770 GB/s), 16384 -> 2.0707ms
#   4-byte lanes, 16.7M: 65536 -> 0.0806ms (2498 GB/s), 8192 -> 0.0831ms
#   4-byte lanes, 536M : 65536 -> 2.6653ms (2417 GB/s), 8192 -> 2.8093ms
#   2-byte lanes, 4096 elems: 1024 -> 3.34us, 4096 -> 3.90us, 8192 -> 4.22us
#   2-byte lanes, 65536 elems: 4096 -> 4.30us, 8192 -> 4.42us
#   2-byte lanes, 1M elems: 8192 -> 7.59us, 4096 -> 8.70us
# Small inputs need a proportional tile (the mask cost of a huge tile lands on
# one program); large 4-byte inputs prefer the widest tile.
_SILU_TILE_LEVELS = (
    (1024, 256),
    (4096, 1024),
    (32768, 2048),
    (262144, 4096),
    (1 << 22, 8192),
)
_SILU_BLOCK_16BIT = 8192
_SILU_BLOCK_WIDE = 65536
_SILU_MAX_GRID = 48
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def silu_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    sig = 1 / (1 + tl.exp(-x_fp32))
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu
    return dx, dy


@libentry()
@triton.jit
def _silu_and_mul_flat_kernel(X, Y, Out, N, BLOCK: tl.constexpr):
    """Persistent flat kernel: every program strides BLOCK elements per step."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N, num_programs * BLOCK):
        off = start + cols
        mask = off < N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        y = tl.load(Y + off, mask, other=0.0).to(tl.float32)
        tl.store(Out + off, tl.fdiv(x, 1.0 + tl.exp(-x)) * y, mask)


def _flat_eligible(x, y, out):
    """The flat kernel indexes linearly, so it needs dense same-shape operands.

    Tiny inputs (<= 8192 elements) stay on the codegen kernel: there the whole
    tensor is a single short-lived launch and the flat kernel measured 15%
    slower (fp16 4096 elems: 4.68us vs 4.01us), the opposite of the large-shape
    result.
    """
    return (
        x.dtype == y.dtype
        and x.dtype == out.dtype
        and x.dtype in _FLAT_DTYPES
        and x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and out.is_contiguous()
        and x.numel() > 8192
    )


def _flat_block(n, element_size):
    if element_size >= 4 and n > (1 << 22):
        return _SILU_BLOCK_WIDE
    for limit, tile in _SILU_TILE_LEVELS:
        if n <= limit:
            return tile
    return _SILU_BLOCK_16BIT


def _launch_flat(x, y, out):
    n = x.numel()
    block = _flat_block(n, out.element_size())
    grid = (min(_SILU_MAX_GRID, triton.cdiv(n, block)),)
    _silu_and_mul_flat_kernel[grid](x, y, out, n, block, num_warps=4)


def silu_and_mul_out(A, B, out):
    # Reuse the out0 path of the enflame-specialized codegen: the generic out implementation
    # in pointwise_dynamic measures 2.4x slower on large shapes; the specialized version shares the same efficient
    # kernels as the main op
    logger.debug("GEMS_ENFLAME SILU_AND_MUL_OUT")
    if _flat_eligible(A, B, out):
        _launch_flat(A, B, out)
        return out
    silu_and_mul_kernel(A, B, out0=out)
    return out


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        logger.debug("GEMS_ENFLAME SILU_AND_MUL_FORWARD")
        return silu_and_mul_kernel(A, B)

    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
        return grad_A, grad_B


def silu_and_mul(A, B):
    # Inference / no-grad paths use the codegen out0 channel directly, skipping autograd.Function.apply
    # context construction overhead (measured 15.2us -> 3.9us on small shapes); gradient cases keep the original
    # semantics
    logger.debug("GEMS_ENFLAME SILU_AND_MUL")
    if not torch.is_grad_enabled() or not (A.requires_grad or B.requires_grad):
        out = torch.empty_like(A)
        if _flat_eligible(A, B, out):
            _launch_flat(A, B, out)
        else:
            silu_and_mul_kernel(A, B, out0=out)
        return out
    return SiluAndMul.apply(A, B)
