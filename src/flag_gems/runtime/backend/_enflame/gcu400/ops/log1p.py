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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs). The generic kernel launches
# cdiv(n_elements, BLOCK_SIZE) CTAs, which exceeds the 65535 grid.x hardware
# limit for large inputs (e.g. 2**28 elements -> 262144 CTAs ->
# OutOfResources). Striding from a fixed grid keeps every launch within the
# limit regardless of input size.
_GRID_SIZE = 48


@libentry()
@triton.jit
def log1p_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.log(1.0 + x.to(tl.float32)).to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)


def _config(n_elements, dtype):
    """Returns (BLOCK_SIZE, GRID_SIZE, num_warps, NUM_STAGES), tiered by dtype x size (v2).

    Official bench kernel mode (do_bench median w1000/r100, GCU400/Libra) sweep results
    (40+ configs x 6 variant-body ablations, see the log1p__opt.py docstring):
    - fp32: BS8192/nw4/ST4 is optimal across the whole space (tl.log becomes an extern call,
      leaving no room in the config domain; 2^24 75.1us vs the 50.2us copy floor);
    - fp16/bf16 n>=2^26 use BS131072 (+2.4~4.9%), 4096<n<2^26 keeps BS65536;
    - n<=4096 uses BS=max(512, npo2(n//2)) to spread launch latency over more CTAs (+5.6~8.3%).
    """
    if dtype == torch.float32:
        BS, G, nw, ST = 8192, _GRID_SIZE, 4, 4
    else:
        BS = 131072 if n_elements >= (1 << 26) else 65536
        G, nw, ST = _GRID_SIZE, 2, 1
    if n_elements <= 4096:
        BS = max(512, triton.next_power_of_2(n_elements // 2))
        nw, ST = 2, 1
    elif n_elements < BS:
        BS = max(1024, triton.next_power_of_2(n_elements))
    num_tile = triton.cdiv(n_elements, BS)
    return BS, min(G, num_tile), nw, ST


def _launch(x_contig: torch.Tensor, out_contig: torch.Tensor):
    n_elements = out_contig.numel()
    if n_elements == 0:
        return
    BS, G, nw, ST = _config(n_elements, out_contig.dtype)
    grid = (G, 1, 1)
    with torch_device_fn.device(out_contig.device.index):
        log1p_kernel[grid](
            x_contig,
            out_contig,
            n_elements,
            BLOCK_SIZE=BS,
            GRID_SIZE=G,
            NUM_STAGES=ST,
            num_warps=nw,
        )


def log1p(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME LOG1P")
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    _launch(x_contig.view(-1), out.view(-1))
    return out.view_as(x)


def log1p_(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME LOG1P_")
    if not x.is_floating_point():
        # Integer in-place is not representable; mirror the functional path.
        x.copy_(log1p(x))
        return x

    if x.is_contiguous():
        _launch(x.view(-1), x.view(-1))
    else:
        x_contig = x.contiguous()
        _launch(x_contig.view(-1), x_contig.view(-1))
        x.copy_(x_contig)
    return x


def log1p_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME LOG1P_OUT")
    x_contig = x.to(out.dtype).contiguous()

    if out.is_contiguous():
        _launch(x_contig.view(-1), out.view(-1))
    else:
        out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch(x_contig.view(-1), out_contig.view(-1))
        out.copy_(out_contig)
    return out


@triton.jit
def _log1p_backward_grad(grad, x):
    """``grad / (1 + x)`` with ATen's two-rounding order and IEEE division.

    ATen evaluates ``1 + x`` in float32 opmath, rounds the sum back to the input
    dtype, then divides in float32 opmath and rounds to the output dtype.
    Reproducing that order keeps fp16/bf16 bit-exact, and ``div_rn`` replaces
    Triton's fast division so the quotient is round-to-nearest as well.
    """
    value = x.to(tl.float32)
    denominator = (1.0 + value).to(x.dtype)
    return tl.math.div_rn(grad.to(tl.float32), denominator.to(tl.float32)).to(
        grad.dtype
    )


@libentry()
@triton.jit
def log1p_backward_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Fixed-grid strided kernel: one fused pass, no aten add/div composite."""
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        tl.store(out_ptr + offsets, _log1p_backward_grad(grad, x), mask=mask)


def log1p_backward(grad_output: torch.Tensor, x: torch.Tensor):
    """Fused gradient of ``log1p``: ``grad_output / (1 + x)``.

    Both operands are flattened in elementwise order after materializing
    non-contiguous / zero-stride (expanded) gradients, so a single flat kernel
    serves every stride pattern while still reading x's logical values.
    """
    x_contig = x if x.is_contiguous() else x.contiguous()
    grad_contig = (
        grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    )
    if x_contig.dtype != grad_contig.dtype:
        grad_contig = grad_contig.to(x_contig.dtype)

    out = torch.empty_like(x_contig)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    BS, G, nw, ST = _config(n_elements, out.dtype)
    with torch_device_fn.device(out.device.index):
        log1p_backward_kernel[(G, 1, 1)](
            grad_contig.view(-1),
            x_contig.view(-1),
            out.view(-1),
            n_elements,
            BLOCK_SIZE=BS,
            GRID_SIZE=G,
            NUM_STAGES=ST,
            num_warps=nw,
        )
    return out
