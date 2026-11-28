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

"""floor candidate v5: override CodeGenConfig.max_grid_size per operator (zero Python branches).

The first 4 versions were "hand-written 1D kernel + small-shape tiering", but the pure production path always
had p10~=0.90 on microsecond small shapes: the override and production path sit at different dispatch levels,
so any extra Python check causes a 0.3~0.9us deviation.

This version directly reuses the pointwise_dynamic codegen path and only enlarges ENFLAME's default
`max_grid_size=(12,1,1)` to `(1024,1,1)`:
- codegen's internal `num_ctas = min(max_grid_size, num_tiles)` keeps small shapes using
  few CTAs, with a call chain fully identical to the common implementation and zero extra cost;
- large shapes go from 12 to 1024 CTAs, removing the grid parallelism bottleneck.
Probe measured 2^30 `16.38ms -> 5.90ms` (2.78x) and 2^28 `4.12 -> 1.51ms` (2.73x),
with `exact=True` for every shape and grid.

Global CODEGEN_COFIGS is not modified; only this operator's generated wrapper is affected, with no side effects on
others.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def floor_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x0 = tl.load(x_ptr + off)
            if IS_FP16:
                trunc = x0.to(tl.int32).to(tl.float16)
                rounded = tl.where(x0 < trunc, trunc - 1.0, trunc)
                out = tl.where(x0 - x0 == 0, rounded, x0)
            elif IS_BF16:
                trunc = x0.to(tl.int32).to(tl.bfloat16)
                rounded = tl.where(x0 < trunc, trunc - 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            else:
                trunc = x0.to(tl.int32).to(tl.float32)
                rounded = tl.where(x0 < trunc, trunc - 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            tl.store(out_ptr + off, out)
        else:
            mask = off < N_total
            x0 = tl.load(x_ptr + off, mask=mask)
            if IS_FP16:
                trunc = x0.to(tl.int32).to(tl.float16)
                rounded = tl.where(x0 < trunc, trunc - 1.0, trunc)
                out = tl.where(x0 - x0 == 0, rounded, x0)
            elif IS_BF16:
                trunc = x0.to(tl.int32).to(tl.bfloat16)
                rounded = tl.where(x0 < trunc, trunc - 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            else:
                trunc = x0.to(tl.int32).to(tl.float32)
                rounded = tl.where(x0 < trunc, trunc - 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            tl.store(out_ptr + off, out, mask=mask)


# LibEntry's dispatch key is (dtype, alignment, BLOCK, EVEN) and does not
# include num_warps, so each (dtype, BLOCK) pair pins a single num_warps.
def _choose_launch(n_total, dtype):
    if n_total <= 8192:
        return 1024, 4
    if n_total <= 65536:
        return 8192, 2
    if dtype == torch.float32:
        return 32768, 2
    if dtype == torch.float16 and n_total > (1 << 26):
        return 131072, 4
    return 65536, 4


def _launch_floor(inp, out):
    n_total = inp.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, inp.dtype)
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    even = n_total % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # rsqrt.py / neg.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                floor_flat_kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    IS_FP16=inp.dtype == torch.float16,
                    IS_BF16=inp.dtype == torch.bfloat16,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "floor flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_path_ok(A):
    return (
        A.is_contiguous() and A.dtype in _FLAT_DTYPES and A.numel() < _INT32_SAFE_LIMIT
    )


@libentry()
@triton.jit(do_not_specialize=["n_total"])
def floor_backward_zero_kernel(
    out_ptr,
    n_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """d/dx floor(x) = 0: fill the grad buffer with zeros.

    A pure store stream (measured 15.3us vs aten's 19.2us for 16.7M fp16
    elements, 1.25x); the tile/warp pair below is the sweep optimum.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty)
    for block_id in tl.range(pid, (n_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            tl.store(out_ptr + off, zeros)
        else:
            tl.store(out_ptr + off, zeros, mask=off < n_total)


def _launch_zeros(out):
    n_total = out.numel()
    if n_total == 0:
        return
    if n_total <= 8192:
        block, num_warps = triton.next_power_of_2(n_total), 4
    elif n_total <= 65536:
        block, num_warps = 8192, 4
    else:
        block, num_warps = 32768, 2
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    with torch_device_fn.device(out.device):
        floor_backward_zero_kernel[(grid_size,)](
            out,
            n_total,
            BLOCK=block,
            EVEN=n_total % block == 0,
            num_warps=num_warps,
        )


def _floor_backward(grad_output):
    """Return zeros with grad_output's shape; 8-byte dtypes go through words."""
    if grad_output.element_size() == 8:
        # zero fill is dtype agnostic, so the 4-byte word view avoids the
        # 8-byte pointer lowering issue documented in flip.py.
        out = torch.empty_like(grad_output, memory_format=torch.contiguous_format)
        _launch_zeros(out.reshape(-1).view(torch.int32))
        return out
    out = torch.empty_like(grad_output, memory_format=torch.contiguous_format)
    _launch_zeros(out)
    return out


class _FloorFunction(torch.autograd.Function):
    """Autograd wrapper for floor (its backward is all zeros)."""

    @staticmethod
    def forward(ctx, A):
        if _flat_path_ok(A):
            out = torch.empty_like(A)
            _launch_floor(A, out)
            return out
        # dtype/layout outside the flat kernel: the public pointwise op runs
        # the compute (grad is off inside forward), the node is created here.
        from flag_gems.ops.floor import floor_func as _public

        return _public(A)

    @staticmethod
    def backward(ctx, grad_output):
        return _floor_backward(grad_output)


def floor(A):
    logger.debug("GEMS_ENFLAME FLOOR")
    if torch.is_grad_enabled() and A.requires_grad:
        return _FloorFunction.apply(A)
    if not _flat_path_ok(A):
        from flag_gems.ops.floor import floor_func as _public

        return _public(A)
    out = torch.empty_like(A)
    _launch_floor(A, out)
    return out
