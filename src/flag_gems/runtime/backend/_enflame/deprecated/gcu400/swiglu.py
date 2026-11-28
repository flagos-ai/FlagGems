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

"""gcu400 SwiGLU and dSwiGLU on row-tiled streaming kernels.

The generic implementation walks the (M, 2H) input with a two dimensional
64x64 tile and one CTA per tile, so the program count grows with
``M * H / 4096``: 1e9 elements launch 262144 CTAs and the launch cost dominates
(measured 31.6 ms for 1e9 bf16 elements, ~95 GB/s).

Two things matter on this backend:

1. Launch shape.  A small fixed CTA grid (48) walking a pipelined
   ``tl.range`` tile loop keeps every CTA on a long streaming run instead of
   paying a per-CTA setup cost, exactly like the ``gelu_and_mul``/``log1p``
   family.  Small inputs shrink the tile so the work still spreads over the
   grid.

2. Tile shape.  A flat one-dimensional tile has to address the gate and shift
   halves as ``2 * off - (off % H)`` and ``+ H``; the compiler cannot prove the
   last dimension is unit stride, so the accesses degrade to per-element loads
   (9 ms for 1e9 elements, six times slower than a contiguous copy of the same
   bytes).  A ``(R, H)`` tile keeps the head dimension unit stride, which
   vectorises the loads and stores: 3.4 ms for the same 1e9 elements, ~900 GB/s.

The tile is therefore ``R`` whole rows of ``H`` output elements, with ``H``
passed as a constexpr so the row shift is a compile-time constant.  Heads wider
than ``_MAX_TILE_COLS`` fall back to the flat kernel, and heads that are not a
power of two are handled by the same row kernel (no divisibility assumption is
made about ``H``).
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _sigmoid(x):
    """Sigmoid through the backend fast-exp/fdiv path.

    ``tl.sigmoid`` lowers to a libdevice call that costs several times more on
    this backend; the explicit exp/fdiv form is the same function at a fraction
    of the element cost (same trick as the gelu family kernels).
    """
    return tl.fdiv(1.0, 1.0 + tl.exp(-x))


def _env_int(name, default):
    """Read an integer tuning knob, falling back to the tuned default."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


# Tuned on GCU400/L600 (kernel-mode do_bench).  The env overrides only exist for
# the offline sweep that produced these numbers.
_GRID_SIZE = _env_int("FLAGGEMS_SWIGLU_GRID", 48)
_TILE_ELEMS = _env_int("FLAGGEMS_SWIGLU_TILE", 65536)
_MAX_TILE_COLS = 65536
# Element count below which the flat kernel wins: its tile spans whole rows, so
# it keeps the accesses wide even when the per-CTA tile has to shrink.
_TILE_MIN_ELEMS = 1 << 19
# Below this head width the last dimension is too short to vectorise, and the
# flat tile (which packs whole rows into one stream) wins.
_MIN_TILE_COLS = 128
_NUM_WARPS = _env_int("FLAGGEMS_SWIGLU_NW", 4)
_FORCE_FLAT = _env_int("FLAGGEMS_SWIGLU_FORCE_FLAT", 0)


@libentry()
@triton.jit
def _swiglu_tile(
    X,
    Out,
    M,
    H: tl.constexpr,
    R: tl.constexpr,
    GRID: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tile = (M + R - 1) // R
    rows = tl.arange(0, R)
    cols = tl.arange(0, H)
    for tile_id in tl.range(pid, num_tile, GRID):
        row = tile_id * R + rows
        in_off = row[:, None] * (2 * H) + cols[None, :]
        out_off = row[:, None] * H + cols[None, :]
        if MASKED:
            mask = (row < M)[:, None]
            x_a = tl.load(X + in_off, mask=mask, other=0.0).to(tl.float32)
            x_b = tl.load(X + in_off + H, mask=mask, other=0.0).to(tl.float32)
            tl.store(Out + out_off, (x_a * _sigmoid(x_a)) * x_b, mask=mask)
        else:
            x_a = tl.load(X + in_off).to(tl.float32)
            x_b = tl.load(X + in_off + H).to(tl.float32)
            tl.store(Out + out_off, (x_a * _sigmoid(x_a)) * x_b)


@libentry()
@triton.jit
def _dswiglu_tile(
    GO,
    X,
    DX,
    M,
    H: tl.constexpr,
    R: tl.constexpr,
    GRID: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tile = (M + R - 1) // R
    rows = tl.arange(0, R)
    cols = tl.arange(0, H)
    for tile_id in tl.range(pid, num_tile, GRID):
        row = tile_id * R + rows
        in_off = row[:, None] * (2 * H) + cols[None, :]
        go_off = row[:, None] * H + cols[None, :]
        if MASKED:
            mask = (row < M)[:, None]
            grad_out = tl.load(GO + go_off, mask=mask, other=0.0).to(tl.float32)
            x_a = tl.load(X + in_off, mask=mask, other=0.0).to(tl.float32)
            x_b = tl.load(X + in_off + H, mask=mask, other=0.0).to(tl.float32)
        else:
            grad_out = tl.load(GO + go_off).to(tl.float32)
            x_a = tl.load(X + in_off).to(tl.float32)
            x_b = tl.load(X + in_off + H).to(tl.float32)
        sig = _sigmoid(x_a)
        silu = x_a * sig
        # d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        d_silu = sig * (1.0 + x_a * (1.0 - sig))
        grad_a = grad_out * x_b * d_silu
        grad_b = grad_out * silu
        if MASKED:
            mask = (row < M)[:, None]
            tl.store(DX + in_off, grad_a, mask=mask)
            tl.store(DX + in_off + H, grad_b, mask=mask)
        else:
            tl.store(DX + in_off, grad_a)
            tl.store(DX + in_off + H, grad_b)


@libentry()
@triton.jit
def _swiglu_flat(
    X,
    Out,
    N,
    H,
    H_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
    POW2: tl.constexpr,
):
    """Fallback for tiny inputs and wide/odd head dims: flat stream."""
    pid = tl.program_id(0)
    num_tile = (N + BLOCK - 1) // BLOCK
    cols = tl.arange(0, BLOCK)
    if I64:
        cols = cols.to(tl.int64)
    for tile_id in tl.range(pid, num_tile, GRID):
        off = tile_id * BLOCK + cols
        if POW2:
            a_off = 2 * off - (off & H_MASK)
        else:
            a_off = 2 * off - (off % H)
        if EVEN:
            x_a = tl.load(X + a_off).to(tl.float32)
            x_b = tl.load(X + a_off + H).to(tl.float32)
            tl.store(Out + off, (x_a * _sigmoid(x_a)) * x_b)
        else:
            mask = off < N
            x_a = tl.load(X + a_off, mask=mask, other=0.0).to(tl.float32)
            x_b = tl.load(X + a_off + H, mask=mask, other=0.0).to(tl.float32)
            tl.store(Out + off, (x_a * _sigmoid(x_a)) * x_b, mask=mask)


@libentry()
@triton.jit
def _dswiglu_flat(
    GO,
    X,
    DX,
    N,
    H,
    H_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
    POW2: tl.constexpr,
):
    """Backward fallback for tiny inputs and wide/odd head dims."""
    pid = tl.program_id(0)
    num_tile = (N + BLOCK - 1) // BLOCK
    cols = tl.arange(0, BLOCK)
    if I64:
        cols = cols.to(tl.int64)
    for tile_id in tl.range(pid, num_tile, GRID):
        off = tile_id * BLOCK + cols
        if POW2:
            a_off = 2 * off - (off & H_MASK)
        else:
            a_off = 2 * off - (off % H)
        if EVEN:
            grad_out = tl.load(GO + off).to(tl.float32)
            x_a = tl.load(X + a_off).to(tl.float32)
            x_b = tl.load(X + a_off + H).to(tl.float32)
        else:
            mask = off < N
            grad_out = tl.load(GO + off, mask=mask, other=0.0).to(tl.float32)
            x_a = tl.load(X + a_off, mask=mask, other=0.0).to(tl.float32)
            x_b = tl.load(X + a_off + H, mask=mask, other=0.0).to(tl.float32)
        sig = _sigmoid(x_a)
        silu = x_a * sig
        d_silu = sig * (1.0 + x_a * (1.0 - sig))
        grad_a = grad_out * x_b * d_silu
        grad_b = grad_out * silu
        if EVEN:
            tl.store(DX + a_off, grad_a)
            tl.store(DX + a_off + H, grad_b)
        else:
            tl.store(DX + a_off, grad_a, mask=mask)
            tl.store(DX + a_off + H, grad_b, mask=mask)


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def _plan(num_rows, head_dim, element_size=2):
    """Choose the tiling strategy for one launch.

    Returns ``("tile", rows, grid, masked)`` for the row-tiled kernel, or
    ``("flat", block, grid, even, i64, pow2)`` for the flat fallback.  Tiny
    problems keep the single-CTA exact-size tile that the flat path provides;
    the row tile needs both dimensions to be powers of two because Triton
    ranges must be.
    """
    n = num_rows * head_dim
    pow2 = _is_pow2(head_dim)
    # A 4-byte element halves the number of lanes a tile can hold.
    tile_elems = _TILE_ELEMS if element_size <= 2 else _TILE_ELEMS // 2
    if (
        pow2
        and not _FORCE_FLAT
        and n >= _TILE_MIN_ELEMS
        and _MIN_TILE_COLS <= head_dim <= _MAX_TILE_COLS
    ):
        rows = max(1, min(tile_elems // head_dim, -(-num_rows // _GRID_SIZE)))
        rows = 1 << (rows.bit_length() - 1)  # Triton ranges must be powers of two
        grid = min(_GRID_SIZE, triton.cdiv(num_rows, rows))
        return ("tile", rows, grid, (num_rows % rows) != 0)
    if n <= _TILE_ELEMS:
        block, grid = max(64, triton.next_power_of_2(n)), 1
    else:
        block = min(
            tile_elems,
            max(2048, triton.next_power_of_2(triton.cdiv(n, _GRID_SIZE))),
        )
        grid = min(_GRID_SIZE, triton.cdiv(n, block))
    return ("flat", block, grid, (n % block) == 0, n > (1 << 30), pow2)


def _check_input(input_tensor):
    if input_tensor.shape[-1] % 2 != 0:
        raise ValueError(
            f"The last dimension of must be even number, got {input_tensor.shape[-1]}"
        )
    if not input_tensor.is_cuda:
        raise ValueError("Only CUDA tensor is supported by SwiGLU")


def swiglu(input_tensor: torch.Tensor, quantizer=None) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME SWIGLU")
    _check_input(input_tensor)
    shape = input_tensor.shape
    head_dim = shape[-1] // 2
    if head_dim == 0:
        return torch.empty(
            shape[:-1] + (0,), device=input_tensor.device, dtype=input_tensor.dtype
        )
    num_rows = input_tensor.numel() // (2 * head_dim)
    input_flat = input_tensor.contiguous().view(-1)
    out_flat = torch.empty(
        num_rows * head_dim, device=input_tensor.device, dtype=input_tensor.dtype
    )

    plan = _plan(num_rows, head_dim, input_flat.element_size())
    if plan[0] == "tile":
        _, rows, grid, masked = plan
        _swiglu_tile[(grid, 1, 1)](
            input_flat,
            out_flat,
            num_rows,
            head_dim,
            rows,
            grid,
            masked,
            num_warps=_NUM_WARPS,
        )
    else:
        _, block, grid, even, i64, pow2 = plan
        _swiglu_flat[(grid, 1, 1)](
            input_flat,
            out_flat,
            num_rows * head_dim,
            head_dim,
            head_dim - 1,
            block,
            grid,
            even,
            i64,
            pow2,
            num_warps=_NUM_WARPS,
        )
    return out_flat.view(*shape[:-1], head_dim)


def dswiglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer=None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME DSWIGLU")
    _check_input(input_tensor)
    shape = input_tensor.shape
    head_dim = shape[-1] // 2
    if head_dim == 0:
        return torch.empty_like(input_tensor)
    num_rows = input_tensor.numel() // (2 * head_dim)
    grad_out_flat = grad_output.contiguous().view(-1)
    input_flat = input_tensor.contiguous().view(-1)
    grad_in_flat = torch.empty_like(input_flat)

    plan = _plan(num_rows, head_dim, input_flat.element_size())
    if plan[0] == "tile":
        _, rows, grid, masked = plan
        _dswiglu_tile[(grid, 1, 1)](
            grad_out_flat,
            input_flat,
            grad_in_flat,
            num_rows,
            head_dim,
            rows,
            grid,
            masked,
            num_warps=_NUM_WARPS,
        )
    else:
        _, block, grid, even, i64, pow2 = plan
        _dswiglu_flat[(grid, 1, 1)](
            grad_out_flat,
            input_flat,
            grad_in_flat,
            num_rows * head_dim,
            head_dim,
            head_dim - 1,
            block,
            grid,
            even,
            i64,
            pow2,
            num_warps=_NUM_WARPS,
        )
    return grad_in_flat.view_as(input_tensor)


__all__ = ["swiglu", "dswiglu"]
