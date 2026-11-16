# Copyright 2026, The FlagOS Contributors.
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

"""Kunlunxin XPU3 override for ``aten::inner``.

The generic implementation already carries an XPU tuning branch, but two
measured XPU3 wins are left on the table:

* 16-bit large GEMM: a mask-free ``256x256x256`` tile is ~30% faster than the
  generic tiles on the big shapes (fp16 145.5 -> 94.3 us at
  m=1024,n=2048,k=4096 and 258.8 -> 178.0 us at 2048x2048x4096; bf16
  213.6 -> 187.0 us and 418.9 -> 361.9 us).  It is only selected when every
  tile dimension divides exactly, so the ``tl.dot`` never consumes a masked
  load (masked dot loads are a known silent-miscompile hazard here).
  ``num_stages`` is dtype-specific: 4 for fp16, 2 for bf16, and the bf16 wide
  tile additionally requires ``k >= 2048`` (shorter-K bf16 keeps the generic
  tile, which measured marginally faster there).
* 1-D dot: the generic single-program path serializes once ``k`` grows past
  ~8K (index/load loop inside one program).  A 4096-wide partial sum over
  ``ceil(k/4096)`` programs plus a tiny tree-reduce is ~20% faster at
  k=16384 (10.6 -> 8.8 us) and 2x at k=65536 (25.6 -> 12.0 us).  Below 8K the
  extra launch loses, so those keep the single-program path.

Everything else reuses the generic kernels verbatim; only the tile/segment
selection changes.  No compute op is added on the host side.
"""

import logging

import torch
import triton

from flag_gems.ops.inner import (
    _fp32_dot_precision,
    _inner_dot_kernel,
    _inner_dot_reduce_kernel,
    _inner_gemm_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Below this many elements the single-program contraction wins on XPU3; above
# it the partial+tree-reduce split wins (measured crossover between 8K and 12K).
_SINGLE_PROGRAM_MAX_K = 8192
# Widest per-program segment for the split 1-D path (<= 4096 avoids the
# BLOCK>=32768 illegal-access wall and the BLOCK<64 store spill).
_SPLIT_BLOCK = 4096
_SPLIT_MAX_GRID = 64


def _select_block_config(m, n, k, dtype):
    """Pick GEMM tile sizes for XPU3.

    Identical to the generic XPU branch except the 16-bit large tile, which
    uses a wider, mask-free 256x256x256 tile when the shape divides evenly.
    """
    if m < 16 or n < 16:
        return 16, 16, 32, 4, 3
    large = m >= 512 and n >= 512 and k >= 512
    if not large:
        return 128, 128, 64, 4, 4
    divisible = m % 256 == 0 and n % 256 == 0 and k % 256 == 0
    if dtype == torch.float16:
        if divisible:
            return 256, 256, 256, 8, 4
        return 256, 128, 128, 8, 4
    if dtype == torch.bfloat16:
        # Only when the K loop is long enough to hide the wider tile's setup;
        # short-K bf16 (e.g. k=1024) is marginally faster on the generic tile.
        if divisible and k >= 2048:
            return 256, 256, 256, 8, 2
        return 256, 256, 128, 8, 4
    # fp32 keeps the generic 256x256x128 tile: a wider BLOCK_K measured slower
    # there (200+ us vs 175 us at m=1024,n=2048,k=4096).
    return 256, 256, 128, 8, 4


def _inner_1d(input, other, out_dtype, is_fp64):
    """1-D x 1-D contraction (scalar dot product)."""
    device = input.device
    out = torch.empty((), dtype=out_dtype, device=device)
    k = input.numel()
    if k == 0:
        return torch.zeros((), dtype=out_dtype, device=device)
    if k <= _SINGLE_PROGRAM_MAX_K:
        # Short contraction is launch-bound: one program keeps the whole
        # accumulation in registers and avoids the second launch.
        block_size = min(triton.next_power_of_2(k), _SPLIT_BLOCK)
        with torch_device_fn.device(device):
            _inner_dot_kernel[(1,)](
                input,
                other,
                out,
                k,
                BLOCK_SIZE=block_size,
                IS_FP64=is_fp64,
            )
        return out
    # Long contraction: split into 4096-wide partials (one per program) and
    # tree-reduce them.  Wider segments than the generic 64 keep the inner
    # loop short and the per-program reduction efficient.
    block_size = _SPLIT_BLOCK
    grid = (min((k + block_size - 1) // block_size, _SPLIT_MAX_GRID),)
    mid_dtype = torch.float64 if is_fp64 else torch.float32
    mid = torch.empty(grid[0], dtype=mid_dtype, device=device)
    with torch_device_fn.device(device):
        _inner_dot_kernel[grid](
            input,
            other,
            mid,
            k,
            BLOCK_SIZE=block_size,
            IS_FP64=is_fp64,
        )
        _inner_dot_reduce_kernel[(1,)](
            mid,
            out,
            grid[0],
            BLOCK_N=triton.next_power_of_2(grid[0]),
        )
    return out


def inner(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Compute the inner product of two tensors (Kunlunxin XPU3 override)."""
    logger.debug("GEMS INNER")

    if input.dim() == 0 and other.dim() == 0:
        return (input * other).reshape(())

    if input.dtype != other.dtype:
        raise RuntimeError("both inputs should have same dtype")

    if input.dim() == 0:
        return (other * input).reshape(other.shape)
    if other.dim() == 0:
        return (input * other).reshape(input.shape)

    if input.size(-1) != other.size(-1):
        raise RuntimeError(
            "inner() the last dimension must match on both input tensors but "
            f"got shapes {list(input.shape)} and {list(other.shape)}"
        )

    out_dtype = input.dtype
    device = input.device

    if input.dim() == 1 and other.dim() == 1:
        return _inner_1d(input, other, out_dtype, out_dtype == torch.float64)

    k = input.size(-1)
    out_shape = input.shape[:-1] + other.shape[:-1]
    if k == 0:
        return torch.zeros(out_shape, dtype=out_dtype, device=device)
    a2 = input.reshape(-1, k)
    b2 = other.reshape(-1, k)
    m = a2.shape[0]
    n = b2.shape[0]

    out = torch.empty((m, n), dtype=out_dtype, device=device)
    if m == 0 or n == 0:
        return out.view(out_shape)

    block_m, block_n, block_k, num_warps, num_stages = _select_block_config(
        m, n, k, out_dtype
    )
    grid = (((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),)
    has_mask = not (m % block_m == 0 and n % block_n == 0 and k % block_k == 0)
    fp32_precision = _fp32_dot_precision() if out_dtype == torch.float32 else "ieee"
    with torch_device_fn.device(device):
        _inner_gemm_kernel[grid](
            a2,
            b2,
            out,
            m,
            n,
            k,
            a2.stride(0),
            a2.stride(1),
            b2.stride(0),
            b2.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            HAS_MASK=has_mask,
            IS_FP32=out_dtype == torch.float32,
            IS_FP64=out_dtype == torch.float64,
            FP32_PRECISION=fp32_precision,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out.view(out_shape)
