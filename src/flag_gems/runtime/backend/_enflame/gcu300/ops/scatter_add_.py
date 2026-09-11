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

from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(
    f'flag_gems.runtime._enflame.gcu300.ops.{__name__.split(".")[-1]}'
)


# ---------------------------------------------------------------------------
# Triton kernel: 2D dim=1 scatter_add without tl.atomic_add
#
# Key insight: each output element independently scans all source entries.
#   out[b, v] += sum_{n: index[b,n]=v} src[b, n]
#
# Each program handles one (batch, V_element) pair in a persistent grid-stride
# loop. For each V element, it scans source entries in tiles of N_TILE,
# using broadcast comparison (v vs vector) → tl.sum of matching values.
#
# Total work = B * V * ceil(N / N_TILE) comparisons.
# For vLLM sampling (B≈1-32, V≈50k-150k, N≈100-1000), this is acceptable.
# ---------------------------------------------------------------------------
@triton.jit
def _scatter_add_2d_dim1_kernel(
    out_ptr,  # [B, V] output (flattened)
    index_ptr,  # [B, N] indices
    src_ptr,  # [B, N] source values
    B: int,
    V: int,
    N: int,
    stride_o0: int,
    stride_i0: int,
    stride_s0: int,
    N_TILE: tl.constexpr,
):
    """Scatter-add on dim=1 for 2D tensors.  No tl.atomic_add.

    Each program processes multiple (b, v) work items via grid-stride loop.
    For a single output element out[b, v]:
      1. Load current out[b, v]
      2. Scan all source entries src[b, *] to find those with index == v
      3. Accumulate and store back
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_work = B * V

    # Persistent kernel: grid-stride loop
    for wid in range(pid, total_work, num_programs.to(tl.int32)):
        b = wid // V
        v = wid % V

        # Base pointers for this batch row
        b64 = b.to(tl.int64)
        out_b = out_ptr + b64 * stride_o0.to(tl.int64)
        idx_b = index_ptr + b64 * stride_i0.to(tl.int64)
        src_b = src_ptr + b64 * stride_s0.to(tl.int64)

        # Load current output value → float32 accumulator
        # v = wid % V is always in [0, V) → safe scalar load
        acc = tl.load(out_b + v.to(tl.int32))
        acc = acc.to(tl.float32)

        # Scan all N source entries in tiles of N_TILE
        for n_start in range(0, N, N_TILE):
            n_offs = n_start + tl.arange(0, N_TILE)
            n_mask = n_offs < N

            # Load tile of source indices and values
            # 'other=-1' → out-of-bounds entries never match
            cur_idx = tl.load(idx_b + n_offs, mask=n_mask, other=-1)
            cur_val = tl.load(src_b + n_offs, mask=n_mask, other=0.0)
            cur_val = cur_val.to(tl.float32)

            # Broadcast comparison: (N_TILE,) == scalar → (N_TILE,) bool
            matches = (cur_idx == v.to(tl.int32)) & n_mask
            # Sum matching values into a scalar contribution
            acc += tl.sum(tl.where(matches, cur_val, 0.0))

        # Write back (cast to output dtype) – v is always valid
        tl.store(out_b + v.to(tl.int32), acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Fast-path entry: 2D dim=1 (covers vLLM sampling)
# ---------------------------------------------------------------------------
def scatter_add_2d_dim1(inp, index, src):
    """Triton-based 2D dim=1 scatter_add without tl.atomic_add.

    inp:  (B, V)  – output tensor
    index: (B, N) – index tensor (int32/int64)
    src:   (B, N) – source tensor (matching index shape on dim=1)
    """
    B, V = inp.shape
    N = index.size(1) if index.dim() > 1 else 1
    orig_dtype = inp.dtype

    # GCU Triton doesn't support int64 → convert to int32
    index = index.to(torch.int32) if index.dtype == torch.int64 else index

    # Promote half-dtypes → float32 for safe accumulation
    dtype_convert = inp.dtype in (torch.float16, torch.bfloat16)
    if dtype_convert:
        out = inp.to(torch.float32)
    else:
        out = inp

    # Choose N_TILE: balance between inner loop unroll & register pressure
    n_tile = min(128, max(32, triton.next_power_of_2(min(N, 256))))

    # Grid: persistent kernel, 24 programs max (GCU300 constraint)
    grid = (24,) if B * V > 24 else (B * V,)

    _scatter_add_2d_dim1_kernel[grid](
        out,
        index,
        src,
        B,
        V,
        N,
        out.stride(0),
        index.stride(0),
        src.stride(0),
        N_TILE=n_tile,
    )

    if dtype_convert:
        inp.copy_(out.to(orig_dtype))
        return inp
    return out


# ---------------------------------------------------------------------------
# General fallback (non-2D dim=1) – fast vectorised CPU path
# ---------------------------------------------------------------------------
def _general_fallback(inp, dim, index, src):
    """CPU fallback for general scatter_add_ shapes."""
    dim = dim % inp.ndim
    dim_stride = inp.stride(dim)
    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)

    N = index.numel()
    if N == 0:
        return inp

    flat_index = index.reshape(-1).to(torch.int64).cpu()
    flat_src = src_strided.reshape(-1).contiguous().cpu()
    out_cpu = inp.cpu()

    shape = index.shape
    strides = inp_restrided.stride()

    idx = torch.arange(N, dtype=torch.int64)
    base = torch.zeros(N, dtype=torch.int64)
    for d in reversed(range(len(shape))):
        c = idx % shape[d]
        idx = idx // shape[d]
        if d != dim:
            base += c * strides[d]

    out_flat = out_cpu.reshape(-1)
    for i in range(N):
        offset = int(base[i].item()) + int(flat_index[i].item()) * dim_stride
        out_flat[offset] += float(flat_src[i].item())

    inp.copy_(out_cpu.to(inp.device))
    return inp


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def scatter_add_(inp, dim, index, src):
    """GCU300 scatter_add_ without tl.atomic_add.

    Fast path: 2D tensors with dim=1 → Triton kernel.
    General case: CPU fallback.
    """
    logger.debug("GEMS_GCU300 SCATTER_ADD_ (no-atomic Triton)")

    dim = dim % inp.ndim

    if inp.dim() == 2 and dim == 1:
        return scatter_add_2d_dim1(inp, index, src)

    return _general_fallback(inp, dim, index, src)
