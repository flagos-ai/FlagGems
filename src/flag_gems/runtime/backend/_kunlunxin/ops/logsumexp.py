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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Tile budget (in elements) for the multirow constexpr-N kernel. A [TILE_M, N]
# fp32 tile stays within this budget so it fits XPU uni_sram. N values above
# this fall back to the per-row 1D-loop kernel.
_MULTIROW_BUDGET = 32768

# Redispatch key used to reach PyTorch's native (vendor) logsumexp. On this XPU
# the vendor's fused logsumexp kernel beats any Triton path we can express for a
# middle-dim (K>1) reduction (see the module docstring / solution doc), so the
# K>1 branch defers to it instead of materializing a slow transpose copy.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


@libentry()
@triton.jit
def logsumexp_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    """Reduce the innermost dim N for many rows per program.

    N is a constexpr so ``tl.arange(0, N)`` spans exactly [0, N): the
    ``[TILE_M, N]`` tile is one stride-1 contiguous block -> block DMA on XPU
    (vs. the discrete access a runtime N produces). One program handles TILE_M
    rows, amortizing launch overhead for large M.
    """
    pid = ext.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    m_mask = m_offsets < M
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    inp = tl.load(input_ptr + offsets, mask=m_mask[:, None], other=-float("inf")).to(
        tl.float32
    )
    m = tl.max(inp, axis=1)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
    out = tl.where(m == float("-inf"), m, safe_m + tl.log(z))
    tl.store(output_ptr + m_offsets, out, mask=m_mask)


@libentry()
@triton.jit
def logsumexp_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    """Per-row 1D-loop kernel, used as the fallback for large N (N > budget)."""
    pid_m = ext.program_id(0)
    m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
    input_ptr += pid_m * N

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m_new = tl.maximum(m, inp)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
        m = m_new

    m_reduced = tl.max(m, axis=0)
    z = tl.sum(z * tl.exp(m - m_reduced), axis=0)
    m = m_reduced
    # Handle case where all inputs were -inf
    out = tl.where(m == float("-inf"), m, m + tl.log(z))
    output_ptrs = output_ptr + pid_m
    tl.store(output_ptrs, out)


def _reduce_inner(inp, rows, N):
    """logsumexp over the innermost dim N of a contiguous [rows, N] tensor."""
    out = torch.empty((rows,), dtype=inp.dtype, device=inp.device)
    if N <= _MULTIROW_BUDGET:
        TILE_M = max(1, _MULTIROW_BUDGET // N)
        grid = (triton.cdiv(rows, TILE_M), 1, 1)
        logsumexp_kernel_multirow[grid](
            out,
            inp,
            rows,
            N=N,
            TILE_M=TILE_M,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    else:
        TILE_N = min(triton.next_power_of_2(N), 4096)
        grid = (rows, 1, 1)
        logsumexp_kernel_inner[grid](
            out,
            inp,
            rows,
            N,
            TILE_N=TILE_N,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return out


def _native_logsumexp(inp, dim, keepdim):
    """Reach PyTorch's native (vendor) logsumexp, bypassing the gems override."""
    return torch.ops.aten.logsumexp.default.redispatch(
        _FALLBACK_KEYSET, inp, dim, keepdim
    )


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            # Empty dim list means no reduction, just return the input.
            return inp.clone()
        if len(dim) != 1:
            # Multi-dim reduction: the vendor's native kernel beats a sequence
            # of Triton reductions on this XPU.
            return _native_logsumexp(inp, list(dim), keepdim)
        dim = dim[0]

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    N = inp.shape[dim]
    K = 1
    for i in range(dim + 1, inp.ndim):
        K *= inp.shape[i]

    # Middle-dim reduction (K > 1) or a size-1 reduction: defer to the native
    # vendor kernel. A Triton middle reduction on XPU is a dead end -- a physical
    # transpose+contiguous can't reach the vendor's fast copy once gems overrides
    # copy_, and a direct strided/discrete reduction either overflows uni_sram or
    # mis-computes (2D axis=0 reduce is rejected). N==1 is a trivial identity that
    # the native kernel does faster than a gems copy.
    if K > 1 or N == 1:
        return _native_logsumexp(inp, [dim], keepdim)

    # K == 1: innermost-dim reduction -> fast contiguous Triton multirow tile.
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    shape = list(inp.shape)
    shape[dim] = 1

    with torch_device_fn.device(inp.device):
        out = _reduce_inner(inp, M, N).view(shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
