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

"""GCU400-specific addmv: like mv.py, fixed tile/grid with in-kernel striding, no autotune."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)

# GCU400 tuning constants aligned with mv.py
NUM_SIPS = 48
BLOCK_N = 32
BLOCK_M = 1024
# Number of static unroll blocks per group along the K direction: pins down DSM usage (the number of groups is
# determined by K, while the unroll count within a group is fixed)
GROUP_ITERS = 4
SMALL_N_MAX = 128
SMALL_BLOCK_M_MAX = 2048
# Maximum number of static unrolls along K allowed for the small-N row-streaming kernel, to avoid excessive unrolling
# blowing the DSM budget
SMALL_M_ITERS_MAX = 8

# (rows, K, dtype, scalar_bias) -> launch configuration, see _launch_plan
_LAUNCH_PLAN_CACHE = {}


@triton.jit
def _accumulate_group(
    A,
    B,
    row_index,
    k_base,
    M,
    N,
    stride_an,
    stride_am,
    stride_bm,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_ITERS: tl.constexpr,
    EVEN: tl.constexpr,
):
    """(BLOCK_N, BLOCK_M) accumulation for one K group, GROUP_ITERS statically unrolled."""
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for it in tl.static_range(GROUP_ITERS):
        k_index = k_base + it * BLOCK_M + offs_m
        if EVEN:
            a = tl.load(
                A + row_index[:, None] * stride_an + k_index[None, :] * stride_am
            )
            b = tl.load(B + k_index * stride_bm)
        else:
            k_mask = k_index < M
            a = tl.load(
                A + row_index[:, None] * stride_an + k_index[None, :] * stride_am,
                mask=k_mask[None, :] & (row_index[:, None] < N),
                other=0.0,
            )
            b = tl.load(B + k_index * stride_bm, mask=k_mask, other=0.0)
        acc += a * b
    return acc


@libentry()
@triton.jit(do_not_specialize=["M", "N", "alpha", "beta"])
def addmv_kernel(
    A,
    B,
    Inp,
    Out,
    N,
    M,
    alpha,
    beta,
    stride_an,
    stride_am,
    stride_bm,
    stride_in,
    stride_outn,
    BIAS_IS_SCALAR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_ITERS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EVEN: tl.constexpr,
):
    num_programs = tl.num_programs(0)
    num_row_blocks = (N + BLOCK_N - 1) // BLOCK_N
    group_span: tl.constexpr = GROUP_ITERS * BLOCK_M
    for row_block in tl.range(tl.program_id(0), num_row_blocks, num_programs):
        row_index = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if NUM_GROUPS == 1:
            acc = _accumulate_group(
                A,
                B,
                row_index,
                0,
                M,
                N,
                stride_an,
                stride_am,
                stride_bm,
                BLOCK_N,
                BLOCK_M,
                GROUP_ITERS,
                EVEN,
            )
        else:
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for group in tl.range(0, NUM_GROUPS):
                acc += _accumulate_group(
                    A,
                    B,
                    row_index,
                    group * group_span,
                    M,
                    N,
                    stride_an,
                    stride_am,
                    stride_bm,
                    BLOCK_N,
                    BLOCK_M,
                    GROUP_ITERS,
                    EVEN,
                )
        vector = tl.sum(acc, axis=1)
        if BIAS_IS_SCALAR:
            inp = tl.load(Inp).to(tl.float32)
        else:
            inp = tl.load(
                Inp + row_index * stride_in, mask=row_index < N, other=0.0
            ).to(tl.float32)
        if EVEN:
            tl.store(Out + row_index * stride_outn, vector * alpha + inp * beta)
        else:
            tl.store(
                Out + row_index * stride_outn,
                vector * alpha + inp * beta,
                mask=row_index < N,
            )


@libentry()
@triton.jit(do_not_specialize=["M", "N", "alpha", "beta"])
def addmv_kernel_small(
    A,
    B,
    Inp,
    Out,
    N,
    M,
    alpha,
    beta,
    stride_an,
    stride_am,
    stride_bm,
    stride_in,
    stride_outn,
    BIAS_IS_SCALAR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_M_ITERS: tl.constexpr,
):
    """Row-stream kernel for small N: one program handles a full row, K statically unrolled."""
    num_programs = tl.num_programs(0)
    offs_m = tl.arange(0, BLOCK_M)
    for row in tl.range(tl.program_id(0), N, num_programs):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for it in tl.static_range(NUM_M_ITERS):
            k_index = it * BLOCK_M + offs_m
            k_mask = k_index < M
            a = tl.load(
                A + row * stride_an + k_index * stride_am, mask=k_mask, other=0.0
            )
            b = tl.load(B + k_index * stride_bm, mask=k_mask, other=0.0)
            acc += a * b
        reduced = tl.sum(acc, axis=0)
        if BIAS_IS_SCALAR:
            inp = tl.load(Inp).to(tl.float32)
        else:
            inp = tl.load(Inp + row * stride_in).to(tl.float32)
        tl.store(Out + row * stride_outn, reduced * alpha + inp * beta)


def _next_pow2(value):
    return int(triton.next_power_of_2(max(int(value), 1)))


def _launch_plan(num_rows, reduction_size, dtype, bias_is_scalar):
    """Caches the tier by (rows, K, dtype, scalar_bias) to avoid recomputing pow2/cdiv and branches:

    small shapes take only 5~12us end to end, so ~7us of host arithmetic must be cached away."""
    key = (num_rows, reduction_size, dtype, bias_is_scalar)
    plan = _LAUNCH_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    if num_rows <= SMALL_N_MAX:
        block_m = min(SMALL_BLOCK_M_MAX, _next_pow2(reduction_size))
        num_m_iters = triton.cdiv(reduction_size, block_m)
        if num_m_iters <= SMALL_M_ITERS_MAX:
            # Small-N row-streaming kernel: ("small", grid, BLOCK_M, NUM_M_ITERS)
            plan = ("small", (min(NUM_SIPS, num_rows),), block_m, num_m_iters)
            _LAUNCH_PLAN_CACHE[key] = plan
            return plan
    block_m = min(BLOCK_M, _next_pow2(reduction_size))
    num_m_iters = triton.cdiv(reduction_size, block_m)
    group_iters = min(GROUP_ITERS, num_m_iters)
    num_groups = triton.cdiv(num_m_iters, group_iters)
    # Narrow BLOCK_N when multiple K groups are needed (K > GROUP_ITERS*BLOCK_M):
    # (1024,65536) fp16 bn16 97.0us vs bn32 133.8us; fp32 has a tile twice as wide and must also be narrowed even for a
    # single group
    # ((4096,4096) fp32 bn16 39.0us vs bn32 47.6us).
    narrow = num_m_iters > GROUP_ITERS or (
        dtype == torch.float32 and num_m_iters >= GROUP_ITERS
    )
    block_n = min(16 if narrow else BLOCK_N, _next_pow2(num_rows))
    # The last group may run past M, in which case the masked branch must be taken
    even = (num_rows % block_n == 0) and (
        num_groups * group_iters * block_m == reduction_size
    )
    # Row kernel: ("row", grid, BLOCK_N, BLOCK_M, GROUP_ITERS, NUM_GROUPS, EVEN)
    plan = (
        "row",
        (min(NUM_SIPS, triton.cdiv(num_rows, block_n)),),
        block_n,
        block_m,
        group_iters,
        num_groups,
        even,
    )
    _LAUNCH_PLAN_CACHE[key] = plan
    return plan


def _launch_addmv(self, mat, vec, out, beta, alpha):
    """Shared launch for addmv / addmv_out / addmv_; they differ only in how out is obtained."""
    num_rows, reduction_size = mat.shape
    bias_is_scalar = self.numel() == 1
    plan = _launch_plan(num_rows, reduction_size, mat.dtype, bias_is_scalar)
    # When the shapes match, self already has the result layout, saving one broadcast_to view creation
    bias = self if self.shape == out.shape else self.broadcast_to(out.shape)
    with torch_device_fn.device(mat.device):
        if plan[0] == "small":
            _, grid, block_m, num_m_iters = plan
            addmv_kernel_small[grid](
                mat,
                vec,
                bias,
                out,
                num_rows,
                reduction_size,
                alpha,
                beta,
                mat.stride(0),
                mat.stride(1),
                vec.stride(0),
                bias.stride(0),
                out.stride(0),
                BIAS_IS_SCALAR=bias_is_scalar,
                BLOCK_M=block_m,
                NUM_M_ITERS=num_m_iters,
                num_warps=1,
            )
        else:
            _, grid, block_n, block_m, group_iters, num_groups, even = plan
            addmv_kernel[grid](
                mat,
                vec,
                bias,
                out,
                num_rows,
                reduction_size,
                alpha,
                beta,
                mat.stride(0),
                mat.stride(1),
                vec.stride(0),
                bias.stride(0),
                out.stride(0),
                BIAS_IS_SCALAR=bias_is_scalar,
                BLOCK_N=block_n,
                BLOCK_M=block_m,
                GROUP_ITERS=group_iters,
                NUM_GROUPS=num_groups,
                EVEN=even,
                num_warps=1,
            )
    return out


def addmv(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_ENFLAME ADDMV")
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    out = torch.empty((mat.shape[0],), device=mat.device, dtype=mat.dtype)
    if mat.numel() == 0:
        # K=0: the result degenerates to beta*bias, no kernel launch needed
        bias = self.broadcast_to(out.shape)
        out.copy_((bias * beta).to(out.dtype))
        return out
    return _launch_addmv(self, mat, vec, out, beta, alpha)


def addmv_out(self, mat, vec, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_ENFLAME ADDMV_OUT")
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    if out is None:
        out = torch.empty((mat.shape[0],), device=mat.device, dtype=mat.dtype)
    else:
        assert out.shape == (mat.shape[0],), "Incompatible output shape"
    if mat.numel() == 0:
        bias = self.broadcast_to(out.shape)
        out.copy_((bias * beta).to(out.dtype))
        return out
    return _launch_addmv(self, mat, vec, out, beta, alpha)
