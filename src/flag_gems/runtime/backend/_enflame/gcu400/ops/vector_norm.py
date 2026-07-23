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

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim

pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

_min = builtins.min
_max = builtins.max
MAX_GRID = 48


# ============================================================
# Dim-reduction kernels (reduce along last dim after dim_compress)
# ============================================================


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def l2_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0.0).to(tl.float32)
            _sum += a * a

        total = tl.sum(_sum, axis=1)
        out = tl.sqrt(total)[:, None]
        tl.store(Out + m_offset, out, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def max_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0.0).to(tl.float32)
            _max = tl.maximum(tl.abs(a), _max)

        result = tl.max(_max, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def min_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=float("inf")).to(
                tl.float32
            )
            _min = tl.minimum(tl.abs(a), _min)

        result = tl.min(_min, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def l0_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0).to(tl.float32)
            _sum += tl.where(a != 0, 1.0, 0.0)

        result = tl.sum(_sum, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def l1_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0.0).to(tl.float32)
            _sum += tl.abs(a)

        result = tl.sum(_sum, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "ord"])
def v_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    ord,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0.0).to(tl.float32)
            _sum += pow(tl.abs(a), ord)

        total = tl.sum(_sum, axis=1)
        result = pow(total, 1.0 / ord)[:, None]
        tl.store(Out + m_offset, result, row_mask)


# ============================================================
# Global single-pass kernels (loop-based, 1 launch for small/medium M)
# ============================================================


@libentry()
@triton.jit(do_not_specialize=["M"])
def l2_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    tl.store(Out, tl.sqrt(tl.sum(acc)))


@libentry()
@triton.jit(do_not_specialize=["M"])
def max_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc = tl.maximum(tl.abs(x), acc)
    tl.store(Out, tl.max(acc))


@libentry()
@triton.jit(do_not_specialize=["M"])
def min_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.full([BLOCK], value=float("inf"), dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=float("inf")).to(tl.float32)
        acc = tl.minimum(tl.abs(x), acc)
    tl.store(Out, tl.min(acc))


@libentry()
@triton.jit(do_not_specialize=["M"])
def l0_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += (x != 0).to(tl.float32)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit(do_not_specialize=["M"])
def l1_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += tl.abs(x)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit(do_not_specialize=["M", "ord"])
def v_global_single(X, Out, ord, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += pow(tl.abs(x), ord)
    tl.store(Out, pow(tl.sum(acc), 1.0 / ord))


# ============================================================
# Global-reduction kernels (two-stage: partial → final)
# ============================================================


@libentry()
@triton.jit(do_not_specialize=["M"])
def l2_global_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(Mid + tile_id, tl.sum(x * x))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def l2_global_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out, tl.sqrt(tl.sum(mid)))


@libentry()
@triton.jit(do_not_specialize=["M"])
def max_global_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(Mid + tile_id, tl.max(tl.abs(x)))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def max_global_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out, tl.max(mid))


@libentry()
@triton.jit(do_not_specialize=["M"])
def min_global_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=float("inf")).to(tl.float32)
        tl.store(Mid + tile_id, tl.min(tl.abs(x)))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def min_global_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=float("inf")).to(tl.float32)
    tl.store(Out, tl.min(mid))


@libentry()
@triton.jit(do_not_specialize=["M"])
def l0_global_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        cnt = (x != 0).to(tl.float32)
        tl.store(Mid + tile_id, tl.sum(cnt))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def l0_global_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out, tl.sum(mid))


@libentry()
@triton.jit(do_not_specialize=["M"])
def l1_global_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(Mid + tile_id, tl.sum(tl.abs(x)))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def l1_global_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out, tl.sum(mid))


@libentry()
@triton.jit(do_not_specialize=["M", "ord"])
def v_global_kernel_1(
    X, Mid, ord, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(Mid + tile_id, tl.sum(pow(tl.abs(x), ord)))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE", "ord"])
def v_global_kernel_2(Mid, Out, ord, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out, pow(tl.sum(mid), 1.0 / ord))


# ============================================================
# Host dispatch
# ============================================================


def _launch_global_norm(x, out, M, ord_val, dtype):
    SINGLE_PASS_LIMIT = 32768
    SINGLE_BLOCK = 32768

    with torch_device_fn.device(x.device):
        if M <= SINGLE_PASS_LIMIT:
            bs = _max(1024, _min(triton.next_power_of_2(M), SINGLE_BLOCK))
            if ord_val == 2:
                l2_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == float("inf"):
                max_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == -float("inf"):
                min_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == 0:
                l0_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == 1:
                l1_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            else:
                v_global_single[(1,)](x, out, ord_val, M, BLOCK=bs, num_warps=1)
            return

        block_size = 32768 if dtype == torch.float32 else 65536
        mid_size = triton.cdiv(M, block_size)
        grid_size = _min(mid_size, MAX_GRID)
        block_mid = triton.next_power_of_2(mid_size)
        num_stages = 3 if mid_size > grid_size else 1

        mid = torch.empty([mid_size], dtype=torch.float32, device=x.device)

        if ord_val == 2:
            l2_global_kernel_1[(grid_size,)](
                x, mid, M, BLOCK_SIZE=block_size, num_stages=num_stages, num_warps=4
            )
            l2_global_kernel_2[(1,)](mid, out, mid_size, block_mid, num_warps=1)
        elif ord_val == float("inf"):
            max_global_kernel_1[(grid_size,)](
                x, mid, M, BLOCK_SIZE=block_size, num_stages=num_stages, num_warps=4
            )
            max_global_kernel_2[(1,)](mid, out, mid_size, block_mid, num_warps=1)
        elif ord_val == -float("inf"):
            min_global_kernel_1[(grid_size,)](
                x, mid, M, BLOCK_SIZE=block_size, num_stages=num_stages, num_warps=4
            )
            min_global_kernel_2[(1,)](mid, out, mid_size, block_mid, num_warps=1)
        elif ord_val == 0:
            l0_global_kernel_1[(grid_size,)](
                x, mid, M, BLOCK_SIZE=block_size, num_stages=num_stages, num_warps=4
            )
            l0_global_kernel_2[(1,)](mid, out, mid_size, block_mid, num_warps=1)
        elif ord_val == 1:
            l1_global_kernel_1[(grid_size,)](
                x, mid, M, BLOCK_SIZE=block_size, num_stages=num_stages, num_warps=4
            )
            l1_global_kernel_2[(1,)](mid, out, mid_size, block_mid, num_warps=1)
        else:
            v_global_kernel_1[(grid_size,)](
                x,
                mid,
                ord_val,
                M,
                BLOCK_SIZE=block_size,
                num_stages=num_stages,
                num_warps=4,
            )
            v_global_kernel_2[(1,)](mid, out, ord_val, mid_size, block_mid, num_warps=1)


def _launch_dim_norm(x, out, M, N, ord_val):
    BLOCK_N = _min(triton.next_power_of_2(N), 2048)
    BLOCK_M = _max(1, _min(128, 32768 // BLOCK_N))
    grid_m = _min(triton.cdiv(M, BLOCK_M), MAX_GRID)

    with torch_device_fn.device(x.device):
        if ord_val == 2:
            l2_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        elif ord_val == float("inf"):
            max_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        elif ord_val == -float("inf"):
            min_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        elif ord_val == 0:
            l0_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        elif ord_val == 1:
            l1_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        else:
            v_norm_dim_kernel[(grid_m,)](
                x, out, M, N, ord_val, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_ENFLAME VECTOR_NORM")
    if dtype is not None:
        dtype = torch.dtype(dtype)
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    if (not dim) or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        x = dim_compress(x, dim)
        M = x.numel()
        out = torch.empty(shape, dtype=dtype, device=x.device)
        _launch_global_norm(x, out, M, ord, dtype)
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        out = torch.empty(shape, dtype=dtype, device=x.device)
        _launch_dim_norm(x, out, M, N, ord)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
