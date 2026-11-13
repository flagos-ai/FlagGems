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
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# namedtuple 类构造约 87us，逐次调用会压过小 shape 的 kernel 时间，提到模块级复用。
MaxOut = namedtuple("max", ["values", "indices"])


@libentry()
@triton.jit
def max_kernel_1_simple(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=min_value)
    tl.store(mid + pid, tl.max(inp_val))


@libentry()
@triton.jit(do_not_specialize=["M"])
def max_kernel_1_grid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    dtype = inp.type.element_ty
    min_val = get_dtype_min(dtype)

    acc = tl.full([BLOCK_SIZE], value=min_val, dtype=dtype)

    block_id = pid
    while block_id * BLOCK_SIZE < M:
        offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        vals = tl.load(inp + offset, mask=mask, other=min_val)
        acc = tl.where(vals > acc, vals, acc)
        block_id += num_progs

    tl.store(mid + pid, tl.max(acc))


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def max_kernel_inner_1d(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    if tl.constexpr(dtype == tl.float16) | tl.constexpr(dtype == tl.bfloat16):
        # max only compares and selects values, so keeping f16/bf16 instead
        # of widening to f32 is bit-exact.  It also avoids a predicate that is
        # computed through an f16 -> f32 widening and then drives the i32 index
        # select, which GCU400 miscompiles: on (4096, 256) dim=1 the value is
        # right but the index points at a non-maximum element for ~46% of the
        # rows.  fp32 input, which is not widened, is unaffected.
        acc_dtype = dtype
    elif tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype

    for row_id in tl.range(pid, M, step, num_stages=num_stages):
        max_vals = tl.full([BLOCK_N], value=get_dtype_min(acc_dtype), dtype=acc_dtype)
        max_idxs = tl.zeros([BLOCK_N], dtype=tl.int32)

        base = row_id * N
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            mask = n_offset < N
            x = tl.load(inp + base + n_offset, mask=mask, other=min_value).to(acc_dtype)

            update = x > max_vals
            max_vals = tl.where(update, x, max_vals)
            max_idxs = tl.where(update, n_offset, max_idxs)

        final_max = tl.max(max_vals, axis=0)
        eq_mask = max_vals == final_max
        max_idxs_masked = tl.where(eq_mask, max_idxs, 2147483647)
        final_idx = tl.min(max_idxs_masked, axis=0)

        tl.store(out_value + row_id, final_max.to(dtype))
        tl.store(out_index + row_id, final_idx)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K", "num_k_tiles"])
def max_kernel_non_inner(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    num_k_tiles,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    total_work = M * num_k_tiles

    dtype = inp.type.element_ty
    if tl.constexpr(dtype == tl.float16) | tl.constexpr(dtype == tl.bfloat16):
        # See max_kernel_inner_1d: bit-exact for max, and avoids the predicate
        # computed through an f16 -> f32 widening that GCU400 miscompiles.
        acc_dtype = dtype
    elif tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype
    min_value = get_dtype_min(acc_dtype)
    orig_min = get_dtype_min(dtype)

    for work_id in tl.range(pid, total_work, num_prog, num_stages=num_stages):
        m = work_id // num_k_tiles
        k_tile = work_id % num_k_tiles
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        max_vals = tl.full([BLOCK_K], dtype=acc_dtype, value=min_value)
        max_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)

        base = m * N * K
        for n_off in tl.range(0, N, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            offset = base + n_idx[:, None] * K + k_offset[None, :]
            mask = (n_idx[:, None] < N) & k_mask[None, :]
            vals = tl.load(inp + offset, mask=mask, other=orig_min).to(acc_dtype)

            # tl.max promotes f16/bf16 to f32; cast back so the index select
            # below still sees an f16 predicate.  The result is one of the
            # inputs, so this is exact.
            local_max = tl.max(vals, axis=0).to(acc_dtype)
            eq = vals == local_max[None, :]
            idx_matrix = tl.where(eq, n_idx[:, None], 2147483647)
            local_argmax = tl.min(idx_matrix, axis=0)

            update = local_max > max_vals
            max_vals = tl.where(update, local_max, max_vals)
            max_idxs = tl.where(update, local_argmax, max_idxs)

        out_offset = m * K + k_offset
        tl.store(out_value + out_offset, max_vals.to(dtype), k_mask)
        tl.store(out_index + out_offset, max_idxs, k_mask)


def max(inp):
    logger.debug("GEMS_ENFLAME MAX")

    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    if M <= 10 * 1024 * 1024:
        bsize = 16384
        num_programs = min(triton.cdiv(M, bsize), 48)
        block_mid = triton.next_power_of_2(num_programs)
        mid = torch.empty((num_programs,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            max_kernel_1_grid[(num_programs,)](
                inp, mid, M, BLOCK_SIZE=bsize, num_warps=1
            )
            max_kernel_2[(1,)](mid, out, num_programs, block_mid, num_warps=1)
    else:
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            max_kernel_1_simple[(mid_size, 1, 1)](inp, mid, M, block_size, num_warps=1)
            max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, num_warps=1)
    return out


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME MAX_DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim

    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    if not inp.is_contiguous():
        inp = inp.contiguous()

    shape_list = list(shape)
    if keepdim:
        shape_list[dim] = 1
    else:
        # 两个 kernel 都按扁平的 m*K+k 布局写回，直接分配最终形状即可，
        # 避免两次 torch.squeeze 带来的设备同步与 Python 开销。
        shape_list.pop(dim)
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    # GCU400 Triton 的 int32 索引存储每个 lane 只写 4 字节，若直接写入 int64
    # 输出张量会得到大量 0；先用 int32 缓冲承接 kernel 写回，再统一转成 int64。
    out_index = torch.empty(shape_list, dtype=torch.int32, device=inp.device)

    if K == 1:
        BLOCK_N = min(triton.next_power_of_2(N), 4096)
        grid_m = min(M, 48)
        num_stages = 3 if M > grid_m else 1
        with torch_device_fn.device(inp.device):
            max_kernel_inner_1d[(grid_m,)](
                inp,
                out_value,
                out_index,
                M,
                N,
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=1,
            )
    else:
        # BLOCK_K is the width of the loop-carried accumulator (output
        # columns; the reduction walks BLOCK_N).  GCU400 keeps such a vector
        # in at most 512B of OACC: 128 int64 lanes need 1024B and the
        # reduction then silently covers only part of the axis -- the value is
        # the max over a subset and the index matches that wrong value.
        # int64 dim=0 on (4096, 256) gets 219/256 columns wrong, while int32
        # at the same 128 lanes (512B) is exact.  The wide tile is also ~12x
        # slower.
        _acc_bpes = {
            torch.float64: 8,
            torch.int64: 8,
            torch.float32: 4,
            torch.int32: 4,
            torch.int16: 4,
            torch.int8: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }
        BLOCK_K = min(
            triton.next_power_of_2(K), min(512 // _acc_bpes.get(inp.dtype, 8), 128)
        )
        BLOCK_N = min(triton.next_power_of_2(N), 256)

        num_k_tiles = triton.cdiv(K, BLOCK_K)
        total_work = M * num_k_tiles
        grid_size = min(total_work, 48)

        with torch_device_fn.device(inp.device):
            max_kernel_non_inner[(grid_size,)](
                inp,
                out_value,
                out_index,
                M,
                N,
                K,
                num_k_tiles,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_N,
                num_stages=1,
                num_warps=1,
            )

    return MaxOut(values=out_value, indices=out_index.to(torch.int64))
