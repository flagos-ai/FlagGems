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

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow


def heur_block_m(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def heur_block_n(args):
    return builtins.min(args["N"], 8192)


@libentry()
@triton.jit
def zero_workspace_kernel(X, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(X + offsets, 0.0)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a * a
    sum = tl.sum(_sum, axis=1)

    out = tl.sqrt(sum)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def l2_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l2_norm_tail_kernel(
    X,
    Mid,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    MID_INDEX: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    total = 0.0
    full_size = (TAIL_SIZE // 8) * 8
    for offset in tl.range(0, full_size, 8):
        values = tl.load(X + TAIL_OFFSET + offset + tl.arange(0, 8)).to(tl.float32)
        total += tl.sum(values * values)
    for offset in tl.static_range(TAIL_SIZE % 8):
        value = tl.load(X + TAIL_OFFSET + full_size + offset).to(tl.float32)
        total += value * value
    tl.store(Mid + MID_INDEX, total)


@libentry()
@triton.jit
def l2_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def max_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _max = tl.maximum(tl.abs(a), _max)

    max = tl.max(_max, axis=1)
    out = max[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def max_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.max(tl.abs(x))
    tl.store(Mid, mid)


@libentry()
@triton.jit
def max_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.max(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def min_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=float("inf")).to(tl.float32)
        _min = tl.minimum(tl.abs(a), _min)

    min = tl.min(_min, axis=1)
    out = min[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def min_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=float("inf")).to(tl.float32)
    mid = tl.min(tl.abs(x))
    tl.store(Mid, mid)


@libentry()
@triton.jit
def min_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=float("inf")).to(tl.float32)
    out = tl.min(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0).to(tl.float32)
        _sum += tl.where(a != 0, 1, 0)
    sum = tl.sum(_sum, axis=1)
    out = sum[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def l0_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    cnt = (x != 0).to(tl.float32)
    mid = tl.sum(cnt)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l0_norm_tail_kernel(
    X,
    Mid,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    MID_INDEX: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    total = 0.0
    full_size = (TAIL_SIZE // 8) * 8
    for offset in tl.range(0, full_size, 8):
        values = tl.load(X + TAIL_OFFSET + offset + tl.arange(0, 8)).to(tl.float32)
        total += tl.sum((values != 0).to(tl.float32))
    for offset in tl.static_range(TAIL_SIZE % 8):
        value = tl.load(X + TAIL_OFFSET + full_size + offset).to(tl.float32)
        total += (value != 0).to(tl.float32)
    tl.store(Mid + MID_INDEX, total)


@libentry()
@triton.jit
def l0_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sum(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ord = ord.to(tl.float32)
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += pow(tl.abs(a), ord)
    sum = tl.sum(_sum, axis=1)
    out = pow(sum, 1 / ord)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_1(
    X, Mid, ord, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    ord = ord.to(tl.float32)
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(pow(tl.abs(x), ord))
    tl.store(Mid, mid)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_tail_kernel(
    X,
    Mid,
    ord,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    MID_INDEX: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    ord = ord.to(tl.float32)
    total = 0.0
    full_size = (TAIL_SIZE // 8) * 8
    for offset in tl.range(0, full_size, 8):
        values = tl.load(X + TAIL_OFFSET + offset + tl.arange(0, 8)).to(tl.float32)
        total += tl.sum(pow(tl.abs(values), ord))
    for offset in tl.static_range(TAIL_SIZE % 8):
        value = tl.load(X + TAIL_OFFSET + full_size + offset).to(tl.float32)
        total += pow(tl.abs(value), ord)
    tl.store(Mid + MID_INDEX, total)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(
    Mid, Out, ord, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    ord = ord.to(tl.float32)
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = pow(tl.sum(mid), 1 / ord)
    tl.store(Out, out)


@libentry()
@triton.jit
def l1_norm_rows_kernel_1(
    X,
    Mid,
    M,
    N,
    MID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
    row = pid // MID_SIZE
    chunk = pid % MID_SIZE
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        X + row * N + offsets,
        mask=(row < M) & (offsets < N),
        other=0.0,
    ).to(tl.float32)
    tl.store(Mid + row * MID_SIZE + chunk, tl.sum(tl.abs(values)), mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_tail_kernel(
    X,
    Mid,
    M,
    N,
    MID_SIZE: tl.constexpr,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    row = ext.program_id(0).to(tl.int64)
    total = 0.0
    for offset in tl.static_range(TAIL_SIZE):
        value = tl.load(X + row * N + TAIL_OFFSET + offset).to(tl.float32)
        total += tl.abs(value)
    tl.store(Mid + row * MID_SIZE + MID_SIZE - 1, total, mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_kernel_2(
    Mid,
    Next,
    M,
    MID_SIZE: tl.constexpr,
    NEXT_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
    row = pid // NEXT_SIZE
    chunk = pid % NEXT_SIZE
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    partial = tl.load(
        Mid + row * MID_SIZE + offsets,
        mask=(row < M) & (offsets < MID_SIZE),
        other=0.0,
    ).to(tl.float32)
    tl.store(Next + row * NEXT_SIZE + chunk, tl.sum(partial), mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_reduce_tail_kernel(
    Mid,
    Next,
    M,
    MID_SIZE: tl.constexpr,
    NEXT_SIZE: tl.constexpr,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    row = ext.program_id(0).to(tl.int64)
    total = 0.0
    for offset in tl.static_range(TAIL_SIZE):
        total += tl.load(Mid + row * MID_SIZE + TAIL_OFFSET + offset).to(tl.float32)
    tl.store(Next + row * NEXT_SIZE + NEXT_SIZE - 1, total, mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_kernel_3(
    Next,
    Out,
    M,
    NEXT_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    row = ext.program_id(0).to(tl.int64)
    total = 0.0
    for offset in tl.static_range(NEXT_SIZE):
        total += tl.load(Next + row * NEXT_SIZE + offset).to(tl.float32)
    tl.store(Out + row, total, mask=row < M)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_KUNLUNXIN VECTOR_NORM")
    if dtype is None:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    if dim is None:
        dim = list(range(x.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    normalized_dim = []
    for d in dim:
        if d < -x.ndim or d >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-x.ndim}, {x.ndim - 1}], but got {d})"
            )
        normalized_dim.append(d % x.ndim)
    if len(set(normalized_dim)) != len(normalized_dim):
        raise RuntimeError("dim must contain all distinct dimensions")
    dim = normalized_dim

    with torch_device_fn.device(x.device):
        if len(dim) == x.ndim:
            shape = [1] * x.ndim
            x = dim_compress(x, dim)
            M = x.numel()
            cluster_num = 12
            # XPU: tl.sum over a 1D tile is only correct up to a bounded lane
            # count. Empirically BLOCK_SIZE=32768 (bsl=2048) is the safe max;
            # a larger tile silently drops lanes. Cap here (dtype-independent)
            # keeps stage-1 tiles correct AND bounds MID_SIZE <= 32768 for all
            # M <= 2**30 so stage-2's tl.sum(mid) is also within the safe range.
            # The old cap int(1024*64/element_size) gave 16384 for fp32 -> for
            # M=2**30 MID_SIZE=65536 which broke stage-2 (wrong fp32 results).
            BLOCK_SIZE = min(
                triton.next_power_of_2(triton.cdiv(M, cluster_num)),
                32768,
            )
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            # Stage-2 reduces a power-of-two tile. Pad and explicitly clear its
            # workspace so XPU masked loads never consume memory past MID_SIZE.
            mid = torch.empty([BLOCK_MID], dtype=torch.float32, device=x.device)
            zero_workspace_kernel[(1,)](mid, BLOCK_MID)
            out = torch.empty(shape, dtype=dtype, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                tail_size = M % BLOCK_SIZE
                if tail_size:
                    l2_norm_tail_kernel[(1,)](
                        x,
                        mid,
                        M - tail_size,
                        tail_size,
                        MID_SIZE - 1,
                        buffer_size_limit=2048,
                    )
                l2_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == float("inf"):
                max_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                max_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == -float("inf"):
                min_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                min_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == 0:
                l0_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                tail_size = M % BLOCK_SIZE
                if tail_size:
                    l0_norm_tail_kernel[(1,)](
                        x,
                        mid,
                        M - tail_size,
                        tail_size,
                        MID_SIZE - 1,
                        buffer_size_limit=2048,
                    )
                l0_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            else:
                l1_norm_kernel_1[(MID_SIZE,)](
                    x, mid, ord, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                tail_size = M % BLOCK_SIZE
                if tail_size:
                    l1_norm_tail_kernel[(1,)](
                        x,
                        mid,
                        ord,
                        M - tail_size,
                        tail_size,
                        MID_SIZE - 1,
                        buffer_size_limit=2048,
                    )
                l1_norm_kernel_2[(1,)](
                    mid, out, ord, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
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
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            if ord == 2:
                l2_norm_kernel[grid](x, out, M, N)
            elif ord == float("inf"):
                max_norm_kernel[grid](x, out, M, N)
            elif ord == -float("inf"):
                min_norm_kernel[grid](x, out, M, N)
            elif ord == 0:
                l0_norm_kernel[grid](x, out, M, N)
            elif ord == 1 and N > 1024:
                BLOCK_SIZE = 1024
                MID_SIZE = triton.cdiv(N, BLOCK_SIZE)
                mid = torch.empty((M, MID_SIZE), dtype=torch.float32, device=x.device)
                l1_norm_rows_kernel_1[(M * MID_SIZE,)](
                    x,
                    mid,
                    M,
                    N,
                    MID_SIZE,
                    BLOCK_SIZE,
                    buffer_size_limit=2048,
                )
                tail_size = N % BLOCK_SIZE
                if tail_size:
                    l1_norm_rows_tail_kernel[(M,)](
                        x,
                        mid,
                        M,
                        N,
                        MID_SIZE,
                        N - tail_size,
                        tail_size,
                        buffer_size_limit=2048,
                    )
                if MID_SIZE <= 1024:
                    l1_norm_rows_kernel_3[(M,)](
                        mid,
                        out,
                        M,
                        MID_SIZE,
                        triton.next_power_of_2(MID_SIZE),
                        buffer_size_limit=2048,
                    )
                else:
                    NEXT_SIZE = triton.cdiv(MID_SIZE, BLOCK_SIZE)
                    BLOCK_NEXT = triton.next_power_of_2(NEXT_SIZE)
                    next_mid = torch.empty(
                        (M, NEXT_SIZE), dtype=torch.float32, device=x.device
                    )
                    l1_norm_rows_kernel_2[(M * NEXT_SIZE,)](
                        mid,
                        next_mid,
                        M,
                        MID_SIZE,
                        NEXT_SIZE,
                        BLOCK_SIZE,
                        buffer_size_limit=2048,
                    )
                    tail_size = MID_SIZE % BLOCK_SIZE
                    if tail_size:
                        l1_norm_rows_reduce_tail_kernel[(M,)](
                            mid,
                            next_mid,
                            M,
                            MID_SIZE,
                            NEXT_SIZE,
                            MID_SIZE - tail_size,
                            tail_size,
                            buffer_size_limit=2048,
                        )
                    l1_norm_rows_kernel_3[(M,)](
                        next_mid,
                        out,
                        M,
                        NEXT_SIZE,
                        BLOCK_NEXT,
                        buffer_size_limit=2048,
                    )
            else:
                v_norm_kernel[grid](x, out, M, N, ord, isCloseUnrollControl=True)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
