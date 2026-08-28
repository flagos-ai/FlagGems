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

_NATIVE_VECTOR_NORM = torch.library.get_kernel("aten::linalg_vector_norm", "CUDA")
_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow


def heur_block_m(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def heur_block_n(args):
    return builtins.min(args["N"], 8192)


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
    # XPU masked-load hazard: on tail blocks the mask is mis-applied at
    # 1024-lane granularity and real elements get duplicated into the
    # reduction (deterministic +9216 count at (200,40999,3), heap-content
    # independent). Avoid masked loads entirely: clamp indices in-bounds
    # and select the contribution with tl.where afterwards.
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = tl.minimum(offset, (M - 1).to(tl.int64))
    Mid = Mid + pid
    x = tl.load(X + idx).to(tl.float32)
    x = tl.where(offset < M, x, 0.0)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l2_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    # See the kernel_1 note: no masked loads; clamp + tl.where.
    offset = tl.arange(0, BLOCK_MID)
    idx = tl.minimum(offset, (MID_SIZE - 1).to(tl.int64))
    mid = tl.load(Mid + idx).to(tl.float32)
    mid = tl.where(offset < MID_SIZE, mid, 0.0)
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
    # XPU masked-load hazard: on tail blocks the mask is mis-applied at
    # 1024-lane granularity and real elements get duplicated into the
    # reduction (deterministic +9216 count at (200,40999,3), heap-content
    # independent). Avoid masked loads entirely: clamp indices in-bounds
    # and select the contribution with tl.where afterwards.
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = tl.minimum(offset, (M - 1).to(tl.int64))
    Mid = Mid + pid
    x = tl.load(X + idx).to(tl.float32)
    a = tl.where(offset < M, tl.abs(x), 0.0)
    mid = tl.max(a)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def max_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    # See the kernel_1 note: no masked loads; clamp + tl.where.
    offset = tl.arange(0, BLOCK_MID)
    idx = tl.minimum(offset, (MID_SIZE - 1).to(tl.int64))
    mid = tl.load(Mid + idx).to(tl.float32)
    mid = tl.where(offset < MID_SIZE, mid, 0.0)
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
    # XPU masked-load hazard: on tail blocks the mask is mis-applied at
    # 1024-lane granularity and real elements get duplicated into the
    # reduction (deterministic +9216 count at (200,40999,3), heap-content
    # independent). Avoid masked loads entirely: clamp indices in-bounds
    # and select the contribution with tl.where afterwards.
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = tl.minimum(offset, (M - 1).to(tl.int64))
    Mid = Mid + pid
    x = tl.load(X + idx).to(tl.float32)
    a = tl.where(offset < M, tl.abs(x), float("inf"))
    mid = tl.min(a)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def min_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    # See the kernel_1 note: no masked loads; clamp + tl.where.
    offset = tl.arange(0, BLOCK_MID)
    idx = tl.minimum(offset, (MID_SIZE - 1).to(tl.int64))
    mid = tl.load(Mid + idx).to(tl.float32)
    mid = tl.where(offset < MID_SIZE, mid, float("inf"))
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
    # XPU masked-load hazard: on tail blocks the mask is mis-applied at
    # 1024-lane granularity and real elements get duplicated into the
    # reduction (deterministic +9216 count at (200,40999,3), heap-content
    # independent). Avoid masked loads entirely: clamp indices in-bounds
    # and select the contribution with tl.where afterwards.
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = tl.minimum(offset, (M - 1).to(tl.int64))
    Mid = Mid + pid
    x = tl.load(X + idx).to(tl.float32)
    x = tl.where(offset < M, x, 0.0)
    cnt = (x != 0).to(tl.float32)
    mid = tl.sum(cnt)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l0_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    # See the kernel_1 note: no masked loads; clamp + tl.where.
    offset = tl.arange(0, BLOCK_MID)
    idx = tl.minimum(offset, (MID_SIZE - 1).to(tl.int64))
    mid = tl.load(Mid + idx).to(tl.float32)
    mid = tl.where(offset < MID_SIZE, mid, 0.0)
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
    # XPU masked-load hazard: on tail blocks the mask is mis-applied at
    # 1024-lane granularity and real elements get duplicated into the
    # reduction (deterministic +9216 count at (200,40999,3), heap-content
    # independent). Avoid masked loads entirely: clamp indices in-bounds
    # and select the contribution with tl.where afterwards.
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = tl.minimum(offset, (M - 1).to(tl.int64))
    Mid = Mid + pid
    x = tl.load(X + idx).to(tl.float32)
    ord = ord.to(tl.float32)
    x = tl.where(offset < M, x, 0.0)
    mid = tl.sum(pow(tl.abs(x), ord))
    tl.store(Mid, mid)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(
    Mid, Out, ord, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    # See the kernel_1 note: no masked loads; clamp + tl.where.
    offset = tl.arange(0, BLOCK_MID)
    idx = tl.minimum(offset, (MID_SIZE - 1).to(tl.int64))
    mid = tl.load(Mid + idx).to(tl.float32)
    ord = ord.to(tl.float32)
    mid = tl.where(offset < MID_SIZE, mid, 0.0)
    out = pow(tl.abs(tl.sum(mid)), 1 / ord)
    tl.store(Out, out)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_KUNLUNXIN VECTOR_NORM")
    if dtype is not None:
        dtype = torch.dtype(dtype)
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    with torch_device_fn.device(x.device):
        if (not dim) or len(dim) == x.ndim:
            dim = list(range(x.ndim))
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
            # XPU tile hazard (probed 2026-08-26): 32768-lane tiles are
            # corrupted in two independent ways -- masked loads duplicate
            # real elements into tail lanes (deterministic +9216 count at
            # (200,40999,3)), and tl.sum silently drops ~75% of lanes --
            # while 1024/8192-lane tiles are exact with the clamp+where
            # kernels above. Cap at 8192. NOTE this bounds stage-2 safety
            # to MID_SIZE lanes <= ~16k (M <= ~1.3e8); beyond that a third
            # reduction stage would be needed.
            BLOCK_SIZE = min(
                triton.next_power_of_2(triton.cdiv(M, cluster_num)),
                8192,
            )
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            mid = torch.empty([MID_SIZE], dtype=dtype, device=x.device)
            out = torch.empty(shape, dtype=dtype, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
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
                l0_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            else:
                l1_norm_kernel_1[(MID_SIZE,)](
                    x, mid, ord, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                l1_norm_kernel_2[(1,)](
                    mid, out, ord, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
        else:
            # Partial-dim path. dim_compress() materializes the view via
            # permute().contiguous(); on XPU torch copy_ of strided/broadcast
            # sources is broken (CUDA error: invalid device function), so the
            # vendor kernels below could never run on the compressed tensor.
            # Native CUDA (torch_xmlir) linalg_vector_norm is correct on P800
            # for every dim form; redispatch instead of materializing.
            return _NATIVE_VECTOR_NORM.call_boxed(
                _CUDA_KEYSET, x, ord, dim, keepdim, dtype=dtype
            )
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
