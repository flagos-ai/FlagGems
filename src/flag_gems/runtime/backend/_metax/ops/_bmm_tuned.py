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

"""Static MetaX GEMM configurations derived from measured candidates.

The public dispatcher admits FP32 only when TF32 is explicitly enabled.
Other inputs retain the existing implementation.
"""

import torch
import triton
import triton.language as tl

_PREC_HW = 0
_PREC_IEEE = 1
_PREC_TF32 = 2
_PREC_SPLIT2 = 3
_TF32_MIN_K = 2048
_SPLIT2_MIN_K = 768
_HALF_WIDE = (128, 256, 16, 2, 8, 4, _PREC_HW)
_HALF_NARROW = (64, 64, 32, 1, 4, 3, _PREC_HW)
_FP32_TF32_WIDE = (128, 128, 32, 2, 8, 4, _PREC_TF32)
_FP32_TF32_NARROW = (64, 64, 64, 1, 4, 3, _PREC_TF32)
_FP32_SPLIT2_WIDE = (128, 128, 32, 2, 8, 3, _PREC_SPLIT2)
_FP32_SPLIT2_NARROW = (64, 64, 32, 1, 4, 3, _PREC_SPLIT2)
_FP32_IEEE = (32, 32, 32, 1, 4, 3, _PREC_IEEE)
_FP64_CFG = (64, 64, 32, 1, 4, 2, _PREC_IEEE)
_CPASYNC_SHAPES = frozenset(
    (
        (_HALF_WIDE[0], _HALF_WIDE[1], _HALF_WIDE[2]),
        (_HALF_NARROW[0], _HALF_NARROW[1], _HALF_NARROW[2]),
    )
)
_CTA_FLOOR_FALLBACK = 256
_SM_COUNT_CACHE = {}


def _sm_count(device):
    index = device.index
    if index is None:
        try:
            index = torch.cuda.current_device()
        except Exception:
            return 0
    key = int(index)
    count = _SM_COUNT_CACHE.get(key)
    if count is None:
        try:
            count = int(torch.cuda.get_device_properties(key).multi_processor_count)
            if count < 1:
                count = 0
        except Exception:
            count = 0
        _SM_COUNT_CACHE[key] = count
    return count


def _pick_config(dtype, batch, M, N, K, device):
    if dtype is torch.float64:
        return _FP64_CFG
    if dtype is torch.float32:
        if K >= _TF32_MIN_K:
            (wide, narrow) = (_FP32_TF32_WIDE, _FP32_TF32_NARROW)
        elif K >= _SPLIT2_MIN_K:
            (wide, narrow) = (_FP32_SPLIT2_WIDE, _FP32_SPLIT2_NARROW)
        else:
            return _FP32_IEEE
    else:
        (wide, narrow) = (_HALF_WIDE, _HALF_NARROW)
    sm = _sm_count(device)
    floor = 2 * sm if sm > 0 else _CTA_FLOOR_FALLBACK
    n_cta = triton.cdiv(M, wide[0]) * triton.cdiv(N, wide[1]) * batch
    if n_cta >= floor:
        return wide
    return narrow


@triton.jit
def _bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIV_M: tl.constexpr,
    DIV_N: tl.constexpr,
    DIV_K: tl.constexpr,
    PREC: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    if GROUP_M == 1:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    else:
        num_pid_m = tl.num_programs(0)
        num_pid_n = tl.num_programs(1)
        pid = tl.program_id(0) + tl.program_id(1) * num_pid_m
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
        pid_n = pid % num_pid_in_group // group_size_m
    pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        A
        + pid_b * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    if DIV_M:
        mask_m = None
    else:
        mask_m = offs_m < M
    if DIV_N:
        mask_n = None
    else:
        mask_n = offs_n < N
    if IS_FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
        acc2 = acc
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if PREC == 3:
            acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        else:
            acc2 = acc
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        if DIV_K:
            if DIV_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if DIV_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            mask_k = offs_k < K
            if DIV_M:
                a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            if DIV_N:
                b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        if IS_FP64:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        elif PREC == 0:
            acc = tl.dot(a, b, acc)
        elif PREC == 1:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        elif PREC == 2:
            acc = tl.dot(a, b, acc, input_precision="tf32")
        else:
            a_hi = (a.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
            b_hi = (b.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
            acc = tl.dot(a_hi, b_hi, acc, input_precision="tf32")
            acc2 = tl.dot(a - a_hi, b_hi, acc2, input_precision="tf32")
    if PREC == 3:
        acc += acc2
    o_ptrs = (
        O
        + pid_b * stride_ob
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    if DIV_M and DIV_N:
        tl.store(o_ptrs, acc.to(O.dtype.element_ty))
    elif DIV_M:
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_n[None, :])
    elif DIV_N:
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.store(
            o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :]
        )


def tuned_bmm(self, mat2, out=None):
    a = self
    b = mat2
    (batch, M, K) = a.shape
    N = b.shape[2]
    if out is None:
        out = torch.empty((batch, M, N), dtype=a.dtype, device=a.device)
    if batch == 0 or M == 0 or N == 0 or (K == 0):
        return out
    (block_m, block_n, block_k, group_m, num_warps, num_stages, prec) = _pick_config(
        a.dtype, batch, M, N, K, a.device
    )
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), batch)
    launch_kwargs = {}
    div_m = M % block_m == 0
    div_n = N % block_n == 0
    div_k = K % block_k == 0
    if (
        prec == _PREC_HW
        and div_m
        and div_n
        and div_k
        and ((block_m, block_n, block_k) in _CPASYNC_SHAPES)
    ):
        launch_kwargs["pipeline"] = "cpasync"
    _bmm_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        DIV_M=div_m,
        DIV_N=div_n,
        DIV_K=div_k,
        PREC=prec,
        IS_FP64=a.dtype is torch.float64,
        num_warps=num_warps,
        num_stages=num_stages,
        **launch_kwargs,
    )
    return out
