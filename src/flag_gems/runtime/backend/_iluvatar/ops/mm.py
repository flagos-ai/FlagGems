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

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a
    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a
    raise AssertionError("unreachable")


def _to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 15,
    },
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    acc_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    # matrix multiplication
    if UPGRADE:
        pid = tle.program_id(0)
        pid_z = tle.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    # grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # # re-order program ID for better L2 performance
    # width = GROUP_M * grid_n
    # group_id = pid // width
    # group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    # pid_m = group_id * GROUP_M + (pid % group_size)
    # pid_n = (pid % width) // (group_size)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ram = (tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rbn = (tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)).to(tl.int64)
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for k in range(0, loop_num):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


# Minimum tile size from tuning configs.  When any matrix dimension is smaller
# than this, the Triton kernel's tl.multiple_of / tl.max_contiguous hints become
# invalid (they assume the range spans at least BLOCK elements) and the Iluvatar
# compiler may crash or produce wrong results.  We pad the matrices to the
# minimum size and then slice the result.
_MIN_TRITON_DIM = 32

# The Iluvatar Triton compiler produces incorrect results in the EVEN_K=False
# (K remainder) code path.  To avoid this, K is always padded to a multiple of
# the largest BLOCK_K used in the tuning configs so that EVEN_K is guaranteed
# True regardless of which config the autotuner picks.
_MAX_BLOCK_K = 128


def _pad_dims(a, b, M, N, K):
    """Pad a (M×K) and b (K×N) so all dims >= _MIN_TRITON_DIM and K is a
    multiple of _MAX_BLOCK_K (ensures EVEN_K=True).

    Returns (a_padded, b_padded, padded_M, padded_N, padded_K).
    """
    pad_M = max(_MIN_TRITON_DIM - M, 0)
    pad_N = max(_MIN_TRITON_DIM - N, 0)
    # Pad K to the next multiple of _MAX_BLOCK_K to guarantee EVEN_K=True.
    new_K = K + max(_MIN_TRITON_DIM - K, 0)
    remainder = new_K % _MAX_BLOCK_K
    if remainder:
        new_K += _MAX_BLOCK_K - remainder
    pad_K = new_K - K
    if pad_M or pad_K:
        a = torch.nn.functional.pad(a, (0, pad_K, 0, pad_M))
    if pad_K or pad_N:
        b = torch.nn.functional.pad(b, (0, pad_N, 0, pad_K))
    return a, b, M + pad_M, N + pad_N, new_K


def _launch_mm(a, b, c, M, N, K):
    """Launch Triton matmul _kernel; c must be pre-allocated."""
    # Pad when any dim is below the minimum tile or K is not a multiple of
    # _MAX_BLOCK_K (EVEN_K=False is broken on the Iluvatar compiler).
    need_pad = (
        M < _MIN_TRITON_DIM
        or N < _MIN_TRITON_DIM
        or K < _MIN_TRITON_DIM
        or K % _MAX_BLOCK_K != 0
    )
    if need_pad:
        a, b, pM, pN, pK = _pad_dims(a, b, M, N, K)
        c_padded = torch.empty((pM, pN), device=c.device, dtype=c.dtype)
    else:
        pM, pN, pK = M, N, K
        c_padded = c

    ab_dtype = get_higher_dtype(a.dtype, b.dtype)
    acc_dtype_tl = tl.float32
    ab_dtype_tl = _to_tl_type(ab_dtype)

    grid = lambda META: (
        triton.cdiv(pM, META["BLOCK_M"]) * triton.cdiv(pN, META["BLOCK_N"]),
        META["SPLIT_K"],
    )

    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c_padded,
            pM,
            pN,
            pK,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c_padded.stride(0),
            c_padded.stride(1),
            acc_dtype=acc_dtype_tl,
            input_precision=None,
            fp8_fast_accum=True,
            GROUP_M=8,
            AB_DTYPE=ab_dtype_tl,
        )

    if need_pad:
        c.copy_(c_padded[:M, :N])

    return c


def _ensure_mm_layout(a, b):
    """Ensure inputs have a valid layout for the Triton mm kernel.

    The kernel requires that at least one stride equals 1 (row- or column-major).
    Additionally, on the Iluvatar backend the compiler crashes when both matrices
    reference the same storage with transposed strides (self-transpose pattern) and
    dtype is float32.  Making at least one input contiguous avoids this.
    """
    # If both strides > 1, the tensor is neither row- nor column-major.
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # Guard against the self-transpose compiler bug: when a and b share storage
    # and both are transposed (stride != 1 in leading dim), make b contiguous to
    # give the compiler a clean layout.
    if a.data_ptr() == b.data_ptr():
        b = b.contiguous()
    return a, b


def mm(a, b):
    logger.debug("GEMS_ILUVATAR MM")
    device = a.device
    a, b = _ensure_mm_layout(a, b)
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    return _launch_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS_ILUVATAR MM_OUT")
    a, b = _ensure_mm_layout(a, b)
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    return _launch_mm(a, b, out, M, N, K)
