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
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def heur_split_k(args):
    return 1


def heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def heur_group_m(args):
    if args["BLOCK_M"] > args["BLOCK_N"]:
        return 1
    else:
        return (args["M"] + args["BLOCK_M"] - 1) // args["BLOCK_M"]


autotune_decorator = triton.autotune(
    configs=[],
    generate_configs="mm",
    key=["M", "N", "K"],
)


KLX_USE_AUTOTUNE = os.environ.get("KLX_USE_AUTOTUNE", "1") == "1"

if not KLX_USE_AUTOTUNE:

    # XPU tile probe (2026-08-13, XPU 4, 7 unique shapes x 3 dtypes, direct
    # do_bench warm+rep medians): the 256^3 tile is the floor for M,N > 512 on
    # all dtypes (fp16 4096^3 0.68ms / 0.82x, fp32 1.37ms / 0.90x), while small
    # shapes (M,N <= 512) are launch-bound and prefer the 128-tile w4 config
    # (384^3: 0.014 -> 0.0087ms fp16, 0.021 -> 0.012ms bf16, 0.014 -> 0.011ms
    # fp32, ~1.05-1.12x vs torch). num_stages stays at backend default (2):
    # bf16 BK=256 collapses at s3 (1.82ms vs 1.32ms on 4096^3). BK is kept at
    # 256 for the 256-tile (fp16 needs BK=256: 1.01ms at BK=128 vs 0.68ms at
    # BK=256 on 4096^3).

    def heur_block_m(args):
        M = args["M"]
        if M <= 512:
            return 128
        return 256

    def heur_block_n(args):
        N = args["N"]
        if N <= 512:
            return 128
        return 256

    def heur_block_k(args):
        M = args["M"]
        N = args["N"]
        if M <= 512 and N <= 512:
            return 128
        return 256

    def heur_num_warps(args):
        M = args["M"]
        N = args["N"]
        if M <= 512 and N <= 512:
            return 4
        return 8

    autotune_decorator = triton.heuristics(
        {
            "BLOCK_M": heur_block_m,
            "BLOCK_N": heur_block_n,
            "BLOCK_K": heur_block_k,
            "num_warps": heur_num_warps,
        }
    )


@libentry()
@autotune_decorator
@triton.heuristics(
    {
        "SPLIT_K": heur_split_k,
        "EVEN_K": heur_even_k,
        "GROUP_M": heur_group_m,
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
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = ext.program_id(0)
    pid_z = ext.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]

_FAST_MODE_ENV = "XMLIR_MATMUL_FAST_MODE"


def _set_matmul_fast_mode(a_dtype, M, N, K):
    """XPU probe (2026-08-13, XPU 4): XMLIR_MATMUL_FAST_MODE=1 speeds the
    bf16 tl.dot lowering for large-K GEMMs (4096^3 1.32->0.81ms; 2048^3
    0.20->0.15ms; K=65536 1.86->1.44ms) while fp16/fp32 kernels are
    unaffected; on small bf16 shapes it regresses (64^3 0.012->0.018ms), so
    the flag is applied selectively: bf16 only, with K >= 2048 and both M, N
    >= 128."""
    if (
        a_dtype == torch.bfloat16
        and K >= 2048
        and M >= 128
        and N >= 128
    ):
        saved = os.environ.get(_FAST_MODE_ENV)
        os.environ[_FAST_MODE_ENV] = "1"
        return saved
    return None


def _restore_matmul_fast_mode(saved):
    if saved is None:
        os.environ.pop(_FAST_MODE_ENV, None)
    else:
        os.environ[_FAST_MODE_ENV] = saved


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


def mm(a, b):
    logger.debug("GEMS_KUNLUNXIN MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    saved = _set_matmul_fast_mode(a.dtype, M, N, K)
    try:
        with torch_device_fn.device(a.device):
            mm_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                dot_out_dtype=dot_out_dtype,
            )
    finally:
        _restore_matmul_fast_mode(saved)
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_KUNLUNXIN MM_OUT")
    # handle non-contiguous inputs if necessary
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    dot_out_dtype = tl.float32
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    saved = _set_matmul_fast_mode(a.dtype, M, N, K)
    try:
        with torch_device_fn.device(a.device):
            mm_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                dot_out_dtype=dot_out_dtype,
            )
    finally:
        _restore_matmul_fast_mode(saved)
    return c
