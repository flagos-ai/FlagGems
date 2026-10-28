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
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.ops.scaled_mm import _prepare_scaled_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _scaled_mm_kernel(
    A,
    B,
    SA,
    SB,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    ROW_A: tl.constexpr,
    ROW_B: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DESCRIPTOR: tl.constexpr,
    DIRECT_FP8: tl.constexpr,
    A_E5: tl.constexpr,
    B_E5: tl.constexpr,
    SMALL: tl.constexpr,
):
    # Descriptor cache keys omit base dtype. Keep both encodings in constexpr
    # arguments so alternating E4/E5 inputs cannot reuse the wrong binary.
    if SMALL:
        pm = tl.program_id(0)
        pn = tl.program_id(1)
    else:
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        group = pid // (8 * grid_n)
        group_m = tl.minimum(8, grid_m - group * 8)
        pm = group * 8 + pid % group_m
        pn = pid % (8 * grid_n) // group_m
    m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    WIDE: tl.constexpr = (
        M * AM + K * AK > 2147483647
        or K * BK + N * BN > 2147483647
        or M * CM + N * CN > 2147483647
    )
    if WIDE:
        m = m.to(tl.int64)
        n = n.to(tl.int64)
        k = k.to(tl.int64)
    if not DIRECT_FP8 and not SMALL:
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        if DESCRIPTOR:
            a = tl.load_tensor_descriptor(A, [pm * BLOCK_M, block * BLOCK_K])
            bt = tl.load_tensor_descriptor(B, [pn * BLOCK_N, block * BLOCK_K])
            b = tl.trans(bt)
        else:
            if WIDE:
                kk = block.to(tl.int64) * BLOCK_K + k
            else:
                kk = block * BLOCK_K + k
            if SMALL:
                a_ptr = A + m[:, None] * AM + kk[None, :] * AK
            else:
                a_ptr = A + m[:, None].to(tl.int64) * AM + kk[None, :].to(tl.int64) * AK
            a = tl.load(
                a_ptr,
                (m[:, None] < M) & (kk[None, :] < K),
                other=0.0,
            )
            if SMALL:
                b = tl.load(
                    B + kk[:, None] * BK + n[None, :] * BN,
                    (kk[:, None] < K) & (n[None, :] < N),
                    other=0.0,
                )
            else:
                bt = tl.load(
                    B + n[:, None].to(tl.int64) * BN + kk[None, :].to(tl.int64) * BK,
                    (n[:, None] < N) & (kk[None, :] < K),
                    other=0.0,
                )
                b = tl.trans(bt)
        if DIRECT_FP8:
            acc = tl.dot(a, b, acc)
        elif SMALL:
            acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)
        else:
            # Form C.T = B.T @ A.T. Convert before transposing the second
            # operand to reduce the mixed-FP8 staging cost on MUSA.
            acc = tl.dot(bt.to(tl.float16), tl.trans(a.to(tl.float16)), acc)
    if not DIRECT_FP8 and not SMALL:
        acc = tl.trans(acc)
    if ROW_A:
        acc *= tl.load(SA + m, m < M, other=0.0)[:, None]
    else:
        acc *= tl.load(SA)
    if ROW_B:
        acc *= tl.load(SB + n, n < N, other=0.0)[None, :]
    else:
        acc *= tl.load(SB)
    if HAS_BIAS and K > 0:
        acc += tl.load(Bias + n, n < N, other=0.0)[None, :].to(tl.float32)
    if SMALL:
        c_ptr = C + m[:, None] * CM + n[None, :] * CN
    else:
        c_ptr = C + m[:, None].to(tl.int64) * CM + n[None, :].to(tl.int64) * CN
    tl.store(
        c_ptr,
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


def _can_use_descriptor(a, b):
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and min(a.shape[0], b.shape[1], a.shape[1]) >= 256
        and a.stride(1) == b.stride(0) == 1
        and a.stride(0) > 0
        and b.stride(1) > 0
        and a.stride(0) % 16 == b.stride(1) % 16 == 0
        and a.data_ptr() % 16 == b.data_ptr() % 16 == 0
    )


def _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out):
    scale_a, scale_b, bias, out, row_a, row_b = _prepare_scaled_mm(
        self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out
    )
    m, k = self.shape
    n = mat2.shape[1]
    if m == 0 or n == 0:
        return out
    with torch_device_fn.device(self.device):
        descriptor = _can_use_descriptor(self, mat2)
        direct = self.dtype == mat2.dtype and self.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        if descriptor:
            bm, bn = 32, 64
            bk = min(512, triton.next_power_of_2(k))
            warps = 4
            a = TensorDescriptor(self, [m, k], list(self.stride()), [bm, bk])
            b = TensorDescriptor(
                mat2, [n, k], [mat2.stride(1), mat2.stride(0)], [bn, bk]
            )
        else:
            a, b = self, mat2
            if m <= 128:
                bm, bn, bk, warps = 16, 32, 64, 4
            elif not direct:
                bm, bn, bk, warps = 32, 64, 128, 8
            else:
                bm, bn, bk, warps = 32, 32, 64, 4
        small = m <= 128
        grid = (
            (triton.cdiv(m, bm), triton.cdiv(n, bn))
            if small
            else (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        )
        _scaled_mm_kernel[grid](
            a,
            b,
            scale_a,
            scale_b,
            bias,
            out,
            m,
            n,
            k,
            *self.stride(),
            *mat2.stride(),
            *out.stride(),
            row_a,
            row_b,
            bias is not None,
            bm,
            bn,
            bk,
            descriptor,
            direct,
            self.dtype == torch.float8_e5m2,
            mat2.dtype == torch.float8_e5m2,
            small,
            num_warps=warps,
            num_stages=3 if small else 1,
            enable_backend_opt=True,
        )
    return out


def scaled_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_MTHREADS SCALED_MM")
    # Match ATen GPU semantics: scale_result and the fast accumulation hint
    # do not change the mathematical result in this implementation.
    return _scaled_mm_impl(
        self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, None
    )


def scaled_mm_out(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    logger.debug("GEMS_MTHREADS SCALED_MM_OUT")
    return _scaled_mm_impl(
        self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out
    )
