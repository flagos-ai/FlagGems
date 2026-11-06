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

"""Lean launches for dense, same-dtype addmm with alpha=beta=1."""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .kernels import _addmm_compact_body, addmm_fp32_kernel, addmm_kernel
from .ring import addmm_ring_kernel

# Reuse the existing arithmetic with shorter launch signatures. Dense strides
# and alpha/beta flags are compile-time constants, so these entry points do not
# need the generic tuning wrapper on each call.
_fp32_body = addmm_fp32_kernel.jit_function
_generic_body = addmm_kernel.jit_function
_ring_body = addmm_ring_kernel.jit_function


@libentry()
@triton.jit
def _dense_compact(
    A,
    B,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    _addmm_compact_body(
        A,
        B,
        Bias,
        C,
        1,
        1,
        M,
        N,
        K,
        N,
        1,
        N,
        1,
        BM,
        BN,
        BK,
        1,
        PREFETCH,
        True,
        False,
        True,
    )


@libentry()
@triton.jit
def _dense_fp32(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    _fp32_body(
        A,
        B,
        Bias,
        C,
        1,
        1,
        M,
        N,
        K,
        N,
        1,
        N,
        1,
        N,
        1,
        64,
        128,
        32,
        8,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    )


@libentry()
@triton.jit
def _dense_generic(
    A,
    B,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    _generic_body(
        A,
        B,
        Bias,
        C,
        1,
        1,
        M,
        N,
        K,
        N,
        1,
        N,
        1,
        N,
        1,
        BM,
        BN,
        BK,
        GROUP,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        M % BM == 0,
        N % BN == 0,
        K % BK == 0,
        False,
        False,
    )


@libentry()
@triton.jit
def _dense_ring(
    A,
    B,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BF16: tl.constexpr,
    LOGK: tl.constexpr,
):
    _ring_body(
        A,
        B,
        C,
        Bias,
        1,
        1,
        N,
        1,
        N,
        1,
        True,
        False,
        True,
        M,
        N,
        K,
        256,
        256,
        16,
        1,
        BF16,
        LOGK,
    )


def get_dense_metadata(bias, a, b, out, fresh_out, shape=None):
    # Preserve generic handling of broadcasting, promotion, layouts and aliasing.
    if shape is None:
        m, k = a.shape
        n = b.shape[1]
    else:
        m, n, k = shape
    dtype = a.dtype
    device = a.device
    if (
        dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or min(m, n, k) < 128
        or bias.shape != (m, n)
        or device.type != "cuda"
    ):
        return None
    if not (
        b.dtype == bias.dtype == dtype
        and device == b.device == bias.device
        and a.is_contiguous()
        and b.is_contiguous()
        and bias.is_contiguous()
    ):
        return None
    element_size = 4 if dtype == torch.float32 else 2
    if max(m * k, k * n, m * n) * element_size >= 2**31:
        return None
    if fresh_out:
        # The caller just allocated this contiguous output using a's metadata.
        return m, n, k, dtype, device
    if (
        out.dtype == dtype
        and out.device == device
        and out.is_contiguous()
        and not any(torch._C._overlaps(out, tensor) for tensor in (a, b, bias))
    ):
        return m, n, k, dtype, device
    return None


def launch_dense(bias, a, b, out, metadata, use_ring):
    # Use validated tile families; unsupported dimensions retain tuned fallback.
    m, n, k, dtype, device = metadata
    ring_candidate = (
        dtype != torch.float32 and ((m + 255) // 256) * ((n + 255) // 256) >= 24
    )
    with torch_device_fn.device(device):
        if ring_candidate and use_ring(bias, a, b, out):
            _dense_ring[(((m + 255) // 256) * ((n + 255) // 256),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                dtype == torch.bfloat16,
                min(k.bit_length() - 1, 12),
                num_warps=8,
                num_stages=1,
                allow_flush_denorm=True,
                enable_fp_fusion=True,
            )
        elif (
            dtype == torch.float32
            and min(m, n, k) >= 1536
            and m % 128 == 0
            and n % 128 == 0
            and k % 64 == 0
        ):
            _dense_fp32[(((m + 63) // 64) * ((n + 127) // 128),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=16,
            )
        else:
            if max(m, n, k) > 3072:
                return None
            if dtype == torch.float32:
                bm, bn, bk, warps, stages = (
                    32,
                    64,
                    64,
                    4,
                    2 if max(m, n, k) <= 512 else 1,
                )
                group = 1
                if max(m, n, k) <= 512:
                    bn = 32
                elif max(m, n, k) <= 1536:
                    bm, bn, bk, group = 64, 128, 32, 8
                elif min(m, n, k) >= 1536:
                    bm, bn, group = 64, 64, 8
                _dense_generic[(((m + bm - 1) // bm) * ((n + bn - 1) // bn),)](
                    a,
                    b,
                    bias,
                    out,
                    m,
                    n,
                    k,
                    bm,
                    bn,
                    bk,
                    group,
                    num_warps=warps,
                    num_stages=stages,
                )
                return out
            elif max(m, n, k) <= 512:
                bm, bn, bk, warps, stages = 32, 32, 128, 2, 1
            else:
                bm, bn, bk, warps, stages = 64, 64, 64, 2, 2
            _dense_compact[(((m + bm - 1) // bm) * ((n + bn - 1) // bn),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                max(m, n, k) > 512,
                num_warps=warps,
                num_stages=stages,
                matrix_instr_nonkdim=16,
            )
    return out
