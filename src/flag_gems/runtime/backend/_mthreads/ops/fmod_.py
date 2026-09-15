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

from flag_gems.ops.fmod import fmod_scalar_ as default_fmod_scalar_
from flag_gems.ops.fmod import fmod_tensor_ as default_fmod_tensor_
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_fmod = tl_extra_shim.fmod

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_FLOAT_DTYPES = _SUPPORTED_DTYPES | {torch.float64}


@triton.jit
def _fmod_scalar_kernel(
    x_ptr,
    b,
    n_elements,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    COMPUTE_F32: tl.constexpr,
    IS_INT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        if IS_INT:
            r = x % b
        elif COMPUTE_F32:
            r = _fmod(x.to(tl.float32), b.to(tl.float32)).to(x.dtype)
        else:
            r = _fmod(x, b.to(x.dtype))
        tl.store(x_ptr + offs, r, mask=mask)
    else:
        x = tl.load(x_ptr + offs)
        if IS_INT:
            r = x % b
        elif COMPUTE_F32:
            r = _fmod(x.to(tl.float32), b.to(tl.float32)).to(x.dtype)
        else:
            r = _fmod(x, b.to(x.dtype))
        tl.store(x_ptr + offs, r)


@triton.jit
def _fmod_flat_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    EVEN_N: tl.constexpr,
    PROMOTE: tl.constexpr,
    IS_FP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN_N:
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
    else:
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
    if IS_FP:
        if PROMOTE:
            r = _fmod(a.to(tl.float32), b.to(tl.float32)).to(a.dtype)
        else:
            r = _fmod(a, b)
    else:
        r = a % b
    if EVEN_N:
        tl.store(a_ptr + offs, r)
    else:
        tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _fmod_bcast_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    EVEN_N: tl.constexpr,
    PROMOTE: tl.constexpr,
    IS_FP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    dims = [d0, d1, d2, d3, d4, d5, d6, d7]
    sds = [s0, s1, s2, s3, s4, s5, s6, s7]
    rem = offs.to(tl.int64)
    b_idx = tl.zeros((BLOCK,), dtype=tl.int64)
    for j in tl.static_range(8):
        dd = dims[7 - j]
        coord = rem % dd
        b_idx += coord * sds[7 - j]
        rem = rem // dd
    if EVEN_N:
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + b_idx)
    else:
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + b_idx, mask=mask)
    if IS_FP:
        if PROMOTE:
            r = _fmod(a.to(tl.float32), b.to(tl.float32)).to(a.dtype)
        else:
            r = _fmod(a, b)
    else:
        r = a % b
    if EVEN_N:
        tl.store(a_ptr + offs, r)
    else:
        tl.store(a_ptr + offs, r, mask=mask)


def _use_triton_kernel(A) -> bool:
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _FLOAT_DTYPES:
        return False
    if not A.is_contiguous() or A.numel() == 0:
        return False
    return True


_SMALL_N = 1 << 20
_HUGE_N = 1 << 26


def _pick_config(dtype, n):
    if n < _SMALL_N:
        return 1024, 4
    if dtype == torch.float32:
        return 4096, 16
    if dtype == torch.float16:
        if n >= _HUGE_N:
            return 4096, 8
        return 1024, 4
    # bfloat16
    if n >= _HUGE_N:
        return 2048, 8
    return 1024, 4


def fmod_scalar_(A, B):
    logger.debug("GEMS_MTHREADS FMOD_SCALAR_")
    if _use_triton_kernel(A) and not isinstance(B, torch.Tensor):
        try:
            scalar = B if isinstance(B, (int, float)) else float(B)
        except Exception:
            return default_fmod_scalar_(A, B)
        n = A.numel()
        dt = A.dtype
        is_int = dt not in _FLOAT_DTYPES
        compute_f32 = dt in (torch.float16, torch.bfloat16)
        if dt == torch.float32:
            BLOCK, nw = 1024, 8
        elif compute_f32:
            if n >= (1 << 21):
                BLOCK, nw = 1024, 4
            else:
                BLOCK, nw = 1024, 8
        else:
            BLOCK, nw = 1024, 4
        masked = (n % BLOCK) != 0
        grid = (triton.cdiv(n, BLOCK),)
        _fmod_scalar_kernel[grid](
            A,
            scalar,
            n,
            BLOCK=BLOCK,
            MASKED=masked,
            COMPUTE_F32=compute_f32,
            IS_INT=is_int,
            num_warps=nw,
        )
        return A
    return default_fmod_scalar_(A, B)


def fmod_tensor_(A, B):
    logger.debug("GEMS_MTHREADS FMOD_TENSOR_")
    if _use_triton_kernel(A) and isinstance(B, torch.Tensor) and B.dtype == A.dtype:
        a_shape = A.shape
        n = A.numel()
        is_fp = A.dtype.is_floating_point
        promote = A.dtype in (torch.float16, torch.bfloat16)
        block, warps = _pick_config(A.dtype, n)
        even = (n % block) == 0
        if a_shape == B.shape and B.is_contiguous():
            _fmod_flat_kernel[(triton.cdiv(n, block),)](
                A,
                B,
                n,
                EVEN_N=even,
                PROMOTE=promote,
                IS_FP=is_fp,
                BLOCK=block,
                num_warps=warps,
            )
            return A
        if A.dim() <= 8 and B.dim() <= 8 and B.is_contiguous():
            R = len(a_shape)
            rb = len(B.shape)
            dims = [1] * 8
            sds = [0] * 8
            ok = True
            for j in range(R):
                jj = 8 - R + j
                dims[jj] = a_shape[j]
                k = j + rb - R
                if k >= 0 and B.shape[k] != 1:
                    if a_shape[j] % B.shape[k] != 0:
                        ok = False
                        break
                    sds[jj] = B.stride()[k]
            if ok:
                _fmod_bcast_kernel[(triton.cdiv(n, 1024),)](
                    A,
                    B,
                    n,
                    dims[0],
                    dims[1],
                    dims[2],
                    dims[3],
                    dims[4],
                    dims[5],
                    dims[6],
                    dims[7],
                    sds[0],
                    sds[1],
                    sds[2],
                    sds[3],
                    sds[4],
                    sds[5],
                    sds[6],
                    sds[7],
                    EVEN_N=even,
                    PROMOTE=promote,
                    IS_FP=is_fp,
                    BLOCK=1024,
                )
                return A
    return default_fmod_tensor_(A, B)


def fmod_(A, B):
    logger.debug("GEMS_MTHREADS FMOD_")
    if isinstance(B, torch.Tensor):
        return fmod_tensor_(A, B)
    return fmod_scalar_(A, B)
