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

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _hermite_scalar_flat_kernel(
    x_ptr,
    out_ptr,
    numel,
    N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float64)
    two_x = 2.0 * x
    if N == 0:
        result = tl.full([BLOCK], 1.0, tl.float64)
    elif N == 1:
        result = two_x
    else:
        h_km1 = tl.full([BLOCK], 1.0, tl.float64)
        h_k = two_x
        for k in tl.static_range(1, N):
            h_kp1 = tl.fma(two_x, h_k, (-2.0 * k) * h_km1)
            h_km1 = h_k
            h_k = h_kp1
        result = h_k
    tl.store(out_ptr + offs, result.to(OUT_DTYPE), mask=mask)


@triton.jit
def _hermite_scalar_bcast_kernel(
    x_ptr,
    out_ptr,
    numel,
    x_strides_ptr,
    sizes_ptr,
    N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x_off = tl.zeros([BLOCK], dtype=tl.int64)
    rem = offs.to(tl.int64)
    for d in tl.static_range(RANK):
        s = tl.load(sizes_ptr + d).to(tl.int64)
        xs = tl.load(x_strides_ptr + d).to(tl.int64)
        c = rem % s
        rem = rem // s
        x_off += c * xs
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float64)
    two_x = 2.0 * x
    if N == 0:
        result = tl.full([BLOCK], 1.0, tl.float64)
    elif N == 1:
        result = two_x
    else:
        h_km1 = tl.full([BLOCK], 1.0, tl.float64)
        h_k = two_x
        for k in tl.static_range(1, N):
            h_kp1 = tl.fma(two_x, h_k, (-2.0 * k) * h_km1)
            h_km1 = h_k
            h_k = h_kp1
        result = h_k
    tl.store(out_ptr + offs, result.to(OUT_DTYPE), mask=mask)


@triton.jit
def _hermite_tensor_flat_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        nv = tl.load(n_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
        nv = tl.load(n_ptr + offs)
    n_i = nv.to(tl.int32)
    n_i = tl.minimum(tl.maximum(n_i, 0), 9)
    two_x = 2.0 * x
    h_km1 = tl.full([BLOCK], 1.0, x.dtype)
    h_k = two_x
    result = tl.where(n_i == 0, h_km1, h_k)
    for k in tl.static_range(1, 9):
        h_kp1 = tl.fma(two_x, h_k, (-2.0 * k) * h_km1)
        h_km1 = h_k
        h_k = h_kp1
        result = tl.where(n_i == (k + 1), h_k, result)
    if MASKED:
        tl.store(out_ptr + offs, result.to(OUT_DTYPE), mask=mask)
    else:
        tl.store(out_ptr + offs, result.to(OUT_DTYPE))


@triton.jit
def _hermite_tensor_bcast_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    x_strides_ptr,
    n_strides_ptr,
    sizes_ptr,
    OUT_DTYPE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x_off = tl.zeros([BLOCK], dtype=tl.int64)
    n_off = tl.zeros([BLOCK], dtype=tl.int64)
    rem = offs.to(tl.int64)
    for d in tl.static_range(RANK):
        s = tl.load(sizes_ptr + d).to(tl.int64)
        xs = tl.load(x_strides_ptr + d).to(tl.int64)
        ns = tl.load(n_strides_ptr + d).to(tl.int64)
        c = rem % s
        rem = rem // s
        x_off += c * xs
        n_off += c * ns
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    nv = tl.load(n_ptr + n_off, mask=mask, other=0.0)
    n_i = nv.to(tl.int32)
    n_i = tl.minimum(tl.maximum(n_i, 0), 9)
    two_x = 2.0 * x
    h_km1 = tl.full([BLOCK], 1.0, x.dtype)
    h_k = two_x
    result = tl.where(n_i == 0, h_km1, h_k)
    for k in tl.static_range(1, 9):
        h_kp1 = tl.fma(two_x, h_k, (-2.0 * k) * h_km1)
        h_km1 = h_k
        h_k = h_kp1
        result = tl.where(n_i == (k + 1), h_k, result)
    tl.store(out_ptr + offs, result.to(OUT_DTYPE), mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

_BLOCK_TENSOR = 1024
_WARPS_TENSOR = 16
_BLOCK_SCALAR = 512
_WARPS_SCALAR = 8


def _map_dtype(dt):
    return {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }[dt]


def _prod(shape):
    numel = 1
    for s in shape:
        numel *= s
    return numel


def _check_n_range(n_int):
    if n_int < 0 or n_int > 9:
        raise ValueError(
            f"special_hermite_polynomial_h only supports n in [0, 9], got n={n_int}"
        )


def _run_scalar(x, n_int):
    out_shape = tuple(x.shape)
    numel = _prod(out_shape)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if numel == 0:
        return out
    out_dtype = _map_dtype(x.dtype)
    if x.is_contiguous() and numel < 2**31:
        grid = (triton.cdiv(numel, _BLOCK_SCALAR),)
        _hermite_scalar_flat_kernel[grid](
            x,
            out,
            numel,
            N=n_int,
            OUT_DTYPE=out_dtype,
            BLOCK=_BLOCK_SCALAR,
            num_warps=_WARPS_SCALAR,
        )
    else:
        rank = len(out_shape)
        sizes = torch.tensor(list(out_shape), dtype=torch.int64, device=x.device)
        xs = torch.tensor(list(x.stride()), dtype=torch.int64, device=x.device)
        grid = (triton.cdiv(numel, _BLOCK_SCALAR),)
        _hermite_scalar_bcast_kernel[grid](
            x,
            out,
            numel,
            xs,
            sizes,
            N=n_int,
            OUT_DTYPE=out_dtype,
            RANK=rank,
            BLOCK=_BLOCK_SCALAR,
            num_warps=_WARPS_SCALAR,
        )
    return out


def _run_tensor(x, n):
    if x.shape == n.shape:
        numel = x.numel()
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if numel == 0:
            return out
        out_dtype = _map_dtype(x.dtype)
        if x.is_contiguous() and n.is_contiguous() and numel < 2**31:
            block = _BLOCK_TENSOR
            masked = (numel % block) != 0
            grid = (triton.cdiv(numel, block),)
            _hermite_tensor_flat_kernel[grid](
                x,
                n,
                out,
                numel,
                OUT_DTYPE=out_dtype,
                BLOCK=block,
                MASKED=masked,
                num_warps=_WARPS_TENSOR,
            )
            return out
        out_shape = tuple(x.shape)
    else:
        out_shape = torch.broadcast_shapes(x.shape, n.shape)
        numel = _prod(out_shape)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        if numel == 0:
            return out
        out_dtype = _map_dtype(x.dtype)
    xv = x.expand(out_shape)
    nv = n.expand(out_shape)
    rank = len(out_shape)
    sizes = torch.tensor(list(out_shape), dtype=torch.int64, device=x.device)
    xs = torch.tensor(list(xv.stride()), dtype=torch.int64, device=x.device)
    ns = torch.tensor(list(nv.stride()), dtype=torch.int64, device=x.device)
    grid = (triton.cdiv(numel, _BLOCK_TENSOR),)
    _hermite_tensor_bcast_kernel[grid](
        x,
        n,
        out,
        numel,
        xs,
        ns,
        sizes,
        OUT_DTYPE=out_dtype,
        RANK=rank,
        BLOCK=_BLOCK_TENSOR,
        num_warps=_WARPS_TENSOR,
    )
    return out


def special_hermite_polynomial_h(x, n):
    if isinstance(n, (int, float)):
        n_int = int(n)
        _check_n_range(n_int)
        return _run_scalar(x, n_int)
    if isinstance(n, torch.Tensor):
        if n.numel() == 1:
            n_int = int(n.item())
            _check_n_range(n_int)
            return _run_scalar(x, n_int)
        return _run_tensor(x, n)
    raise TypeError(f"unsupported n type: {type(n)}")
