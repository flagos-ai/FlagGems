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

import functools
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _xor_combine(a, b):
    return a ^ b


@triton.jit
def _to_i64_bits(val, dtype_code: tl.constexpr):
    # Convert each element to its 64-bit hash pattern, represented as int64
    # (bit-identical to the generic uint64 representation; XOR is bitwise so
    # signedness is irrelevant). Kunlunxin miscompiles fp64/uint64 (they degrade
    # to fp32), so we avoid both and do all arithmetic in int64/uint32.
    if dtype_code == 0 or dtype_code == 1 or dtype_code == 2:
        # float32/float16/bfloat16: exact-widen to float32, then rebuild the
        # float64 bit pattern from the float32 bits. fp16/bf16 -> fp32 -> fp64
        # is bit-identical to fp16/bf16 -> fp64 because both widenings are exact.
        v32 = val.to(tl.float32)
        u32 = v32.to(tl.uint32, bitcast=True)
        sign = (u32 >> 31) & 1
        exp8 = (u32 >> 23) & 0xFF
        frac = u32 & 0x7FFFFF
        sign64 = sign.to(tl.int64)
        exp8_64 = exp8.to(tl.int64)
        frac64 = frac.to(tl.int64)

        is_normal = (exp8 >= 1) & (exp8 <= 254)
        is_infnan = exp8 == 255
        is_zero = (exp8 == 0) & (frac == 0)

        # normal: exp11 = exp8 + (1023 - 127) = exp8 + 896
        normal_bits = (sign64 << 63) | ((exp8_64 + 896) << 52) | (frac64 << 29)
        # inf/nan: exp11 = 0x7FF, mantissa carries the nan payload (0 for inf)
        infnan_bits = (sign64 << 63) | (0x7FF << 52) | (frac64 << 29)
        # zero (incl. -0.0): just the sign bit
        zero_bits = sign64 << 63
        # subnormal: multiply by 2^23 to normalize exactly, then subtract 23
        # from the float64 exponent.
        b2 = (v32 * 8388608.0).to(tl.uint32, bitcast=True)
        sub_bits = (
            (((b2 >> 31) & 1).to(tl.int64) << 63)
            | ((((b2 >> 23) & 0xFF).to(tl.int64) + 896 - 23) << 52)
            | ((b2 & 0x7FFFFF).to(tl.int64) << 29)
        )

        u = tl.where(
            is_normal,
            normal_bits,
            tl.where(is_infnan, infnan_bits, tl.where(is_zero, zero_bits, sub_bits)),
        )
    elif dtype_code == 4:  # int64: bits passthrough
        u = val
    elif dtype_code == 5 or dtype_code == 6 or dtype_code == 7:
        # int32/int16/int8: sign-extend to int64.
        u = val.to(tl.int64)
    elif dtype_code == 8 or dtype_code == 9:
        # uint8/bool: zero-extend.
        u = val.to(tl.int64)
    else:  # 10: int64 partials passthrough
        u = val
    return u


@libentry()
@triton.jit
def hash_tiled_kernel(
    x_ptr,
    out_ptr,
    N,
    row_stride,
    col_stride,
    n_split_size,
    out_row_stride,
    dtype_code: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """XOR-reduce one output row over one slice of the reduction dim, in int64.

    One program per (row, split). Each program performs a SINGLE masked
    tl.reduce of a BLOCK_N-wide chunk (no loop-carried accumulator): the
    Kunlunxin backend miscompiles a loop-carried `acc ^= tl.reduce(...)` scalar
    accumulator (consecutive rows collapse to identical wrong values), so the
    reduction dim is split into fixed BLOCK_N chunks that are each reduced in a
    single tl.reduce and combined in a second stage.
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    n_start = pid_s * n_split_size
    n_stop = tl.minimum(n_start + n_split_size, N)
    base = pid_m * row_stride
    cols = tl.arange(0, BLOCK_N)

    n_idx = n_start + cols
    mask = n_idx < n_stop
    # Clamp out-of-range lanes to the last valid index of this slice, then load
    # unmasked: the Kunlunxin backend ignores `other=` on masked loads and reads
    # adjacent (possibly out-of-bounds) memory, which corrupts the tail chunk.
    safe_idx = tl.minimum(n_idx, n_stop - 1)
    offs = base + safe_idx * col_stride
    v = tl.load(x_ptr + offs)
    u = _to_i64_bits(v, dtype_code)
    u = tl.where(mask, u, 0)
    r = tl.reduce(u, 0, _xor_combine)

    out_off = pid_m * out_row_stride + pid_s
    tl.store(out_ptr + out_off, r)


@functools.lru_cache(maxsize=1024)
def _block_n(N, elem_size):
    np2 = triton.next_power_of_2
    cap = 4096 if elem_size <= 4 else 2048
    return max(1, min(np2(N), cap))


def hash_tensor(x, dim, keepdim=False, mode=0):
    logger.debug("GEMS_KUNLUNXIN HASH_TENSOR")

    dtype = x.dtype
    device = x.device

    dtype_map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float64: 3,
        torch.int64: 4,
        torch.int32: 5,
        torch.int16: 6,
        torch.int8: 7,
        torch.uint8: 8,
        torch.bool: 9,
    }
    dtype_code = dtype_map[dtype]

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dim = list(range(x.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)

    if x.ndim == 0:
        dim = []

    dim = [(d % x.ndim) if x.ndim > 0 else 0 for d in dim]
    dim = sorted(set(dim))

    output_shape = []
    for i in range(x.ndim):
        if i not in dim:
            output_shape.append(x.shape[i])
        elif keepdim:
            output_shape.append(1)

    reduce_size = 1
    for d in dim:
        reduce_size *= x.shape[d]

    output_numel = 1
    for s in output_shape:
        output_numel *= s

    M = output_numel
    N = reduce_size

    perm = [i for i in range(x.ndim) if i not in dim] + dim
    xperm = x if perm == list(range(x.ndim)) else x.permute(perm)
    xr = xperm.reshape(M, N)
    row_stride = xr.stride(0)
    col_stride = xr.stride(1)

    # Compute in int64 space (fp64/uint64 miscompile on Kunlunxin); the final
    # view reinterprets the identical bits as uint64 to match aten's output.
    output = torch.empty(M, dtype=torch.int64, device=device)

    if M > 0 and N > 0:
        BLOCK_N = _block_n(N, x.element_size())

        # Split the reduction dim into fixed BLOCK_N chunks. Each chunk is
        # reduced by a single tl.reduce (the Kunlunxin backend miscompiles a
        # loop-carried XOR accumulator), then combined in a second stage.
        num_n_splits = (N + BLOCK_N - 1) // BLOCK_N

        if num_n_splits == 1:
            grid = (M, 1)
            hash_tiled_kernel[grid](
                xr,
                output,
                N,
                row_stride,
                col_stride,
                N,
                1,
                dtype_code,
                BLOCK_N,
            )
        else:
            partials = torch.empty(M * num_n_splits, dtype=torch.int64, device=device)
            grid = (M, num_n_splits)
            hash_tiled_kernel[grid](
                xr,
                partials,
                N,
                row_stride,
                col_stride,
                BLOCK_N,
                num_n_splits,
                dtype_code,
                BLOCK_N,
            )

            # Stage 2: XOR-combine the num_n_splits partials per output row.
            # Partials are plain int64 (dtype_code 10), so a wide tile is safe.
            M2, N2 = M, num_n_splits
            BLOCK_N2 = triton.next_power_of_2(N2)
            grid2 = (M2, 1)
            hash_tiled_kernel[grid2](
                partials,
                output,
                N2,
                num_n_splits,
                1,
                N2,
                1,
                10,
                BLOCK_N2,
            )

    output = output.view(torch.uint64)

    if len(output_shape) == 0:
        output = output.reshape([])
    else:
        output = output.reshape(output_shape)

    return output
