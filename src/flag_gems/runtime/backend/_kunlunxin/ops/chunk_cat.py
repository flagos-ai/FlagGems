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
from typing import List

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 16384


@triton.jit
def _chunk_cat_f64_main_kernel(
    input_ptr,
    output_ptr,
    n64,
    c_valid: tl.constexpr,
    K: tl.constexpr,
    W64: tl.constexpr,
    T: tl.constexpr,
    t: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """f64-domain kernel for fully-in-bounds chunks (c < c_valid).

    Reinterprets the input/output as float64 (a pure bit-wise copy: per_f64
    elements of the original dtype per f64 lane, where per_f64 = 8/element_size).
    Requires per_f64 | (D*A) and per_f64 | (C*A) so all chunk boundaries stay
    f64-aligned (no odd-element crossing).

    In f64 units: n64 = D*A/per_f64 elements per batch, W64 = C*A/per_f64
    elements per chunk. Chunks c < c_valid are provably in-bounds, so no
    address clamping is needed on the hot path.
    """
    pid_wa = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // c_valid
    c = pid_bc % c_valid
    wa = pid_wa * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = wa < W64
    in_pos = b * n64 + c * W64 + wa
    out_pos = b * (K * T * W64) + c * (T * W64) + t * W64 + wa
    p64 = input_ptr.to(tl.pointer_type(tl.float64))
    o64 = output_ptr.to(tl.pointer_type(tl.float64))
    tl.store(o64 + out_pos, tl.load(p64 + in_pos, mask=m), mask=m)


@triton.jit
def _chunk_cat_f64_tail_kernel(
    input_ptr,
    output_ptr,
    n64,
    c_tail,
    n_tail,
    K: tl.constexpr,
    W64: tl.constexpr,
    T: tl.constexpr,
    t: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """f64-domain kernel for the padded tail chunks (c in [c_tail, c_tail + n_tail)).

    Loads are clamped to the last in-batch element and zeroed with tl.where, so
    the address never goes out of bounds even for fully-padded chunks.
    """
    pid_wa = tl.program_id(0)
    pid_bc = tl.program_id(1)
    c = c_tail + pid_bc % n_tail
    b = pid_bc // n_tail
    wa = pid_wa * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = wa < W64
    in_pos = b * n64 + c * W64 + wa
    out_pos = b * (K * T * W64) + c * (T * W64) + t * W64 + wa
    valid = m & (in_pos < (b + 1) * n64)
    p64 = input_ptr.to(tl.pointer_type(tl.float64))
    o64 = output_ptr.to(tl.pointer_type(tl.float64))
    data = tl.load(p64 + tl.minimum(in_pos, (b + 1) * n64 - 1), mask=valid)
    data = tl.where(valid, data, 0.0)
    tl.store(o64 + out_pos, data, mask=m)


@triton.jit
def _chunk_cat_f32_main_kernel(
    input_ptr,
    output_ptr,
    n,
    c_valid: tl.constexpr,
    K: tl.constexpr,
    W: tl.constexpr,
    T: tl.constexpr,
    t: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise fallback kernel for fully-in-bounds chunks (c < c_valid).

    Used when the f64 reinterpretation is not applicable (odd element counts or
    8-byte element types).
    """
    pid_wa = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // c_valid
    c = pid_bc % c_valid
    wa = pid_wa * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = wa < W
    in_pos = b * n + c * W + wa
    out_pos = b * (K * T * W) + c * (T * W) + t * W + wa
    tl.store(output_ptr + out_pos, tl.load(input_ptr + in_pos, mask=m), mask=m)


@triton.jit
def _chunk_cat_f32_tail_kernel(
    input_ptr,
    output_ptr,
    n,
    c_tail,
    n_tail,
    K: tl.constexpr,
    W: tl.constexpr,
    T: tl.constexpr,
    t: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise fallback kernel for the padded tail chunks."""
    pid_wa = tl.program_id(0)
    pid_bc = tl.program_id(1)
    c = c_tail + pid_bc % n_tail
    b = pid_bc // n_tail
    wa = pid_wa * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = wa < W
    in_pos = b * n + c * W + wa
    out_pos = b * (K * T * W) + c * (T * W) + t * W + wa
    valid = m & (in_pos < (b + 1) * n)
    data = tl.load(input_ptr + tl.minimum(in_pos, (b + 1) * n - 1), mask=valid)
    data = tl.where(valid, data, 0.0)
    tl.store(output_ptr + out_pos, data, mask=m)


def _launch_chunk_cat_kernels(
    tensors: List[torch.Tensor], out: torch.Tensor, dim: int, num_chunks: int
) -> None:
    """Launch one (main + tail) kernel per tensor.

    Layout (f32 elements): B = prod(shape[:dim]), D = shape[dim],
    A = prod(shape[dim+1:]), C = ceil(D/num_chunks), T = len(tensors),
    W = C * A. Output [B, K, C*T, A] (C-order) with
        out[b, c, t*C + w, a] = in[b, c*C + w, a]   (0 if c*C + w >= D)

    A chunk c is fully in-bounds iff (c+1)*W <= D*A, i.e. c < c_valid where
    c_valid = min(D*A // W, K) = min(D // C, K). The remaining chunks (up to
    num_chunks - 1 of them when D < (K-1)*C) may be partially or fully padded
    and are handled by the masked tail kernel.

    Whenever per_f64 = 8 // element_size satisfies per_f64 | (D*A) and
    per_f64 | (C*A), chunks are processed in the float64 domain as a pure bit
    copy (each f64 lane covers per_f64 elements); this emits 128-bit accesses
    (~1.8 TB/s) instead of the 64-bit accesses of f32 loads (~1.0 TB/s) on
    TritonXPU, which roughly doubles memory throughput. Otherwise an
    element-wise float32 fallback kernel is used.
    """
    B = 1
    for s in tensors[0].shape[:dim]:
        B *= s
    A = 1
    for s in tensors[0].shape[dim + 1 :]:
        A *= s
    D = tensors[0].shape[dim]
    C = (D + num_chunks - 1) // num_chunks
    T = len(tensors)

    n = D * A
    W = C * A
    per_f64 = 8 // tensors[0].element_size()

    use_f64 = per_f64 > 1 and n % per_f64 == 0 and W % per_f64 == 0
    if use_f64:
        n64 = n // per_f64
        W64 = W // per_f64
        c_valid = min(n64 // W64, num_chunks)  # == min(D // C, K)
        for t, tensor in enumerate(tensors):
            if c_valid > 0:
                grid = (triton.cdiv(W64, _BLOCK_SIZE), B * c_valid)
                _chunk_cat_f64_main_kernel[grid](
                    tensor,
                    out,
                    n64,
                    c_valid,
                    num_chunks,
                    W64,
                    T,
                    t,
                    BLOCK_SIZE=_BLOCK_SIZE,
                )
            if c_valid < num_chunks:
                grid = (triton.cdiv(W64, _BLOCK_SIZE), B * (num_chunks - c_valid))
                _chunk_cat_f64_tail_kernel[grid](
                    tensor,
                    out,
                    n64,
                    c_valid,
                    num_chunks - c_valid,
                    num_chunks,
                    W64,
                    T,
                    t,
                    BLOCK_SIZE=_BLOCK_SIZE,
                )
    else:
        c_valid = min(n // W, num_chunks)  # == min(D // C, K)
        for t, tensor in enumerate(tensors):
            if c_valid > 0:
                grid = (triton.cdiv(W, _BLOCK_SIZE), B * c_valid)
                _chunk_cat_f32_main_kernel[grid](
                    tensor,
                    out,
                    n,
                    c_valid,
                    num_chunks,
                    W,
                    T,
                    t,
                    BLOCK_SIZE=_BLOCK_SIZE,
                )
            if c_valid < num_chunks:
                grid = (triton.cdiv(W, _BLOCK_SIZE), B * (num_chunks - c_valid))
                _chunk_cat_f32_tail_kernel[grid](
                    tensor,
                    out,
                    n,
                    c_valid,
                    num_chunks - c_valid,
                    num_chunks,
                    W,
                    T,
                    t,
                    BLOCK_SIZE=_BLOCK_SIZE,
                )


def chunk_cat(tensors: List[torch.Tensor], dim: int, num_chunks: int) -> torch.Tensor:
    """_chunk_cat on Kunlunxin.

    A Triton kernel implements the whole interleaving (chunk + pad + cat +
    stack equivalent); no ATen chunk/cat/stack/zeros fallback is used. The
    float64-domain vectorization is a TritonXPU workaround: f32 vector loads
    emit 64-bit accesses (~1 TB/s) while f64 loads emit 128-bit (~1.8 TB/s);
    see _launch_chunk_cat_kernels.
    """
    if len(tensors) == 0:
        raise ValueError("_chunk_cat(): expected a non-empty list of Tensors")

    if num_chunks <= 0:
        raise ValueError(f"_chunk_cat(): num_chunks must be positive, got {num_chunks}")

    ndim = tensors[0].ndim
    if ndim == 0:
        raise ValueError("_chunk_cat(): expected tensors with at least 1 dimension")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"_chunk_cat(): dim {dim} out of range for tensor with {ndim} dimensions"
        )

    dim = dim % ndim
    dim_size = tensors[0].shape[dim]
    chunk_size = (dim_size + num_chunks - 1) // num_chunks
    num_tensors = len(tensors)

    stride_after = 1
    for s in tensors[0].shape[dim + 1 :]:
        stride_after *= s

    out_shape = list(tensors[0].shape[:dim]) + [
        num_chunks,
        chunk_size * num_tensors * stride_after,
    ]
    out = torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)

    total = 1
    for s in tensors[0].shape[:dim]:
        total *= s
    total *= num_chunks * chunk_size * stride_after
    if total == 0:
        return out

    tensors = [t.contiguous() if not t.is_contiguous() else t for t in tensors]
    _launch_chunk_cat_kernels(tensors, out, dim, num_chunks)
    return out
