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

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def prepare_tensor_for_kron(tensor_a, tensor_b):
    a_shape = list(tensor_a.shape)
    b_shape = list(tensor_b.shape)

    if tensor_a.numel() == 0 or tensor_b.numel() == 0:
        if not a_shape:
            a_shape = [0]
        if not b_shape:
            b_shape = [0]

        if len(a_shape) > len(b_shape):
            b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
        elif len(b_shape) > len(a_shape):
            a_shape = [1] * (len(b_shape) - len(a_shape)) + a_shape

        out_shape = tuple(a * b for a, b in zip(a_shape, b_shape))
        return tensor_a.reshape(*a_shape), tensor_b.reshape(*b_shape), out_shape

    if len(a_shape) < 2:
        a_shape = [1] * (2 - len(a_shape)) + a_shape
    if len(b_shape) < 2:
        b_shape = [1] * (2 - len(b_shape)) + b_shape

    if len(a_shape) > len(b_shape):
        b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
    elif len(b_shape) > len(a_shape):
        a_shape = [1] * (len(b_shape) - len(a_shape)) + a_shape

    out_shape = tuple(a * b for a, b in zip(a_shape, b_shape))
    return tensor_a.reshape(*a_shape), tensor_b.reshape(*b_shape), out_shape


def calculate_indices(batch_idx, shape_a, shape_b):
    a_batch_dims = shape_a[:-2] or (1,)
    b_batch_dims = shape_b[:-2] or (1,)
    out_batch_dims = tuple(a * b for a, b in zip(a_batch_dims, b_batch_dims))

    out_indices = []
    remaining = batch_idx
    for dim_size in out_batch_dims[::-1]:
        out_indices.insert(0, remaining % dim_size)
        remaining //= dim_size

    a_idx = b_idx = 0
    for out_idx, (a_dim, b_dim) in zip(out_indices, zip(a_batch_dims, b_batch_dims)):
        a_idx = a_idx * a_dim + (out_idx // b_dim)
        b_idx = b_idx * b_dim + (out_idx % b_dim)

    return a_idx, b_idx


# --- XPU kron kernel (performance rewrite 2026-08-17) -----------------------
# One program owns R consecutive output rows; each output row is written as a
# flat stride-1 contiguous run in BLOCK_N chunks:
#   C[row, col] = A[i1, j1] * B[i2, j2] with i1 = row//M2, i2 = row%M2 and
#   j1 = col//N2, j2 = col%N2 recovered from the flat column index.
#
# Why this shape is fast on XPU vs the previous implementation:
#   * N2 is a tl.constexpr: the per-lane 64-bit col//N2, col%N2 compile into
#     magic-multiply sequences (~3 cyc) instead of runtime 64-bit divisions
#     (30-60 cyc each). The previous kernel passed N2 as a runtime i64 and this
#     single detail accounted for ~10-50x of the runtime on 16x16..256x256.
#   * The store is a flat contiguous 1-D run (BLOCK_N lanes) per chunk, which
#     is the only store pattern that gets near-copy bandwidth on P800 (2-D
#     tile stores measure 3-13x slower; a [BJ,BN2] tile + tl.reshape is
#     UNSUPPORTED on this backend -- inferReshapeOpEncoding is an UNREACHABLE
#     TODO, verified by CompilationError; a scalar-a outer-product variant was
#     ~1.3-2x slower than this flat layout).
#   * NEED_MASK: when N % BLOCK_N == 0 the loads/stores are unmasked entirely.
BLOCK_N_CAP = 8192


@triton.jit
def kron_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N1,
    M2,
    N,
    BATCH: tl.constexpr,
    a1: tl.int64,
    b0: tl.int64,
    b1: tl.int64,
    a_stride: tl.int64,
    N2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    R: tl.constexpr,
):
    pid = ext.program_id(0)
    nrb = (M + R - 1) // R
    if BATCH:
        bt = pid // nrb
        rb = pid % nrb
        ob1 = bt % (a1 * b1)
        ob0 = bt // (a1 * b1)
        aj = (ob0 // b0) * a1 + (ob1 // b1)
        bj = (ob0 % b0) * b1 + (ob1 % b1)
        a_off = aj * a_stride
        b_off = bj * (M2 * N2)
        c_b = bt * (M * N)
    else:
        rb = pid
        a_off = 0
        b_off = 0
        c_b = 0
    r0 = rb * R
    for r in tl.static_range(R):
        row = r0 + r
        i1 = row // M2
        i2 = row % M2
        a_base = a_off + i1 * N1
        b_base = b_off + i2 * N2
        c_base = c_b + row * N
        for off in range(0, N, BLOCK_N):
            col = off + tl.arange(0, BLOCK_N)
            j1 = col // N2
            j2 = col % N2
            if NEED_MASK:
                mask = col < N
                a = tl.load(a_ptr + a_base + j1, mask=mask, other=0.0).to(tl.float32)
                b = tl.load(b_ptr + b_base + j2, mask=mask, other=0.0).to(tl.float32)
                out = (a * b).to(c_ptr.dtype.element_ty)
                tl.store(c_ptr + c_base + col, out, mask=mask)
            else:
                a = tl.load(a_ptr + a_base + j1).to(tl.float32)
                b = tl.load(b_ptr + b_base + j2).to(tl.float32)
                out = (a * b).to(c_ptr.dtype.element_ty)
                tl.store(c_ptr + c_base + col, out)


def _pick_block_n(N):
    # Measured sweet spots on P800 XPU: N<=4096 -> next_pow2(N) (a 4096 block is
    # best for N == 4096); N >= 8192 -> 8192 (best for N == 16384 / 65536).
    return min(triton.next_power_of_2(N), BLOCK_N_CAP)


def _pick_rows(M):
    # R must divide M (the kernel has no tail guard). Larger R amortizes launch
    # for the tiny-row batched case; R = 1 is used for the 2-D case (grid = M).
    for r in (64, 32, 16, 8, 4, 2, 1):
        if M % r == 0:
            return r
    return 1


def kron(A, B):
    logger.debug("GEMS_KUNLUNXIN KRON")
    if A.dim() == 0 and B.dim() == 0:
        return A * B

    if A.numel() == 0 or B.numel() == 0:
        A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)
        output_dtype = torch.promote_types(A.dtype, B.dtype)
        return torch.empty(out_shape, device=A.device, dtype=output_dtype)

    if A.dim() == 0:
        return A.unsqueeze(0) * B
    if B.dim() == 0:
        return A * B.unsqueeze(0)

    A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)
    M1, N1 = A_prepared.shape[-2:]
    M2, N2 = B_prepared.shape[-2:]
    M, N = M1 * M2, N1 * N2

    batch_size = math.prod(out_shape[:-2]) if out_shape[:-2] else 1

    output_dtype = torch.promote_types(A.dtype, B.dtype)
    C = torch.empty(out_shape, device=A.device, dtype=output_dtype)

    C_reshaped = C.view(-1, M, N)
    A_view = A_prepared.reshape(-1, M1, N1)
    B_view = B_prepared.reshape(-1, M2, N2)

    if not A_view.is_contiguous():
        A_view = A_view.contiguous()
    if not B_view.is_contiguous():
        B_view = B_view.contiguous()

    block_n = _pick_block_n(N)
    # XPU codegen quirk: unmasked loads/stores are incorrect for BLOCK_N <= 32
    # (silent miscompile, verified on 8/16/32-lane tiles); mask them.
    need_mask = (N % block_n) != 0 or block_n <= 32

    with torch_device_fn.device(A.device):
        if batch_size == 1:
            kron_kernel[(M,)](
                A_view[0],
                B_view[0],
                C_reshaped[0],
                M,
                N1,
                M2,
                N,
                False,
                1,
                1,
                1,
                M1 * N1,
                N2,
                block_n,
                need_mask,
                1,
            )
        elif A_prepared.dim() == 4 and B_prepared.dim() == 4:
            # Generic 2-D batch decompose; both inputs are 4-D after padding.
            a0, a1, _, _ = A_prepared.shape
            b0, b1, _, _ = B_prepared.shape
            R = _pick_rows(M)
            grid = ((batch_size * M) // R,)
            kron_kernel[grid](
                A_view,
                B_view,
                C_reshaped,
                M,
                N1,
                M2,
                N,
                True,
                a1,
                b0,
                b1,
                M1 * N1,
                N2,
                block_n,
                need_mask,
                R,
            )
        else:
            # Odd batch layouts (3-D inputs, rank-5+): per-batch host loop.
            grid = (M,)
            for bt in range(batch_size):
                a_idx, b_idx = calculate_indices(
                    bt, A_prepared.shape, B_prepared.shape
                )
                kron_kernel[grid](
                    A_view[a_idx],
                    B_view[b_idx],
                    C_reshaped[bt],
                    M,
                    N1,
                    M2,
                    N,
                    False,
                    1,
                    1,
                    1,
                    M1 * N1,
                    N2,
                    block_n,
                    need_mask,
                    1,
                )

    if A.dim() <= 1 and B.dim() <= 1:
        return C.reshape(-1)

    return C