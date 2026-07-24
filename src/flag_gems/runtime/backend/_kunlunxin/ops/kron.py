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

import math

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext


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


# --- Per-output-row div/mod kernel (XPU rewrite) --------------------------------
# The original kernel tiled the output as a 2D [BLOCK_M, BLOCK_N] grid and read A/B
# with  a_row = offs_m // M2 ,  b_col = offs_n % N2  etc. Two problems on XPU:
#   1) BLOCK_M = next_pow2(cdiv(M,12)) is UNBOUNDED -> M=4096 gives BLOCK_M=512 and a
#      giant [512,4096] all-int64 tile: grid collapses to ~8 programs (768 cores idle)
#      and the int64 div/mod offset math blows the MLIR up (2.9M-line IR dump) ->
#      ~0.06 GB/s catastrophe ([64,64] kron [64,64] -> 540ms, speedup 0.000).
#   2) The all-int64 2D-tile offset math is huge.
#
# Rewrite: one program owns ONE output ROW (row = i1*M2 + i2) and writes the whole
# row as a stride-1 CONTIGUOUS run. Within a row, C[row, col] where col = j1*N2 + j2
# equals A[i1, j1] * B[i2, j2]; so j1 = col // N2, j2 = col % N2 recover the A/B
# columns for each output column. Thus:
#   * store  = c_base + col, col = arange(0, BLOCK_N)   -> stride-1 block DMA (fast).
#   * A/B reads use the div/mod (col//N2, col%N2) -> discrete gather (slower), but on
#     XPU a discrete READ is ~1.5x cheaper than a discrete write, so putting the
#     div/mod on the read side and keeping the WRITE contiguous is the right trade.
# The row is looped in BLOCK_N chunks so N of any size works. N is a constexpr so the
# store index `row*N + col` is a compile-time-strided contiguous run.
#
# WHY NOT the 2D outer-product ([BLOCK_N1,N2] = A-row-slice (x) B-row): it is contiguous
# on BOTH sides in theory, but on this XPU the 2D store `c_off = row*N + j1[:,None]*N2 +
# n2[None,:]` SILENTLY MISCOMPILES (columns get duplicated/transposed; verified WRONG for
# (2,2)x(2,2) .. (8,16)x(2,3)). A 1-D per-(row,j1) [N2] store is correct only when N2 is
# large (N2=64 ok, N2=2/3/5 wrong -- small contiguous writes coalesce incorrectly). Only
# the full-row 1-D contiguous store below is correct for ALL shapes.
#
# BATCHING: the kernel handles ONE (A,B)->C matrix per launch (no in-kernel batch
# indirection). An earlier version loaded the per-batch A/B indices from a `map`
# tensor inside the kernel (`a_batch_idx = tl.load(map_ptr + ...)`) and used that
# loaded scalar as an address multiplier -> on XPU that data-dependent scalar gather
# FAULTS (KL_XID_KERNEL_EXCEPTION, err 66250/721). So batching is done on the host
# instead: kron() loops the output batches and launches this kernel once per batch
# on host-sliced contiguous 2D views. batch_size is 1 for all 2D inputs (the whole
# benchmark), so the loop is a no-op there; only genuinely batched (>=3D) inputs pay
# the extra launches, and those tensors are small.
BLOCK_N_CAP = 8192


def heur_block_n(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), BLOCK_N_CAP)


@triton.heuristics({"BLOCK_N": heur_block_n})
@triton.jit
def kron_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N1,
    M2,
    N2,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = ext.program_id(0)
    i1 = row // M2
    i2 = row % M2
    a_base = i1 * N1
    b_base = i2 * N2
    c_base = row * N
    for off in range(0, N, BLOCK_N):
        col = off + tl.arange(0, BLOCK_N)
        mask = col < N
        j1 = col // N2
        j2 = col % N2
        a = tl.load(a_ptr + a_base + j1, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + b_base + j2, mask=mask, other=0.0).to(tl.float32)
        out = (a * b).to(c_ptr.dtype.element_ty)
        tl.store(c_ptr + c_base + col, out, mask=mask)


def kron(A, B):
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

    with torch_device_fn.device(A.device):
        # One program owns ONE output row and writes it as a contiguous run.
        # grid = M. Batching is on the host: launch the kernel once per output batch
        # on host-sliced contiguous 2D views (batch_size is 1 for all 2D inputs). This
        # avoids the in-kernel data-dependent scalar gather that faults on XPU (see
        # header note).
        grid = (M,)

        for bt in range(batch_size):
            a_idx, b_idx = calculate_indices(bt, A_prepared.shape, B_prepared.shape)
            kron_kernel[grid](
                A_view[a_idx],
                B_view[b_idx],
                C_reshaped[bt],
                M,
                N1,
                M2,
                N2,
                N,
            )

    if A.dim() <= 1 and B.dim() <= 1:
        return C.reshape(-1)

    return C
