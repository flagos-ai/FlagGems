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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

GRID_DIM_X = 24


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ONE_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    if ONE_TILE:
        for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
            out_block = tl.make_block_ptr(
                base=out_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            grad_block = tl.make_block_ptr(
                base=out_grad_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            out_tile = tl.load(
                out_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            grad_tile = tl.load(
                grad_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            scale = tl.sum(out_tile * grad_tile, axis=1)
            in_grad_tile = out_tile * (grad_tile - scale[:, None])
            in_grad_block = tl.make_block_ptr(
                base=in_grad_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            tl.store(
                in_grad_block,
                in_grad_tile.to(in_grad_ptr.dtype.element_ty),
                boundary_check=(0, 1),
            )
    else:
        for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
            scale = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for n_start in tl.range(0, N, BLOCK_N):
                out_block = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                grad_block = tl.make_block_ptr(
                    base=out_grad_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                out_tile = tl.load(
                    out_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                grad_tile = tl.load(
                    grad_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                scale += out_tile * grad_tile
            scale = tl.sum(scale, axis=1)

            for n_start in tl.range(0, N, BLOCK_N):
                out_block = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                grad_block = tl.make_block_ptr(
                    base=out_grad_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                in_grad_block = tl.make_block_ptr(
                    base=in_grad_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                out_tile = tl.load(
                    out_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                grad_tile = tl.load(
                    grad_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                in_grad_tile = out_tile * (grad_tile - scale[:, None])
                tl.store(
                    in_grad_block,
                    in_grad_tile.to(in_grad_ptr.dtype.element_ty),
                    boundary_check=(0, 1),
                )


@libentry()
@triton.jit(do_not_specialize=["N"])
def softmax_backward_scale_kernel(
    out_ptr,
    out_grad_ptr,
    partial_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Stage 1 for M=1: compute partial scale sums across programs."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for n_start in tl.range(pid * BLOCK_N, N, num_programs * BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        out_val = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        grad_val = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += out_val * grad_val
    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))


@libentry()
@triton.jit(do_not_specialize=["N", "num_partials"])
def softmax_backward_grad_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    partial_ptr,
    N,
    num_partials,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Stage 2 for M=1: read partial sums, compute total scale, compute in_grad."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    p_offsets = tl.arange(0, BLOCK_P)
    p_mask = p_offsets < num_partials
    partials = tl.load(partial_ptr + p_offsets, mask=p_mask, other=0.0)
    scale = tl.sum(partials, axis=0)

    for n_start in tl.range(pid * BLOCK_N, N, num_programs * BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        out_val = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        grad_val = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        in_grad_val = out_val * (grad_val - scale)
        tl.store(
            in_grad_ptr + n_offsets,
            in_grad_val.to(in_grad_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_k_programs = tl.num_programs(1)

    for k_start in tl.range(pid_k * BLOCK_K, K, num_k_programs * BLOCK_K):
        scale = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        for n_start in tl.range(0, N, BLOCK_N):
            out_block = tl.make_block_ptr(
                base=out_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            grad_block = tl.make_block_ptr(
                base=out_grad_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            out_tile = tl.load(
                out_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            grad_tile = tl.load(
                grad_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            scale += out_tile * grad_tile
        scale = tl.sum(scale, axis=0)

        for n_start in tl.range(0, N, BLOCK_N):
            out_block = tl.make_block_ptr(
                base=out_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            grad_block = tl.make_block_ptr(
                base=out_grad_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            in_grad_block = tl.make_block_ptr(
                base=in_grad_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            out_tile = tl.load(
                out_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            grad_tile = tl.load(
                grad_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            in_grad_tile = out_tile * (grad_tile - scale[None, :])
            tl.store(
                in_grad_block,
                in_grad_tile.to(in_grad_ptr.dtype.element_ty),
                boundary_check=(0, 1),
            )


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS_ENFLAME SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected grad_input dtype {input_dtype}, "
            f"got {grad_input.dtype}"
        )
    K = output.numel() // M // N

    with torch_device_fn.device(grad_input.device):
        if K > 1:
            BLOCK_K = min(triton.next_power_of_2(K), 1024)
            BLOCK_N = max(1, min(64, 32768 // BLOCK_K))
            num_k_programs = min(GRID_DIM_X, triton.cdiv(K, BLOCK_K))
            grid = (M, num_k_programs)
            softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
                K,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                num_warps=4,
            )
        elif M == 1 and N > 2048:
            BLOCK_N = 4096
            num_programs = min(GRID_DIM_X, triton.cdiv(N, BLOCK_N))
            partial = torch.empty(
                num_programs, dtype=torch.float32, device=grad_input.device
            )
            softmax_backward_scale_kernel[(num_programs,)](
                output,
                grad_output,
                partial,
                N,
                BLOCK_N=BLOCK_N,
                num_warps=1,
            )
            BLOCK_P = triton.next_power_of_2(num_programs)
            softmax_backward_grad_kernel[(num_programs,)](
                output,
                grad_output,
                grad_input,
                partial,
                N,
                num_programs,
                BLOCK_N=BLOCK_N,
                BLOCK_P=BLOCK_P,
                num_warps=1,
            )
        else:
            BLOCK_N = min(triton.next_power_of_2(N), 2048)
            BLOCK_M = max(1, min(128, 32768 // BLOCK_N))
            ONE_TILE = N <= BLOCK_N
            num_programs = min(GRID_DIM_X, triton.cdiv(M, BLOCK_M))
            grid = (num_programs,)
            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ONE_TILE=ONE_TILE,
                num_warps=1,
            )
    return grad_input


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_ENFLAME SOFTMAX_VJP")

    in_grad = torch.empty_like(output, dtype=input_dtype)
    return softmax_backward_out(
        grad_output, output, dim, input_dtype, grad_input=in_grad
    )


# ------------------------  forward  -------------------------------
# GCU400 forward softmax is organized by "row": total programs fixed at GRID_DIM_FWD,
# each program strides over several rows in-kernel (row count no longer decides CTA count).
# Rows are split by BLOCK_N; when N is short, pack several rows into one 2D tile to avoid one launch per row.
GRID_DIM_FWD = 48
# Byte budget of a single tile: the in-row contiguous segment length is derived from it (128 KiB measured best
# bandwidth).
FWD_TILE_BYTES = 128 << 10
# Beyond this row length switch to 2D block tiling: full-row reduction width drops bandwidth to 1/3.
BLOCK_TILING_MIN_N = 8192
# Columns per block in 2D block tiling: reduction width is fixed here to avoid wide reductions slowing bandwidth.
BLOCK_TILING_COLS = 4096
# Element cap per tile in block tiling: fp32 intermediates take 128 KiB, larger exceeds DSM.
BLOCK_TILING_TILE_ELEMS = 32768
# Element cap of the single-pass row kernel tile: beyond it register pressure outweighs the parallelism gain.
ROW_TILE_ELEMS = 32768
# log2(e) multiplier for exp via exp2 must be a tl.constexpr to be referenced inside the kernel.
LOG2E = tl.constexpr(1.4426950408889634)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def softmax_forward_row_kernel(
    out_ptr,
    in_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """N fits in one BLOCK_N: one tile per row (or several rows), single-pass read/write."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        mask = (m_offsets[:, None] < M) & n_mask[None, :]
        offsets = m_offsets[:, None] * N + n_offsets[None, :]
        inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(inp, 1)
        exp_val = tl.exp2((inp - row_max[:, None]) * LOG2E)
        inv_sum = 1.0 / tl.sum(exp_val, 1)
        out = (exp_val * inv_sum[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def softmax_forward_row_multipass_kernel(
    out_ptr,
    in_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Row longer than BLOCK_N: pass one computes max/sum online, pass two recomputes exp and writes out."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    n_offsets = tl.arange(0, BLOCK_N)
    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        row_base = m_offsets[:, None] * N
        row_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        row_sum = tl.zeros([BLOCK_M], tl.float32)
        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            offsets = row_base + n_index[None, :]
            mask = m_mask[:, None] & (n_index[None, :] < N)
            inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf"))
            inp = inp.to(tl.float32)
            m_new = tl.maximum(row_max, tl.max(inp, 1))
            # When a whole row is -inf keep the original value, avoiding NaN from -inf - (-inf).
            all_neg_inf = m_new == float("-inf")
            shifted = tl.sum(tl.exp2((inp - m_new[:, None]) * LOG2E), 1)
            row_sum = tl.where(
                all_neg_inf,
                row_sum,
                row_sum * tl.exp2((row_max - m_new) * LOG2E) + shifted,
            )
            row_max = m_new
        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            offsets = row_base + n_index[None, :]
            mask = m_mask[:, None] & (n_index[None, :] < N)
            inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf"))
            exp_val = tl.exp2((inp.to(tl.float32) - row_max[:, None]) * LOG2E)
            out = exp_val * (1.0 / row_sum)[:, None]
            tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def softmax_forward_split_partial_kernel(
    in_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """M == 1 with an extremely long row: each program handles one segment, producing local max / sum."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    n_offsets = tl.arange(0, BLOCK_N)
    # Accumulate with scalar state: state size is independent of BLOCK_N, so tiles can grow to 128 KiB.
    acc_max = float("-inf")
    acc_sum = 0.0
    for n_start in tl.range(pid * BLOCK_N, N, num_programs * BLOCK_N):
        offsets = n_start + n_offsets
        mask = offsets < N
        inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        seg_max = tl.max(inp, 0)
        merged_max = tl.maximum(acc_max, seg_max)
        # All -inf segments do not participate, avoiding NaN from -inf - (-inf).
        seg_sum = tl.sum(tl.exp(inp - merged_max), 0)
        acc_sum = tl.where(
            merged_max == float("-inf"),
            acc_sum,
            acc_sum * tl.exp(acc_max - merged_max) + seg_sum,
        )
        acc_max = merged_max
    tl.store(partial_max_ptr + pid, acc_max)
    tl.store(partial_sum_ptr + pid, acc_sum)


@libentry()
@triton.jit(do_not_specialize=["N"])
def softmax_forward_split_final_kernel(
    out_ptr,
    in_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Reduce segment max / sum into global values, then recompute this segment's output."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    p_offsets = tl.arange(0, BLOCK_P)
    p_mask = p_offsets < num_programs
    part_max = tl.load(partial_max_ptr + p_offsets, mask=p_mask, other=-float("inf"))
    part_sum = tl.load(partial_sum_ptr + p_offsets, mask=p_mask, other=0.0)
    g_max = tl.max(part_max, 0)
    pair_sum = tl.where(
        g_max == float("-inf"), 0.0, part_sum * tl.exp(part_max - g_max)
    )
    g_sum = tl.sum(pair_sum, 0)
    n_offsets = tl.arange(0, BLOCK_N)
    for n_start in tl.range(pid * BLOCK_N, N, num_programs * BLOCK_N):
        offsets = n_start + n_offsets
        mask = offsets < N
        inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        out = tl.exp(inp - g_max) / g_sum
        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "num_n"])
def softmax_forward_block_partial_kernel(
    in_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    M,
    N,
    num_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Stage one: each (BLOCK_M, BLOCK_N) block produces local max / sum for that row segment."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    total_blocks = num_m_blocks * num_n
    n_offsets = tl.arange(0, BLOCK_N)
    for work in tl.range(pid, total_blocks, num_programs):
        m_block = work // num_n
        n_block = work % num_n
        m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        n_index = n_block * BLOCK_N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_index[None, :] < N)
        offsets = m_offsets[:, None] * N + n_index[None, :]
        inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf"))
        inp = inp.to(tl.float32)
        seg_max = tl.max(inp, 1)
        seg_sum = tl.sum(tl.exp2((inp - seg_max[:, None]) * LOG2E), 1)
        out_index = m_offsets * num_n + n_block
        row_mask = m_offsets < M
        tl.store(partial_max_ptr + out_index, seg_max, mask=row_mask)
        tl.store(partial_sum_ptr + out_index, seg_sum, mask=row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "num_n"])
def softmax_forward_block_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    row_max_ptr,
    row_sum_ptr,
    M,
    num_n,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Stage two: merge each row's num_n local values into global max / sum (tiny data volume)."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    p_offsets = tl.arange(0, BLOCK_P)
    for m_block in tl.range(pid, num_m_blocks, num_programs):
        m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offsets < M
        row_base = m_offsets[:, None] * num_n
        g_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        g_sum = tl.zeros([BLOCK_M], tl.float32)
        # num_n may far exceed BLOCK_P (extremely long rows), so merge in rolling BLOCK_P chunks.
        for p_start in tl.range(0, num_n, BLOCK_P):
            p_index = p_start + p_offsets
            mask = row_mask[:, None] & (p_index[None, :] < num_n)
            base = row_base + p_index[None, :]
            part_max = tl.load(partial_max_ptr + base, mask=mask, other=-float("inf"))
            part_sum = tl.load(partial_sum_ptr + base, mask=mask, other=0.0)
            merged = tl.maximum(g_max, tl.max(part_max, 1))
            shifted = tl.sum(
                tl.where(
                    part_max == float("-inf"),
                    0.0,
                    part_sum * tl.exp(part_max - merged[:, None]),
                ),
                1,
            )
            g_sum = tl.where(
                merged == float("-inf"),
                g_sum,
                g_sum * tl.exp(g_max - merged) + shifted,
            )
            g_max = merged
        tl.store(row_max_ptr + m_offsets, g_max, mask=row_mask)
        tl.store(row_sum_ptr + m_offsets, g_sum, mask=row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "num_n"])
def softmax_forward_block_output_kernel(
    out_ptr,
    in_ptr,
    row_max_ptr,
    row_sum_ptr,
    M,
    N,
    num_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Stage three: recompute exp((x - max)) / sum per block and write out."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    total_blocks = num_m_blocks * num_n
    n_offsets = tl.arange(0, BLOCK_N)
    for work in tl.range(pid, total_blocks, num_programs):
        m_block = work // num_n
        n_block = work % num_n
        m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        n_index = n_block * BLOCK_N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_index[None, :] < N)
        offsets = m_offsets[:, None] * N + n_index[None, :]
        row_mask = m_offsets < M
        row_max = tl.load(row_max_ptr + m_offsets, mask=row_mask, other=0.0)
        row_sum = tl.load(row_sum_ptr + m_offsets, mask=row_mask, other=1.0)
        inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf"))
        exp_val = tl.exp2((inp.to(tl.float32) - row_max[:, None]) * LOG2E)
        out = exp_val * (1.0 / row_sum)[:, None]
        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def softmax_forward_kernel_non_inner(
    out_ptr,
    in_ptr,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Non-inner reduction dim (K > 1): grid strides over K, each program reduces the whole N dim online.

    grid.y is bounded by GCU hardware (255), so the K direction uses only GRID_DIM_X
    programs, each striding in-kernel by `num_k_programs * BLOCK_K`,
    consistent with `softmax_backward_kernel_non_inner`.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_k_programs = tl.num_programs(1)

    for k_start in tl.range(pid_k * BLOCK_K, K, num_k_programs * BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        acc_max = tl.full([BLOCK_K], float("-inf"), tl.float32)
        acc_sum = tl.zeros([BLOCK_K], tl.float32)
        for n_start in tl.range(0, N, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            mask = (n_offsets[:, None] < N) & k_mask[None, :]
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(acc_max, tl.max(inp, 0))
            # When a whole column is -inf keep the original value, avoiding NaN from -inf - (-inf).
            all_neg_inf = m_new == float("-inf")
            shifted = tl.sum(tl.exp2((inp - m_new[None, :]) * LOG2E), 0)
            acc_sum = tl.where(
                all_neg_inf,
                acc_sum,
                acc_sum * tl.exp2((acc_max - m_new) * LOG2E) + shifted,
            )
            acc_max = m_new

        for n_start in tl.range(0, N, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            mask = (n_offsets[:, None] < N) & k_mask[None, :]
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            exp_val = tl.exp2((inp - acc_max[None, :]) * LOG2E)
            out = exp_val * (1.0 / acc_sum)[None, :]
            tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["numel"])
def softmax_single_kernel(out_ptr, in_ptr, numel, BLOCK: tl.constexpr):
    """Reduction dim length is 1: result is always 1; -inf elements give NaN by 0/0 semantics (same as aten)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    inp = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(inp == float("-inf"), float("nan"), 1.0)
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_ENFLAME SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        out.zero_()
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]

    source = self.contiguous()
    dtype = torch.float32 if half_to_float else source.dtype
    if tuple(out.shape) != tuple(source.shape):
        out.resize_(source.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")

    K = source.numel() // M // N

    if N == 1:
        # Reduction dim length is 1: one elementwise kernel suffices,
        # an order of magnitude faster than the generic two-pass implementation.
        block = 8192
        grid = (triton.cdiv(source.numel(), block),)
        with torch_device_fn.device(source.device):
            softmax_single_kernel[grid](
                out, source, source.numel(), BLOCK=block, num_warps=2
            )
        return out

    if K > 1:
        BLOCK_K = min(triton.next_power_of_2(K), 1024)
        BLOCK_N = max(1, min(64, 32768 // BLOCK_K))
        num_k_programs = min(GRID_DIM_X, triton.cdiv(K, BLOCK_K))
        grid = (M, num_k_programs)
        with torch_device_fn.device(source.device):
            softmax_forward_kernel_non_inner[grid](
                out,
                source,
                M,
                N,
                K,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                num_warps=4,
            )
        return out

    with torch_device_fn.device(source.device):
        element_size = source.element_size()
        block_n_cap = max(64, FWD_TILE_BYTES // element_size)
        if M == 1 and N > block_n_cap:
            # One extremely long row: block tiling's row dim degenerates to 1, the tile is only BLOCK_N wide,
            # bandwidth drops an order of magnitude, so use segmented reduction (local max/sum + one global merge).
            num_programs = GRID_DIM_FWD
            split_block_n = min(
                block_n_cap,
                max(1024, triton.next_power_of_2(triton.cdiv(N, num_programs))),
            )
            partial_max = torch.empty(
                num_programs, dtype=torch.float32, device=source.device
            )
            partial_sum = torch.empty(
                num_programs, dtype=torch.float32, device=source.device
            )
            softmax_forward_split_partial_kernel[(num_programs,)](
                source,
                partial_max,
                partial_sum,
                N,
                BLOCK_N=split_block_n,
                num_warps=1,
            )
            block_p = triton.next_power_of_2(num_programs)
            softmax_forward_split_final_kernel[(num_programs,)](
                out,
                source,
                partial_max,
                partial_sum,
                N,
                BLOCK_N=split_block_n,
                BLOCK_P=block_p,
                num_warps=1,
            )
        elif M > 1 and N > BLOCK_TILING_MIN_N:
            # Several long rows: 2D block tiling. A tile covers BLOCK_M rows x BLOCK_N columns,
            # row reduction width is bounded by BLOCK_N, and cross-block results are merged in stage two.
            block_n = min(BLOCK_TILING_COLS, triton.next_power_of_2(N))
            batch_rows = triton.next_power_of_2(triton.cdiv(M, GRID_DIM_FWD))
            block_m = max(1, min(BLOCK_TILING_TILE_ELEMS // block_n, batch_rows))
            num_n = triton.cdiv(N, block_n)
            num_m_blocks = triton.cdiv(M, block_m)
            total_blocks = num_m_blocks * num_n
            partial_max = torch.empty(
                M * num_n, dtype=torch.float32, device=source.device
            )
            partial_sum = torch.empty(
                M * num_n, dtype=torch.float32, device=source.device
            )
            row_max = torch.empty(M, dtype=torch.float32, device=source.device)
            row_sum = torch.empty(M, dtype=torch.float32, device=source.device)
            softmax_forward_block_partial_kernel[(min(GRID_DIM_FWD, total_blocks),)](
                source,
                partial_max,
                partial_sum,
                M,
                N,
                num_n,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=2,
            )
            reduce_block_m = 64
            block_p = min(1024, triton.next_power_of_2(num_n))
            softmax_forward_block_reduce_kernel[
                (min(GRID_DIM_FWD, triton.cdiv(M, reduce_block_m)),)
            ](
                partial_max,
                partial_sum,
                row_max,
                row_sum,
                M,
                num_n,
                BLOCK_M=reduce_block_m,
                BLOCK_P=block_p,
                num_warps=1,
            )
            softmax_forward_block_output_kernel[(min(GRID_DIM_FWD, total_blocks),)](
                out,
                source,
                row_max,
                row_sum,
                M,
                N,
                num_n,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=2,
            )
        elif N <= block_n_cap:
            block_n = triton.next_power_of_2(N)
            # Few in-row elements: pack several rows into one tile; the batch row count must be a power of two.
            batch_rows = triton.next_power_of_2(triton.cdiv(M, GRID_DIM_FWD))
            block_m = max(1, min(ROW_TILE_ELEMS // block_n, batch_rows))
            grid = (min(GRID_DIM_FWD, triton.cdiv(M, block_m)),)
            softmax_forward_row_kernel[grid](
                out,
                source,
                M,
                N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=2,
            )
        else:
            # Fallback: single row with a row length outside the tile cap, handled by segmented two-pass processing.
            grid = (min(GRID_DIM_FWD, M),)
            softmax_forward_row_multipass_kernel[grid](
                out,
                source,
                M,
                N,
                BLOCK_M=1,
                BLOCK_N=block_n_cap,
                num_warps=2,
            )
    return out


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_ENFLAME SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        out = torch.empty_like(self)
        out.zero_()
        return out

    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self, dtype=dtype)
    if self.shape[dim % self.ndim] == 1:
        # Reduction dim length is 1: go directly to the dedicated kernel, avoiding generic dispatch.
        source = self.contiguous()
        block = 8192
        grid = (triton.cdiv(source.numel(), block),)
        with torch_device_fn.device(source.device):
            softmax_single_kernel[grid](
                out, source, source.numel(), BLOCK=block, num_warps=2
            )
        return out
    return softmax_out(self, dim, half_to_float, out=out)
