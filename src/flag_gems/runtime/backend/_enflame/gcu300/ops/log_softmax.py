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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

NUM_SIPS = 24


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N / conf.num_warps > 256 * 512:
        return False
    if BLOCK_M * BLOCK_N / conf.num_warps < 4096:
        return False
    if conf.num_warps > 1:
        return False
    return True


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("log_softmax"))),
    key=["M", "N"],
)
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Inner-dim kernel (K==1): Fixed Grid grid-stride over (m_block, k) pairs."""
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    total_blocks = num_m_blocks * K

    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    for block_id in range(pid, total_blocks, num_pid):
        pid_m = block_id // K
        pid_k = block_id % K

        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M

        # --- Pass 1: online max + sum_exp (first chunk as init) ---
        n_offset0 = tl.arange(0, BLOCK_N)
        offset0 = m_offset[:, None] * N * K + n_offset0[None, :] * K + pid_k
        mask0 = row_mask[:, None] & (n_offset0[None, :] < N)
        inp0 = tl.load(input_ptr + offset0, mask=mask0, other=-float("inf")).to(
            tl.float32
        )
        m = inp0
        z = tl.where(inp0 == float("-inf"), 0.0, 1.0)

        if N > BLOCK_N:
            for start_n in tl.range(BLOCK_N, N, BLOCK_N, num_stages=num_stages):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
                mask = row_mask[:, None] & (n_offset[None, :] < N)
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                m_new = tl.maximum(inp, m)
                z = z * tl.math.exp2((m - m_new) * LOG2E) + tl.math.exp2(
                    (inp - m_new) * LOG2E
                )
                m = m_new

        m_reduced = tl.max(m, 1)
        z = tl.sum(z * tl.math.exp2((m - m_reduced[:, None]) * LOG2E), 1)
        m = m_reduced
        mid = m + tl.math.log2(z) * LN2

        # --- Pass 2: compute and store output ---
        for start_n in tl.range(0, N, BLOCK_N, num_stages=num_stages):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
            mask = row_mask[:, None] & (n_offset[None, :] < N)
            inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            o = inp - mid[:, None]
            tl.store(output_ptr + offset, o, mask=mask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def log_softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """Non-inner-dim kernel (K>1): Fixed Grid, TILE_K contiguous K values per program."""
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_k_blocks = tl.cdiv(K, TILE_K)
    total_blocks = M * num_k_blocks

    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    for block_id in range(pid, total_blocks, num_pid):
        pid_m = block_id // num_k_blocks
        pid_k = block_id % num_k_blocks

        k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)
        k_mask = k_offsets < K

        if ONE_TILE_PER_CTA:
            n_offsets = tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_mask[None, :])
            inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m = tl.max(inp, 0)
            e = tl.math.exp2((inp - m[None, :]) * LOG2E)
            z = tl.sum(e, 0)
            out = inp - m[None, :] - tl.math.log2(z) * LN2
            tl.store(output_ptr + offset, out, mask=mask)
        else:
            m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
            z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

            for start_n in range(0, N, TILE_N):
                n_offsets = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
                mask = (n_offsets[:, None] < N) & (k_mask[None, :])
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                m_new = tl.maximum(inp, m)
                all_neg_inf = m_new == float("-inf")
                z = tl.where(
                    all_neg_inf,
                    z,
                    z * tl.math.exp2((m - m_new) * LOG2E)
                    + tl.math.exp2((inp - m_new) * LOG2E),
                )
                m = m_new

            m_reduced = tl.max(m, 0)
            z = tl.sum(z * tl.math.exp2((m - m_reduced[None, :]) * LOG2E), 0)
            m = m_reduced
            mid = m + tl.math.log2(z) * LN2

            for start_n in range(0, N, TILE_N):
                n_offsets = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
                mask = (n_offsets[:, None] < N) & (k_mask[None, :])
                inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                out = inp - mid[None, :]
                tl.store(output_ptr + offset, out, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.jit
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 3,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    total_blocks = num_m_blocks * K

    LOG2E: tl.constexpr = 1.4426950408889634

    for block_id in range(pid, total_blocks, num_pid):
        pid_m = block_id // K
        pid_k = block_id % K

        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M

        scale = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
            mask = row_mask[:, None] & (n_offset[None, :] < N)
            out_grad = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
            scale += tl.sum(out_grad, 1)

        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
            mask = row_mask[:, None] & (n_offset[None, :] < N)
            out_val = tl.load(out_ptr + offset, mask=mask).to(tl.float32)
            out_grad = tl.load(out_grad_ptr + offset, mask=mask).to(tl.float32)
            in_grad = out_grad - tl.math.exp2(out_val * LOG2E) * scale[:, None]
            tl.store(in_grad_ptr + offset, in_grad, mask=mask)


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    if N == 1:
        return torch.zeros_like(inp, dtype=dtype)
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (min(NUM_SIPS, M * triton.cdiv(K, meta["TILE_K"])),)
            log_softmax_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (min(NUM_SIPS, triton.cdiv(M, meta["BLOCK_M"]) * K),)
            log_softmax_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
            )
    return out


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    if out.dtype != dtype:
        raise RuntimeError(
            f"_log_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    if N == 1:
        out.zero_()
        return out
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (min(NUM_SIPS, M * triton.cdiv(K, meta["TILE_K"])),)
            log_softmax_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (min(NUM_SIPS, triton.cdiv(M, meta["BLOCK_M"]) * K),)
            log_softmax_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
            )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    grid = lambda meta: (min(NUM_SIPS, triton.cdiv(M, meta["BLOCK_M"]) * K),)
    with torch_device_fn.device(in_grad.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            K,
        )
    return in_grad


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    if tuple(out.shape) != tuple(output.shape):
        out.resize_(output.shape)
    if out.dtype != input_dtype:
        raise RuntimeError(
            f"_log_softmax_backward_data.out: expected out dtype {input_dtype}, got {out.dtype}"
        )
    K = output.numel() // M // N

    grid = lambda meta: (min(NUM_SIPS, triton.cdiv(M, meta["BLOCK_M"]) * K),)
    with torch_device_fn.device(out.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            out,
            M,
            N,
            K,
        )
    return out
