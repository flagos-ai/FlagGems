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


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N / conf.num_warps > 64 * 512:
        return False
    if BLOCK_M * BLOCK_N < 64 * 64:
        return False
    if conf.num_stages > 1:
        return False
    if conf.num_warps == 1:
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
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 2048,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    for tile_k_id in tl.range(pid_k, K, 48):
        # TODO(chenfeiyu): consider float64 add add a utility function to get accumulator type
        m = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + tile_k_id
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            m_new = tl.maximum(inp, m)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 1)
        z = tl.sum(z * tl.exp(m - m_reduced[:, None]), 1)
        m = m_reduced
        mid = m[:, None] + tl.log(z[:, None])
        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + tile_k_id
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            o = inp - mid
            tl.store(output_ptr + offset, o, mask=mask)


@libentry()
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
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    scale = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        scale += out_grad
    scale = tl.sum(scale, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_fused_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-pass backward for rows that fit in one tile.

    The generic kernel reads grad_output twice (once for the row sum, once for
    the output).  When N <= BLOCK_N the whole row can be kept in registers, so
    one pass is enough: 3 global accesses per element instead of 4.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
    out_grad = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(out_grad, 1)
    in_grad = out_grad - tl.exp(out) * scale[:, None]
    tl.store(in_grad_ptr + offsets, in_grad, mask=mask)


def log_softmax_out(self, dim, half_to_float=False, *, out):
    """K=1 时复用 gcu400 online-softmax kernel 直写 out；K>1 保留公共 non_inner。

    公共 pointwise 的 2D tiled kernel 在 K=1 上比 gcu400 kernel 慢 2.3~4.5x
    （4096² 598.8µs vs 173.3µs）；但 K>1（非 inner dim 归约）时公共的非 inner
    kernel 反过来快 7.6~14x，因此按 K 分档而不是一刀切。
    """
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_OUT")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    dtype = torch.float32 if half_to_float else self.dtype
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
    if K > 1:
        from flag_gems.ops.log_softmax import log_softmax_out as _public_out

        return _public_out(self, dim, half_to_float, out=out)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), min(48, K))
    with torch_device_fn.device(inp.device):
        log_softmax_kernel[grid](out, inp, M, N, K)
    return out


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

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        min(48, K),
    )
    with torch_device_fn.device(inp.device):
        log_softmax_kernel[grid](
            out,
            inp,
            M,
            N,
            K,
        )
    return out


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

    # Single-pass fused path for rows that fit one tile (N <= 4096).
    # Saves one full grad_output read: 4 global accesses -> 3.
    # Tile size is chosen per N: small rows want more rows per program and
    # fewer warps, wide rows want fewer rows to keep the tile in registers.
    if K == 1 and N <= 4096:
        if N <= 256:
            block_m, num_warps = 16, 2
        elif N <= 1024:
            block_m, num_warps = 32, 4
        else:
            block_m, num_warps = 8, 4
        grid = (triton.cdiv(M, block_m), K)
        with torch_device_fn.device(out.device):
            log_softmax_backward_fused_kernel[grid](
                output,
                grad_output,
                out,
                M,
                N,
                K,
                BLOCK_M=block_m,
                BLOCK_N=triton.next_power_of_2(N),
                num_warps=num_warps,
            )
        return out

    # Two-pass path for wide rows (or K > 1).  Autotuning was unstable here:
    # the first two benchmark rounds picked a 530us config while later rounds
    # picked 383us.  A fixed tile is both deterministic and faster to compile.
    block_m = 8 if output.dtype == torch.float32 else 16
    block_n = min(triton.next_power_of_2(N), 4096)
    grid = (triton.cdiv(M, block_m), K)
    with torch_device_fn.device(out.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            out,
            M,
            N,
            K,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_VJP")
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return log_softmax_backward_out(grad_output, output, dim, input_dtype, out=in_grad)
