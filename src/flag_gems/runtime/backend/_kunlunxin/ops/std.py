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
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)


# XPU 归约正确性约束（见 sum.py / HARNESS_SUMMARY.md §2.5）：tl.sum 只在
# tile <= 8192 时精确可靠（32768 需 buffer_size_limit 且 fp32 累加精度不足，
# 会让 std 的方差超出 fp32 的 rtol）。因此全量归约用 8192 无掩码整块 + 单独 tail。
_FLAT_CHUNK = 8192


@libentry()
@triton.jit
def _std_flat_core_kernel(X, Tmp_sum, Tmp_sum_sq, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + off).to(tl.float32)
    tl.store(Tmp_sum + pid, tl.sum(x, axis=0))
    tl.store(Tmp_sum_sq + pid, tl.sum(x * x, axis=0))


@libentry()
@triton.jit
def _std_flat_tail_kernel(X, Tmp_sum, Tmp_sum_sq, start, NTAIL, TL: tl.constexpr):
    off = tl.arange(0, TL)
    x = tl.load(X + start + off, mask=off < NTAIL, other=0.0).to(tl.float32)
    tl.store(Tmp_sum, tl.sum(x, axis=0))
    tl.store(Tmp_sum_sq, tl.sum(x * x, axis=0))


@libentry()
@triton.jit
def _std_flat_tail_staged_kernel(X, Tmp_sum, Tmp_sum_sq, TL: tl.constexpr):
    # 已零填充的 staging 缓冲，无掩码整块归约（tail > 8192 时掩码 load 不可靠）。
    off = tl.arange(0, TL)
    x = tl.load(X + off).to(tl.float32)
    tl.store(Tmp_sum, tl.sum(x, axis=0))
    tl.store(Tmp_sum_sq, tl.sum(x * x, axis=0))


@libentry()
@triton.jit
def _std_flat_merge_kernel(
    Tmp_sum, Tmp_sum_sq, Out, N, correction, nb, NLANES: tl.constexpr
):
    off = tl.arange(0, NLANES)
    mask = off < nb
    s = tl.load(Tmp_sum + off, mask=mask, other=0.0).to(tl.float32)
    sq = tl.load(Tmp_sum_sq + off, mask=mask, other=0.0).to(tl.float32)
    total_sum = tl.sum(s, axis=0)
    total_sum_sq = tl.sum(sq, axis=0)
    mean = total_sum / N
    var = (total_sum_sq / N) - (mean * mean)
    var = var * N / tl.maximum(N - correction, 1.0)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Out, std_dev.to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["correction"])
def _std_dim_row_kernel(
    Out, X, M, N, correction, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    rows_c = tl.where(rows < M, rows, M - 1)
    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(X + rows_c * N + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc += a
        sum_sq_acc += a * a
    total_sum = tl.sum(sum_acc, axis=1)[:, None]
    total_sum_sq = tl.sum(sum_sq_acc, axis=1)[:, None]
    mean = total_sum / N
    var = (total_sum_sq / N) - (mean * mean)
    var = var * N / tl.maximum(N - correction, 1.0)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Out + rows, std_dev.to(Out.dtype.element_ty), row_mask)


def _std_dim_dispatch(out, x_contiguous, M, N, K, effective_correction):
    # dim_compress 保证归约维落在 trailing axis => 恒为 contiguous (M, N) 内归约。
    # 用 2D tile（BLOCK_M 行/CTA）替代原 1 行/CTA 的 1D tile，减少 CTA 数、提升访存。
    BLOCK_M = 128
    BLOCK_N = min(8192, triton.next_power_of_2(N))
    with torch_device_fn.device(x_contiguous.device):
        grid = (triton.cdiv(M, BLOCK_M), 1, 1)
        _std_dim_row_kernel[grid](
            out, x_contiguous, M, N, effective_correction, BLOCK_M, BLOCK_N
        )


def _launch_std_flat(x, out, N, correction):
    # 全量归约：32768 无掩码整块（带 buffer_size_limit=2048）并行 map，单次掩码
    # tail（<=8192），再单次 merge。对齐 sum.py 的验证模式，避免原 1024 小块 +
    # 掩码大 tile 的低效与不可靠。
    CHUNK = _FLAT_CHUNK
    nfull = N // CHUNK
    tail = N % CHUNK
    nb = nfull + (1 if tail else 0)
    tmp_sum = torch.empty((nb,), dtype=torch.float32, device=x.device)
    tmp_sum_sq = torch.empty((nb,), dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        if nfull:
            _std_flat_core_kernel[(nfull, 1, 1)](
                x, tmp_sum, tmp_sum_sq, CHUNK, buffer_size_limit=2048
            )
        if tail and tail <= 8192:
            _std_flat_tail_kernel[(1, 1, 1)](
                x,
                tmp_sum[nfull : nfull + 1],
                tmp_sum_sq[nfull : nfull + 1],
                nfull * CHUNK,
                tail,
                triton.next_power_of_2(tail),
            )
        elif tail:
            TL = triton.next_power_of_2(tail)
            staged = torch.zeros((TL,), dtype=x.dtype, device=x.device)
            torch.ops.aten._copy_from(x[nfull * CHUNK : N], staged[:tail], False)
            _std_flat_tail_staged_kernel[(1, 1, 1)](
                staged,
                tmp_sum[nfull : nfull + 1],
                tmp_sum_sq[nfull : nfull + 1],
                TL,
                buffer_size_limit=2048,
            )
        _std_flat_merge_kernel[(1, 1, 1)](
            tmp_sum,
            tmp_sum_sq,
            out,
            N,
            correction,
            nb,
            triton.next_power_of_2(nb),
        )


def std(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN STD")
    effective_correction = 1.0 if correction is None else float(correction)
    original_shape = x.shape
    input_ndim = x.ndim

    if dim is None:
        N = x.numel()
        if N == 0 or N - effective_correction <= 0:
            return torch.full([], float("nan"), device=x.device, dtype=x.dtype)
        if N == 1 and effective_correction == 0.0:
            out = torch.zeros([], device=x.device, dtype=x.dtype)
            return out.view([1] * input_ndim) if keepdim else out

        out = torch.empty([], device=x.device, dtype=x.dtype)
        _launch_std_flat(x.contiguous(), out, N, effective_correction)
        return out.view([1] * input_ndim) if keepdim else out

    if isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)
    dim_list_normalized = [d % input_ndim for d in dim_list]

    # Route EVERY dim reduction (single-dim AND multi-dim) through dim_compress so
    # the reduced dims land on the trailing axis => it is always a contiguous
    # (M, N) inner reduction (K == 1). We only ever launch the @libentry-cached
    # _std_dim_kernel_inner. This (a) avoids the giant 2D tile + heuristic-supplied
    # launch param IR explosion of the old _std_fused_dim_kernel path
    # (ir-std-dev5.log = 7.7M lines) and (b) avoids the non_inner (K>1) softmax
    # kernel, which was numerically wrong on XPU (std ~sqrt(K)x too small).
    x_view = dim_compress(x, dim_list_normalized)
    N = 1
    for d in dim_list_normalized:
        N *= original_shape[d]
    M = x.numel() // N

    output_shape_kept = list(original_shape)
    for d in dim_list_normalized:
        output_shape_kept[d] = 1

    if M * N > 0 and (N - effective_correction <= 0):
        final_shape = [
            s for i, s in enumerate(original_shape) if i not in dim_list_normalized
        ]
        return torch.full(
            final_shape if not keepdim else output_shape_kept,
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )
    if N == 1 and effective_correction == 0.0:
        final_shape = [
            s for i, s in enumerate(original_shape) if i not in dim_list_normalized
        ]
        return torch.zeros(
            final_shape if not keepdim else output_shape_kept,
            device=x.device,
            dtype=x.dtype,
        )

    out = torch.empty(output_shape_kept, device=x.device, dtype=x.dtype)
    if M * N == 0:
        return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out

    _std_dim_dispatch(out.view(-1), x_view, M, N, 1, effective_correction)
    return out.squeeze(dim=tuple(dim_list_normalized)) if not keepdim else out
