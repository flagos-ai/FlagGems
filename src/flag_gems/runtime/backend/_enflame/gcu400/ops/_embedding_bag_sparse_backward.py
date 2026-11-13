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

import flag_gems
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；只改变 launch 几何与地址计算，不改变语义。
SAMPLES_PER_PROG = 4
BLOCK_D_CAP = 256
D_THRESHOLD = 256
NUM_WARPS_SMALL = 4
NUM_WARPS_LARGE = 8
SPP_LARGE = 1
USE_INT64_OFFSET = 0
USE_CPU_SPARSE = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def _embedding_bag_sparse_backward_kernel(
    grad_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    per_sample_weights_ptr,
    values_ptr,
    num_valid_samples,
    EMBED_DIM,
    MODE: tl.constexpr,
    HAS_PSW: tl.constexpr,
    SAMPLES_PER_PROG: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_INT64_OFFSET: tl.constexpr,
):
    """每个 program 处理 SAMPLES_PER_PROG 个连续有效样本的梯度切片。"""
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    for i in tl.static_range(SAMPLES_PER_PROG):
        pid_s = pid * SAMPLES_PER_PROG + i
        if pid_s < num_valid_samples:
            bag_idx = tl.load(offset2bag_ptr + pid_s).to(tl.int32)
            go_ptrs = grad_ptr + bag_idx * EMBED_DIM + offs_d
            go = tl.load(go_ptrs, mask=mask_d, other=0.0)

            if MODE == 1 or HAS_PSW:
                go = go.to(tl.float32)

            if MODE == 1:
                bag_size = tl.load(bag_size_ptr + bag_idx).to(tl.float32)
                bag_size = tl.where(bag_size == 0.0, 1.0, bag_size)
                go = go * (1.0 / bag_size)

            if HAS_PSW:
                pw = tl.load(per_sample_weights_ptr + pid_s).to(tl.float32)
                go = go * pw

            if USE_INT64_OFFSET:
                out_ptrs = values_ptr + pid_s.to(tl.int64) * EMBED_DIM + offs_d
            else:
                # nnz * EMBED_DIM 在本算子 shape 范围内远小于 2^31，
                # 用 int32 偏移避免 GCU 上 64 位地址计算的额外指令。
                out_ptrs = values_ptr + pid_s * EMBED_DIM + offs_d
            tl.store(out_ptrs, go.to(values_ptr.dtype.element_ty), mask=mask_d)


def _sparse_output(valid_indices, values, num_weights, embedding_dim, device, dtype):
    """构造 COO 稀疏梯度；GCU 无稀疏后端时降级为 CPU 稀疏张量。"""
    if USE_CPU_SPARSE and device.type != "cpu":
        cpu_indices = valid_indices.detach().cpu().contiguous()
        cpu_values = values.detach().cpu().contiguous()
        return torch.sparse_coo_tensor(
            cpu_indices.unsqueeze(0),
            cpu_values,
            (num_weights, embedding_dim),
            device="cpu",
            dtype=dtype,
        )
    return torch.sparse_coo_tensor(
        valid_indices.unsqueeze(0),
        values,
        (num_weights, embedding_dim),
        device=device,
    )


def _embedding_bag_sparse_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
) -> torch.Tensor:
    """Compute the sparse (COO) gradient for embedding_bag backward (GCU400 tuned)."""
    logger.debug("GEMS_ENFLAME _EMBEDDING_BAG_SPARSE_BACKWARD")

    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    assert (
        grad.device.type == flag_gems.device
        and indices.device.type == flag_gems.device
        and offset2bag.device.type == flag_gems.device
        and bag_size.device.type == flag_gems.device
        and grad.device == indices.device
    ), "All inputs must be on the same device."

    device = grad.device
    embedding_dim = grad.shape[1]

    # padding_idx<0 时直接复用输入，避免 host 侧 boolean mask 与额外拷贝。
    has_padding = padding_idx is not None and padding_idx >= 0
    if has_padding:
        valid_mask = indices != padding_idx
        valid_indices = indices[valid_mask].to(torch.int64).contiguous()
        offset2bag_valid = offset2bag[valid_mask].contiguous()
        if per_sample_weights is not None:
            psw_valid = per_sample_weights.contiguous()[valid_mask]
        else:
            psw_valid = None
    else:
        valid_indices = (
            indices if indices.dtype == torch.int64 else indices.to(torch.int64)
        )
        offset2bag_valid = (
            offset2bag if offset2bag.is_contiguous() else offset2bag.contiguous()
        )
        psw_valid = (
            per_sample_weights.contiguous() if per_sample_weights is not None else None
        )

    # GCU 的 aten::_embedding_bag 在 SUM/MEAN 下可能返回空的 offset2bag，
    # ATen 稀疏反向也只对具备 sample→bag 映射的样本产出 COO 行。这里把
    # 有效样本数收敛为映射表长度并对齐三个数组：既与 ATen 语义一致，也避免
    # kernel 越界读空指针后触发异步 core dump。
    mapped_samples = min(valid_indices.numel(), offset2bag_valid.numel())
    if mapped_samples < valid_indices.numel():
        valid_indices = valid_indices[:mapped_samples].contiguous()
        offset2bag_valid = offset2bag_valid[:mapped_samples].contiguous()
        if psw_valid is not None:
            psw_valid = psw_valid[:mapped_samples].contiguous()

    nnz = mapped_samples
    grad_dtype = grad.dtype
    values = torch.empty((nnz, embedding_dim), device=device, dtype=grad_dtype)

    if nnz == 0:
        empty_indices = torch.empty(
            (1, 0), device="cpu" if USE_CPU_SPARSE else device, dtype=torch.int64
        )
        return _sparse_output(
            empty_indices[0],
            (
                values.detach().cpu()
                if USE_CPU_SPARSE and device.type != "cpu"
                else values
            ),
            num_weights,
            embedding_dim,
            device,
            grad_dtype,
        )

    psw_arg = psw_valid if psw_valid is not None else values
    has_psw = psw_valid is not None

    if embedding_dim <= D_THRESHOLD:
        block_d = min(BLOCK_D_CAP, triton.next_power_of_2(max(embedding_dim, 1)))
        samples_per_prog = SAMPLES_PER_PROG
        num_warps = NUM_WARPS_SMALL if block_d < 256 else NUM_WARPS_LARGE
        grid = (triton.cdiv(nnz, samples_per_prog), 1)
    else:
        block_d = BLOCK_D_CAP if BLOCK_D_CAP >= 128 else 128
        samples_per_prog = SPP_LARGE
        num_warps = NUM_WARPS_SMALL
        grid = (nnz, triton.cdiv(embedding_dim, block_d))

    _embedding_bag_sparse_backward_kernel[grid](
        grad,
        offset2bag_valid,
        bag_size,
        psw_arg,
        values,
        nnz,
        embedding_dim,
        MODE=mode,
        HAS_PSW=has_psw,
        SAMPLES_PER_PROG=samples_per_prog,
        BLOCK_D=block_d,
        USE_INT64_OFFSET=USE_INT64_OFFSET,
        num_warps=num_warps,
    )

    return _sparse_output(
        valid_indices, values, num_weights, embedding_dim, device, grad_dtype
    )
