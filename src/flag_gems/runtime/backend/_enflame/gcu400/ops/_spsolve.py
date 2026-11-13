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

import contextlib
import logging

import torch
import triton
import triton.language as tl

try:
    from triton import knobs
except ImportError:
    knobs = None

from flag_gems.ops.linalg_solve import linalg_solve_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 两个 Triton kernel 的 launch 参数集中在此，便于按 shape 分档调优。
_CSR_SCATTER_BLOCK = 64
_CSR_SCATTER_WARPS = 4
_LU_BLOCK_K = 4
_LU_NUM_WARPS = 1
_LU_NUM_STAGES = 1


@libentry()
@triton.jit
def _csr_to_dense_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    dense_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter a sparse CSR matrix into a pre-zeroed dense buffer.

    Each program handles one row. It walks the row's nnz range
    [crow[r], crow[r + 1]) in blocks and writes each value into
    dense[r, col] using a row-major linear index.
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return

    start = tl.load(crow_ptr + row).to(tl.int64)
    end = tl.load(crow_ptr + row + 1).to(tl.int64)
    row_base = row.to(tl.int64) * n_cols

    for blk in range(start, end, BLOCK_SIZE):
        offs = blk + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        cols = tl.load(col_ptr + offs, mask=mask, other=0).to(tl.int64)
        vals = tl.load(val_ptr + offs, mask=mask, other=0.0)
        # Guard against malformed column indices.
        valid = mask & (cols >= 0) & (cols < n_cols)
        tl.store(dense_ptr + row_base + cols, vals, mask=valid)


def _csr_to_dense(A, compute_device):
    """把稀疏 CSR 矩阵散射到 ``compute_device`` 上的预清零 dense buffer。

    GCU 后端不支持稀疏 CSR 存储，torch_gcu 会把 CSR 构造回落到 CPU；因此三个
    组件（crow / col / val）必须显式搬到 GCU 后才能被 Triton kernel 读写，
    dense buffer 也同样创建在计算设备上。
    """
    n_rows, n_cols = A.shape
    crow = A.crow_indices().contiguous().to(compute_device)
    col = A.col_indices().contiguous().to(compute_device)
    val = A.values().contiguous().to(compute_device)

    dense = torch.zeros((n_rows, n_cols), dtype=A.dtype, device=compute_device)

    if n_rows == 0 or n_cols == 0 or val.numel() == 0:
        return dense

    grid = (n_rows,)
    with torch_device_fn.device(compute_device):
        _csr_to_dense_kernel[grid](
            crow,
            col,
            val,
            dense,
            n_rows,
            n_cols,
            BLOCK_SIZE=_CSR_SCATTER_BLOCK,
            num_warps=_CSR_SCATTER_WARPS,
        )
    return dense


def _solve_dense_system(A_dense, B_solve):
    """用固定 ``num_warps=1`` 的 Triton LU kernel 求解单右端项的稠密系统。

    ``linalg_solve`` 的 autotune 会在 GCU400 上选中 ``num_warps=2`` 的 config，
    该 config 的 pivot 标量控制流给出错误解（实测误差 1e-3~1e-2）；直接 launch
    原始 JITFunction 并固定 ``num_warps=1`` 时误差约 1e-7，同时省去 autotune
    的 warmup / 计时与包装开销。
    """
    n = A_dense.shape[0]
    # LU kernel 按 (batch, row, col) 三段 stride 寻址，必须保持 3D batch 布局，
    # 否则 2D 的 stride(1)=1 会被当成行跨度，读到的矩阵内容完全错位。
    A_work = A_dense.reshape(1, n, n).contiguous()
    B_work = B_solve.reshape(1, n, 1).contiguous()
    LU_workspace = torch.empty_like(A_work)
    X_workspace = torch.empty_like(B_work)
    block_n = max(triton.next_power_of_2(n), 4)

    with torch.no_grad(), _no_adjust_block_size():
        linalg_solve_kernel.jit_function[(1,)](
            A_work,
            LU_workspace,
            B_work,
            X_workspace,
            n,
            1,
            A_work.stride(0),
            A_work.stride(1),
            LU_workspace.stride(0),
            LU_workspace.stride(1),
            B_work.stride(0),
            B_work.stride(1),
            X_workspace.stride(0),
            X_workspace.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=_LU_BLOCK_K,
            num_warps=_LU_NUM_WARPS,
            num_stages=_LU_NUM_STAGES,
        )
    return X_workspace.reshape(n)


def spsolve(A, B, *, left=True):
    """
    Solve the sparse linear system ``A @ X = B`` for a CSR matrix ``A``.

    Mirrors ``torch.ops.aten._spsolve``: only a 1-D right-hand side and
    ``left=True`` are supported, matching the native Sparse CSR backend's
    contract. The sparse operand is densified with a Triton scatter kernel and
    the resulting dense system is solved with the FlagGems Triton
    ``linalg_solve`` kernel (device-side), rather than a native dense solve.

    Args:
        A: Sparse CSR tensor of shape (n, n).
        B: Dense right-hand side vector of shape (n,).
        left: Must be True; ``X @ A = B`` is not supported.

    Returns:
        The dense solution vector ``X`` of shape (n,).
    """
    logger.debug("GEMS SPSOLVE")

    if A.layout != torch.sparse_csr:
        raise RuntimeError(
            f"spsolve: expected A to have sparse_csr layout, but got {A.layout}"
        )
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise RuntimeError(
            f"spsolve: expected A to be a square 2-D matrix, but got shape {tuple(A.shape)}"
        )
    if not left:
        raise RuntimeError("spsolve: only left=True is supported")
    if B.dim() != 1:
        raise RuntimeError(
            f"spsolve: expected B to be a 1-D tensor, but got shape {tuple(B.shape)}"
        )
    if B.size(0) != A.size(0):
        raise RuntimeError(
            f"spsolve: linear system size mismatch: A is {tuple(A.shape)}, "
            f"B is {tuple(B.shape)}"
        )
    # torch_gcu 把 GCU 上的 sparse CSR 构造回落到 CPU，所以 A 常驻 CPU 是正常输入；
    # 计算设备取 B 所在设备，并要求它确实是 GCU，避免 CPU 侧 B 进入 Triton launch
    # 时抛 ValueError（对外契约要求的是干净的 RuntimeError）。
    compute_device = B.device
    if compute_device.type != "gcu":
        raise RuntimeError(
            f"spsolve: expected B on a GCU device, but got {B.device}; "
            "a sparse CSR operand may stay on CPU (torch_gcu fallback), "
            "but the right-hand side must be on the compute device"
        )
    if A.device.type not in ("cpu", "gcu"):
        raise RuntimeError(f"spsolve: expected A on CPU or GCU, but got {A.device}")

    A_dense = _csr_to_dense(A, compute_device)

    # The dense Triton solve only supports fp32/fp64; solving a linear system in
    # fp16/bf16 is numerically unstable regardless. Solve in fp32 (or higher) and
    # cast the result back to the input dtype.
    out_dtype = A.dtype
    compute_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    A_solve = A_dense.to(compute_dtype)
    B_solve = B.to(compute_dtype)

    # 求解 A @ X = B：走固定 num_warps=1 的 Triton LU kernel，避开 autotune 在
    # GCU400 上选出的错误 config；AABS 关闭也由该 launch 内部统一处理。
    X = _solve_dense_system(A_solve, B_solve)
    return X.to(out_dtype)


@contextlib.contextmanager
def _no_adjust_block_size():
    """Temporarily disable Triton's auto-adjust-block-size (AABS) knob.

    AABS is a FlagTree-specific Triton extension; stock Triton has no such knob,
    in which case there is nothing to disable and this is a no-op.
    """
    autotuning = getattr(knobs, "autotuning", None)
    if autotuning is None or not hasattr(autotuning, "adjust_block_size"):
        yield
        return

    previous = autotuning.adjust_block_size
    autotuning.adjust_block_size = False
    try:
        yield
    finally:
        autotuning.adjust_block_size = previous
