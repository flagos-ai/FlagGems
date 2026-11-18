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
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_multi_dot import _matrix_chain_order, _validate_and_prepare
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _multi_dot_mm_kernel(
    left,
    right,
    output,
    rows,
    columns,
    inner,
    stride_lm,
    stride_lk,
    stride_rk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    """Small-tile masked GEMM used for one step of a multi_dot chain.

    Mirrors the generic ``_multi_dot_mm_kernel`` (BLOCK 32, masked 2D loads,
    fp32/fp64 accumulation) so the small/ragged shapes that dominate the
    multi_dot benchmark keep the generic kernel's performance.  The single
    difference vs. the generic path is on the host side (see
    ``_matrix_multiply``): for bf16 the accumulator is written to an fp32
    buffer and converted to bf16 on the host, because the XPU's in-kernel
    fp32 -> bf16 store mis-rounds by several bf16 ULPs.
    """
    pid = ext.program_id(0)
    grid_n = tl.cdiv(columns, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N), dtype=tl.float64 if IS_FP64 else tl.float32
    )
    for start_k in range(0, tl.cdiv(inner, BLOCK_K)):
        current_k = start_k * BLOCK_K + offsets_k
        left_values = tl.load(
            left + offsets_m[:, None] * stride_lm + current_k[None, :] * stride_lk,
            mask=(offsets_m[:, None] < rows) & (current_k[None, :] < inner),
            other=0.0,
        )
        right_values = tl.load(
            right + current_k[:, None] * stride_rk + offsets_n[None, :] * stride_rn,
            mask=(current_k[:, None] < inner) & (offsets_n[None, :] < columns),
            other=0.0,
        )
        accumulator += tl.dot(left_values, right_values, allow_tf32=False)

    tl.store(
        output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on,
        accumulator,
        mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < columns),
    )


def _pick_blocks(rows, columns, inner):
    """Tile size per matmul step.

    BLOCK_K is the dominant lever on this backend: a 32-wide K tile runs 16
    sequential small dots for K = 512, which was measured ~4x slower than a
    128-wide K tile.  M/N tiles are kept modest (32/64) so the ragged test
    shapes (dims 1/16/32) do not waste whole 128-wide tiles; the 128 tier is
    only used when the dimension actually reaches it.  Masked loads/stores
    keep every partial tile correct.
    """
    bm = 128 if rows >= 128 else (64 if rows >= 64 else 32)
    bn = 128 if columns >= 128 else (64 if columns >= 64 else 32)
    bk = 128 if inner >= 128 else (64 if inner >= 64 else 32)
    return bm, bn, bk


def _launch_gemm(left, right, output, rows, columns, inner, is_fp64):
    bm, bn, bk = _pick_blocks(rows, columns, inner)
    grid = (triton.cdiv(rows, bm) * triton.cdiv(columns, bn),)
    num_warps = 4 if (bm <= 64 and bn <= 64) else 8
    with torch_device_fn.device(left.device):
        _multi_dot_mm_kernel[grid](
            left,
            right,
            output,
            rows,
            columns,
            inner,
            left.stride(0),
            left.stride(1),
            right.stride(0),
            right.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            IS_FP64=is_fp64,
            num_warps=num_warps,
        )
    return output


def _matrix_multiply(left, right, out=None):
    """Multiply without TF32 so results match ATen's multi_dot precision.

    The XPU ``tl.dot`` accumulates bf16 operands in fp32 correctly, but the
    in-kernel fp32 -> bf16 store (``tl.store`` into a bf16 buffer) rounds the
    mantissa down by up to 4 bf16 ULPs (measured ~0.25 on unit-scale data),
    which compounds through a chain and exceeds the bf16 accuracy tolerance.
    The fp32 accumulator itself is only ~1 fp32 ULP from the native result, and
    converting that fp32 buffer to bf16 on the host is exact, so bf16 goes
    through an fp32 buffer + host-side conversion.  fp16/fp32 keep the generic
    store path (fp16's store error stays within its 1e-3 tolerance).
    """
    if left.device.type == "cpu":
        raise RuntimeError("multi_dot(): Kunlunxin backend requires device tensors")

    rows, inner = left.shape
    columns = right.shape[1]
    dtype = left.dtype

    if dtype == torch.bfloat16:
        out32 = torch.empty((rows, columns), dtype=torch.float32, device=left.device)
        _launch_gemm(left, right, out32, rows, columns, inner, is_fp64=False)
        if out is not None:
            # copy_ with a dtype cast rounds fp32 -> bf16 on the host (exact);
            # this also writes strided/non-contiguous ``out`` correctly.
            out.copy_(out32)
            return out
        return out32.to(torch.bfloat16)

    if out is None:
        out = torch.empty((rows, columns), dtype=dtype, device=left.device)
    _launch_gemm(
        left, right, out, rows, columns, inner, is_fp64=(dtype == torch.float64)
    )
    return out


def _multiply_chain(arrays, splits, start, end, out=None):
    if start == end:
        return arrays[start]

    split = splits[start][end]
    left = _multiply_chain(arrays, splits, start, split)
    right = _multiply_chain(arrays, splits, split + 1, end)
    if out is not None:
        return _matrix_multiply(left, right, out=out)
    return _matrix_multiply(left, right)


def _multiply_three(arrays, out=None):
    a, b, c = arrays
    rows, inner_ab = a.shape
    inner_bc, columns = c.shape
    left_cost = rows * inner_bc * (inner_ab + columns)
    right_cost = inner_ab * columns * (rows + inner_bc)

    if left_cost > right_cost:
        right = _matrix_multiply(b, c)
        return _matrix_multiply(a, right, out=out)

    left = _matrix_multiply(a, b)
    return _matrix_multiply(left, c, out=out)


def _multi_dot_impl(arrays, out=None):
    num_tensors = len(arrays)
    if num_tensors == 2:
        return _matrix_multiply(arrays[0], arrays[1], out=out)
    if num_tensors == 3:
        return _multiply_three(arrays, out=out)

    splits = _matrix_chain_order(arrays)
    return _multiply_chain(arrays, splits, 0, num_tensors - 1, out=out)


def linalg_multi_dot(tensors):
    logger.debug("GEMS_KUNLUNXIN LINALG_MULTI_DOT")
    arrays, output_shape = _validate_and_prepare(tensors)
    result = _multi_dot_impl(arrays)
    return result.view(output_shape)


def linalg_multi_dot_out(tensors, *, out):
    logger.debug("GEMS_KUNLUNXIN LINALG_MULTI_DOT_OUT")
    arrays, output_shape = _validate_and_prepare(tensors)
    first = arrays[0]
    if out.dtype != first.dtype:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to have dtype {first.dtype} "
            f"but got {out.dtype}"
        )
    if out.device != first.device:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to be on device {first.device} "
            f"but got {out.device}"
        )

    if tuple(out.shape) != output_shape:
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(output_shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=2,
            )
        out.resize_(output_shape)

    matrix_out = out.view(arrays[0].shape[0], arrays[-1].shape[1])
    _multi_dot_impl(arrays, out=matrix_out)
    return out
