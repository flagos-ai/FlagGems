import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

config_ = CodeGenConfig(
        256,
        (512, 1, 1),
        32,
        False,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    )


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


@libentry()
@triton.jit
def cat4_dim0_2d_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    out_ptr,
    M,
    N,
    x0_stride0,
    x0_stride1,
    x1_stride0,
    x1_stride1,
    x2_stride0,
    x2_stride1,
    x3_stride0,
    x3_stride1,
    out_stride0,
    out_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    x0 = tl.load(x0_ptr + rows * x0_stride0 + cols * x0_stride1, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + rows * x1_stride0 + cols * x1_stride1, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + rows * x2_stride0 + cols * x2_stride1, mask=mask, other=0.0)
    x3 = tl.load(x3_ptr + rows * x3_stride0 + cols * x3_stride1, mask=mask, other=0.0)
    tl.store(out_ptr + rows * out_stride0 + cols * out_stride1, x0, mask=mask)
    tl.store(out_ptr + (rows + M) * out_stride0 + cols * out_stride1, x1, mask=mask)
    tl.store(out_ptr + (rows + 2 * M) * out_stride0 + cols * out_stride1, x2, mask=mask)
    tl.store(out_ptr + (rows + 3 * M) * out_stride0 + cols * out_stride1, x3, mask=mask)


@libentry()
@triton.jit
def cat4_dim1_2d_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    out_ptr,
    M,
    N,
    x0_stride0,
    x0_stride1,
    x1_stride0,
    x1_stride1,
    x2_stride0,
    x2_stride1,
    x3_stride0,
    x3_stride1,
    out_stride0,
    out_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    x0 = tl.load(x0_ptr + rows * x0_stride0 + cols * x0_stride1, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + rows * x1_stride0 + cols * x1_stride1, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + rows * x2_stride0 + cols * x2_stride1, mask=mask, other=0.0)
    x3 = tl.load(x3_ptr + rows * x3_stride0 + cols * x3_stride1, mask=mask, other=0.0)
    tl.store(out_ptr + rows * out_stride0 + cols * out_stride1, x0, mask=mask)
    tl.store(out_ptr + rows * out_stride0 + (cols + N) * out_stride1, x1, mask=mask)
    tl.store(out_ptr + rows * out_stride0 + (cols + 2 * N) * out_stride1, x2, mask=mask)
    tl.store(out_ptr + rows * out_stride0 + (cols + 3 * N) * out_stride1, x3, mask=mask)


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logging.debug("GEMS CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # Check if only one tensor is non-empty
    non_empty_tensors = [a for a in A if a.shape[dim] != 0]
    if len(non_empty_tensors) == 1:
        # If only one non-empty tensor, return it directly
        return non_empty_tensors[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    if (
        len(A) == 4
        and A[0].ndim == 2
        and all(t.is_contiguous() for t in A)
        and all(tuple(t.shape) == tuple(A[0].shape) for t in A[1:])
        and dim in (0, 1)
    ):
        M, N = A[0].shape
        out_shape = (4 * M, N) if dim == 0 else (M, 4 * N)
        out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
        grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
        if dim == 0:
            cat4_dim0_2d_kernel[grid](
                A[0],
                A[1],
                A[2],
                A[3],
                out0,
                M,
                N,
                A[0].stride(0),
                A[0].stride(1),
                A[1].stride(0),
                A[1].stride(1),
                A[2].stride(0),
                A[2].stride(1),
                A[3].stride(0),
                A[3].stride(1),
                out0.stride(0),
                out0.stride(1),
                BLOCK_M=16,
                BLOCK_N=64,
            )
        else:
            cat4_dim1_2d_kernel[grid](
                A[0],
                A[1],
                A[2],
                A[3],
                out0,
                M,
                N,
                A[0].stride(0),
                A[0].stride(1),
                A[1].stride(0),
                A[1].stride(1),
                A[2].stride(0),
                A[2].stride(1),
                A[3].stride(0),
                A[3].stride(1),
                out0.stride(0),
                out0.stride(1),
                BLOCK_M=16,
                BLOCK_N=64,
            )
        return out0

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    dim_offset = 0
    for a in A:
        out_slices = [slice(None)] * out0.ndim
        out_slices[dim] = slice(dim_offset, dim_offset + a.shape[dim])
        out0[tuple(out_slices)] = a
        dim_offset += a.shape[dim]
    return out0
