import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.softmax import softmax_backward as common_softmax_backward
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@triton.jit
def softmax_kernel_spacemit(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0) * ROW_SIZE
    element_ty = output_ptr.type.element_ty

    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            denominator = tl.zeros((1,), dtype=tl.float32)
            row_max = tl.full((COL_SIZE,), value=-float("inf"), dtype=tl.float32)

            for col_idx in range(0, n_cols, COL_SIZE):
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                row = tl.load(
                    input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                ).to(tl.float32)
                row_max = tl.maximum(row, row_max)

            row_max_total = tl.max(row_max, axis=0)

            for col_idx in range(0, n_cols, COL_SIZE):
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                output_block_ptr = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                row = tl.load(
                    input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                ).to(tl.float32)
                numerator = exp(row - row_max_total)
                denominator += tl.sum(numerator, axis=0)
                tl.store(
                    output_block_ptr, numerator.to(element_ty), boundary_check=(0,)
                )

            inv_denom = 1.0 / denominator
            for col_idx in range(0, n_cols, COL_SIZE):
                output_block_ptr = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(col_idx,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                exp_out = tl.load(output_block_ptr, boundary_check=(0,)).to(tl.float32)
                tl.store(
                    output_block_ptr,
                    (exp_out * inv_denom).to(element_ty),
                    boundary_check=(0,),
                )


def _spacemit_softmax_lastdim(inp, out):
    n_rows, n_cols = inp.shape
    row_size = 1 if n_rows < 2 else (2 if n_rows < 8 else 4)
    col_size = 64
    grid = lambda meta: (triton.cdiv(n_rows, meta["ROW_SIZE"]),)
    softmax_kernel_spacemit[grid](
        out,
        inp,
        inp.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        ROW_SIZE=row_size,
        COL_SIZE=col_size,
    )


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_SPACEMIT SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype

    inp = self.contiguous()

    n_cols = inp.shape[-1]
    n_rows = inp.numel() // n_cols
    inp_2d = inp.view(n_rows, n_cols)
    out_2d = torch.empty_like(inp_2d, dtype=dtype)
    _spacemit_softmax_lastdim(inp_2d, out_2d)
    return out_2d.view_as(inp)


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_SPACEMIT SOFTMAX_VJP")
    return common_softmax_backward(grad_output, output, dim, input_dtype)
