import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer
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
def hstack_copy_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    out_col_offset,
    x_stride_m,
    x_stride_n,
    out_stride_m,
    out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    vals = tl.load(
        x_ptr + rows * x_stride_m + cols * x_stride_n,
        mask=mask,
        other=0,
    )
    tl.store(
        out_ptr + rows * out_stride_m + (cols + out_col_offset) * out_stride_n,
        vals,
        mask=mask,
    )


def hstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> torch.Tensor:
    logging.debug("GEMS HSTACK")

    if len(tensors) == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    if tensors[0].ndim == 0:
        tensors[0] = tensors[0].view(1)
    inp0_shape = tensors[0].shape
    out_shape = list(inp0_shape)
    inp_shapes = [inp0_shape]

    if len(inp0_shape) == 1:
        dim = 0
    else:
        dim = 1

    for tensor_num, tensor in enumerate(tensors[1:]):
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        if tensor.ndim != tensors[0].ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {tensors[0].ndim} and {tensor.ndim}"
            )

        inp_shape = tensor.shape
        inp_shapes.append(inp_shape)

        for i in range(len(inp_shape)):
            if i != dim and inp_shape[i] != inp0_shape[i]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. \
                        Expected size {inp0_shape[i]} but got size {inp_shape[i]} \
                        for tensor number {tensor_num + 1} in the list."
                )

    out_shape[dim] = sum(s[dim] for s in inp_shapes)

    if (
        dim == 1
        and len(tensors) == 3
        and tensors[0].ndim == 2
        and all(t.is_contiguous() for t in tensors)
    ):
        out = torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
        M = tensors[0].shape[0]
        N0 = tensors[0].shape[1]
        N1 = tensors[1].shape[1]
        N2 = tensors[2].shape[1]

        def launch(src, cols, offset):
            grid = (triton.cdiv(M, 32), triton.cdiv(cols, 128))
            hstack_copy_2d_kernel[grid](
                src,
                out,
                M,
                cols,
                offset,
                src.stride(0),
                src.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=32,
                BLOCK_N=128,
            )

        launch(tensors[0], N0, 0)
        launch(tensors[1], N1, N0)
        launch(tensors[2], N2, N0 + N1)
        return out

    out0 = torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    out0_strides = out0.stride()
    out0_offsets = list(
        itertools.accumulate(
            [s[dim] * out0_strides[dim] for s in inp_shapes[:-1]], initial=0
        )
    )

    for a, out0_offset in zip(tensors, out0_offsets):
        in_view = StridedBuffer(a, a.shape, a.stride())
        out_view = StridedBuffer(out0, a.shape, out0.stride(), offset=out0_offset)
        copy_func.instantiate(a.ndim)(in_view, out0=out_view)

    return out0
