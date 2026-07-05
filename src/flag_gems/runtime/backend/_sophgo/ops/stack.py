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
def stack4_dim0_2d_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    out_ptr,
    M,
    N,
    x0_stride_m,
    x0_stride_n,
    x1_stride_m,
    x1_stride_n,
    x2_stride_m,
    x2_stride_n,
    x3_stride_m,
    x3_stride_n,
    out_stride_t,
    out_stride_m,
    out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    x0 = tl.load(x0_ptr + rows * x0_stride_m + cols * x0_stride_n, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + rows * x1_stride_m + cols * x1_stride_n, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + rows * x2_stride_m + cols * x2_stride_n, mask=mask, other=0.0)
    x3 = tl.load(x3_ptr + rows * x3_stride_m + cols * x3_stride_n, mask=mask, other=0.0)
    tl.store(out_ptr + rows * out_stride_m + cols * out_stride_n, x0, mask=mask)
    tl.store(out_ptr + out_stride_t + rows * out_stride_m + cols * out_stride_n, x1, mask=mask)
    tl.store(out_ptr + 2 * out_stride_t + rows * out_stride_m + cols * out_stride_n, x2, mask=mask)
    tl.store(out_ptr + 3 * out_stride_t + rows * out_stride_m + cols * out_stride_n, x3, mask=mask)


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logging.debug("GEMS STACK")

    def _generic_stack():
        from flag_gems.ops.stack import stack as generic_stack

        return generic_stack(tensors, dim)

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i+1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    if (
        dim == 0
        and len(tensors) == 4
        and tensors[0].ndim == 2
        and all(t.is_contiguous() for t in tensors)
    ):
        out = torch.empty(
            (4, tensors[0].shape[0], tensors[0].shape[1]),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        M, N = tensors[0].shape
        grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
        stack4_dim0_2d_kernel[grid](
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3],
            out,
            M,
            N,
            tensors[0].stride(0),
            tensors[0].stride(1),
            tensors[1].stride(0),
            tensors[1].stride(1),
            tensors[2].stride(0),
            tensors[2].stride(1),
            tensors[3].stride(0),
            tensors[3].stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_M=16,
            BLOCK_N=64,
        )
        return out

    # The migrated historical optimization is only validated for the 4x2D dim0
    # hotspot above. The legacy copy-based fallback regresses or fails on the
    # broader pytest surface, so everything else should use the new repository's
    # generic implementation.
    return _generic_stack()
