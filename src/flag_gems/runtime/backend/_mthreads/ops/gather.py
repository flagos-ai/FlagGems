import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.gather import gather as default_gather
from flag_gems.ops.gather import gather_backward as default_gather_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.heuristics(runtime.get_heuristic_config("gather"))
@triton.jit
def _gather_lastdim_kernel(
    inp_ptr,
    index_ptr,
    out_ptr,
    stride_inp_row,
    stride_index_row,
    stride_out_row,
    dim_stride,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    rows = rows.to(tl.int64)
    cols = cols.to(tl.int64)
    mask = (rows < M) & (cols < N)

    row_inp = rows * stride_inp_row
    row_idx = rows * stride_index_row
    row_out = rows * stride_out_row

    idx = tl.load(index_ptr + row_idx + cols, mask=mask, other=0).to(tl.int64)
    gather_ptr = inp_ptr + row_inp + idx * dim_stride
    values = tl.load(gather_ptr, mask=mask, other=0)
    tl.store(out_ptr + row_out + cols, values, mask=mask)


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def _use_triton_kernel(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    out: Optional[torch.Tensor],
) -> bool:
    if inp.device.type != "musa" or index.device != inp.device:
        return False
    if inp.dtype not in _SUPPORTED_DTYPES or index.dtype != torch.long:
        return False

    dim = _normalize_dim(dim, inp.ndim)
    if dim != inp.ndim - 1:
        return False

    if not inp.is_contiguous() or not index.is_contiguous():
        return False
    if out is not None:
        if (
            out.device != inp.device
            or out.dtype != inp.dtype
            or not out.is_contiguous()
        ):
            return False

    if index.shape[:-1] != inp.shape[:-1]:
        return False

    return True


def _launch_triton(
    inp: torch.Tensor,
    index: torch.Tensor,
    out: torch.Tensor,
    dim_stride: int,
) -> torch.Tensor:
    inp_2d = inp.view(-1, inp.shape[-1])
    index_2d = index.view(-1, index.shape[-1])
    out_2d = out.view(-1, index.shape[-1])

    M, N = index_2d.shape
    stride_inp_row = inp_2d.stride(0)
    stride_index_row = index_2d.stride(0)
    stride_out_row = out_2d.stride(0)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    with torch_device_fn.device(out.device):
        _gather_lastdim_kernel[grid](
            inp_2d,
            index_2d,
            out_2d,
            stride_inp_row,
            stride_index_row,
            stride_out_row,
            dim_stride,
            M,
            N,
        )
    return out


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_MTHREADS GATHER")
    if not _use_triton_kernel(inp, dim, index, out):
        return default_gather(inp, dim, index, out, sparse_grad)

    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    dim_stride = inp.stride(_normalize_dim(dim, inp.ndim))
    return _launch_triton(inp, index, out, dim_stride)


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_MTHREADS GATHER BACKWARD")
    return default_gather_backward(grad, self, dim, index, sparse_grad)
