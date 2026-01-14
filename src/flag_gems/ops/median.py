import logging
import os
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def median_value_kernel(
    sorted_inp,
    out_value,
    stride_sm,
    stride_sn,
    M,
    N,
    median_pos,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = pid < M

    base = sorted_inp + pid * stride_sm + median_pos * stride_sn
    val = tl.load(base, mask=mask)

    tl.store(out_value + pid, val, mask=mask)


def median_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    dim_arg = dim
    dim = dim % inp.ndim
    shape = list(inp.shape)
    N = shape[dim]
    if N == 0:
        raise RuntimeError("median: dimension is empty")

    median_pos = (N - 1) // 2

    inp_c = dim_compress(inp, dim)
    M = inp_c.numel() // N
    inp_2d = inp_c.reshape(M, N)

    sorted_vals, sorted_idx = torch.sort(inp_2d, dim=-1, stable=True)

    out_value = torch.empty(
        (M,),
        dtype=inp.dtype,
        device=inp.device,
    )

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        median_value_kernel[grid](
            sorted_vals,
            out_value,
            sorted_vals.stride(0),
            sorted_vals.stride(1),
            M,
            N,
            median_pos,
        )

    out_index = torch.empty(
        (M,),
        dtype=torch.int64,
        device=inp.device,
    )

    has_nan = None
    if inp_2d.is_floating_point() or inp_2d.is_complex():
        nan_mask = torch.isnan(inp_2d)
        has_nan = nan_mask.any(dim=-1)
        first_nan_index = nan_mask.int().argmax(dim=-1)

        if has_nan.any():
            nan_value = torch.tensor(
                float("nan"),
                dtype=out_value.dtype,
                device=out_value.device,
            )
            out_value = torch.where(has_nan, nan_value, out_value)
            out_index[has_nan] = first_nan_index[has_nan]

    if has_nan is None:
        no_nan_mask = torch.ones((M,), dtype=torch.bool, device=inp.device)
    else:
        no_nan_mask = ~has_nan

    if no_nan_mask.any():
        rows = torch.nonzero(no_nan_mask, as_tuple=False).squeeze(1)

        vals = out_value[rows]
        src = inp_2d[rows]
        eq_mask = src == vals.unsqueeze(1)

        assert eq_mask.any(dim=-1).all()

        first_eq_index = eq_mask.int().argmax(dim=-1)
        out_index[rows] = first_eq_index
        
    shape[dim] = 1
    out_value = out_value.reshape(shape)
    out_index = out_index.reshape(shape)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    Median_out = namedtuple("median", ["values", "indices"])
    return Median_out(values=out_value, indices=out_index)
