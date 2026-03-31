import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from ..utils import CORE_NUM

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("nonzero"), key=["n_elements"])
@triton.jit
def nonzero_kernel_1d(
    inp,
    prefix_sum,
    out,
    n_elements,
    dim0_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n_workers = tle.num_programs(0)
    tasks = tl.cdiv(n_elements, BLOCK_SIZE)
    tasks_per_worker = tl.cdiv(tasks, n_workers)

    for task_idx in range(tasks_per_worker):
        task_id = pid + task_idx * n_workers
        block_start = task_id * BLOCK_SIZE
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements

        inp_vals = tl.load(inp + offset, mask=mask, care_padding=False).to(tl.int1)
        nonzero_mask = mask and inp_vals
        out_offset = tl.load(
            prefix_sum + offset, mask=nonzero_mask, care_padding=False
        ) - 1

        tl.store(out + out_offset, offset, mask=nonzero_mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("nonzero"), key=["n_elements"])
@triton.jit
def nonzero_kernel_2d(
    inp,
    prefix_sum,
    out,
    n_elements,
    dim0_size,
    dim1_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n_workers = tle.num_programs(0)
    tasks = tl.cdiv(n_elements, BLOCK_SIZE)
    tasks_per_worker = tl.cdiv(tasks, n_workers)
    ndim: tl.constexpr = 2

    for task_idx in range(tasks_per_worker):
        task_id = pid + task_idx * n_workers
        block_start = task_id * BLOCK_SIZE
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements

        inp_vals = tl.load(inp + offset, mask=mask, care_padding=False).to(tl.int1)
        nonzero_mask = mask and inp_vals
        out_offset = tl.load(
            prefix_sum + offset, mask=nonzero_mask, care_padding=False
        ) - 1

        idx_flat = offset
        dim1_val = idx_flat % dim1_size
        idx_flat //= dim1_size
        dim0_val = idx_flat % dim0_size

        tl.store(out + out_offset * ndim + 1, dim1_val, mask=nonzero_mask)
        tl.store(out + out_offset * ndim, dim0_val, mask=nonzero_mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("nonzero"), key=["n_elements"])
@triton.jit
def nonzero_kernel_3d(
    inp,
    prefix_sum,
    out,
    n_elements,
    dim0_size,
    dim1_size,
    dim2_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n_workers = tle.num_programs(0)
    tasks = tl.cdiv(n_elements, BLOCK_SIZE)
    tasks_per_worker = tl.cdiv(tasks, n_workers)
    ndim: tl.constexpr = 3

    for task_idx in range(tasks_per_worker):
        task_id = pid + task_idx * n_workers
        block_start = task_id * BLOCK_SIZE
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements

        inp_vals = tl.load(inp + offset, mask=mask, care_padding=False).to(tl.int1)
        nonzero_mask = mask and inp_vals
        out_offset = tl.load(
            prefix_sum + offset, mask=nonzero_mask, care_padding=False
        ) - 1

        idx_flat = offset
        dim2_val = idx_flat % dim2_size
        idx_flat //= dim2_size
        dim1_val = idx_flat % dim1_size
        idx_flat //= dim1_size
        dim0_val = idx_flat % dim0_size

        tl.store(out + out_offset * ndim + 2, dim2_val, mask=nonzero_mask)
        tl.store(out + out_offset * ndim + 1, dim1_val, mask=nonzero_mask)
        tl.store(out + out_offset * ndim, dim0_val, mask=nonzero_mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("nonzero"), key=["n_elements"])
@triton.jit
def nonzero_kernel_nd(
    inp,
    prefix_sum,
    out,
    n_elements,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n_workers = tle.num_programs(0)
    tasks = tl.cdiv(n_elements, BLOCK_SIZE)
    tasks_per_worker = tl.cdiv(tasks, n_workers)

    for task_idx in range(tasks_per_worker):
        task_id = pid + task_idx * n_workers
        block_start = task_id * BLOCK_SIZE
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements

        inp_vals = tl.load(inp + offset, mask=mask, care_padding=False).to(tl.int1)
        nonzero_mask = mask and inp_vals
        out_offset = tl.load(
            prefix_sum + offset, mask=nonzero_mask, care_padding=False
        ) - 1

        idx_flat = offset
        for dim in range(ndim - 1, -1, -1):
            dim_size = tl.load(shape + dim)
            remainder = idx_flat % dim_size
            idx_flat //= dim_size
            tl.store(out + out_offset * ndim + dim, remainder, mask=nonzero_mask)


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS_ASCEND NONZERO")

    inp_ndim = inp.ndim
    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    if inp_view.dtype == torch.bool:
        inp_bool = inp_view
    else:
        inp_bool = inp_view != 0

    prefix_sum = inp_bool.cumsum(axis=0, dtype=torch.int32)

    # Allocate full-size output, launch kernel, then trim.
    # This moves the .item() sync after the kernel launch so the kernel
    # can overlap with the device-to-host transfer.
    out = torch.empty(n_elements, inp_ndim, dtype=torch.int64, device=inp.device)

    grid = lambda meta: (min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), CORE_NUM),)

    with torch_device_fn.device(inp.device):
        if inp_ndim == 1:
            nonzero_kernel_1d[grid](
                inp_bool, prefix_sum, out, n_elements, inp.shape[0]
            )
        elif inp_ndim == 2:
            nonzero_kernel_2d[grid](
                inp_bool, prefix_sum, out, n_elements,
                inp.shape[0], inp.shape[1]
            )
        elif inp_ndim == 3:
            nonzero_kernel_3d[grid](
                inp_bool, prefix_sum, out, n_elements,
                inp.shape[0], inp.shape[1], inp.shape[2]
            )
        else:
            shape = torch.tensor(
                tuple(inp.shape), dtype=torch.int32, device=inp.device
            )
            nonzero_kernel_nd[grid](
                inp_bool, prefix_sum, out, n_elements, shape, inp_ndim
            )

    num_nonzeros = prefix_sum[n_elements - 1].item()
    out = out[0:num_nonzeros]

    if as_tuple:
        return torch.unbind(out, dim=0)
    else:
        return out
