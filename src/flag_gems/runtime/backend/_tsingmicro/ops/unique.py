import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

TOTAL_CORE_NUM = torch_device_fn.get_device_properties().multi_processor_count


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2**k}, num_stages=s, num_warps=1)
        for k in range(11, 17, 1)
        for s in [1, 3]
    ],
    key=[
        "tile_size",
    ],
)
@triton.jit
def get_ne_kernel(
    sorted_data_ptr: tl.tensor,
    sorted_data_2: tl.tensor,
    ne_out_ptr: tl.tensor,
    tile_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    split_n = (tile_size + num_jobs - 1) // num_jobs
    start_offset = pid * split_n
    i0 = tl.arange(0, BLOCK_SIZE)

    for i in range(0, split_n, BLOCK_SIZE):
        offset = start_offset + i + i0
        mask = offset < tile_size
        a = tl.load(sorted_data_ptr + offset, mask=mask)
        b = tl.load(sorted_data_2 + offset, mask=mask)
        # ne
        ne_result = (offset > 0) * (a != b)
        tl.store(ne_out_ptr + offset, ne_result, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": k}, num_stages=s, num_warps=1)
        for k in [32, 256, 1024, 2048, 4096]
        for s in [1, 3]
    ],
    key=[
        "tile_size",
    ],
)
@triton.jit
def get_unique_out_kernel(
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    ne_result_ptr: tl.tensor,
    pre_sum_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
    tile_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)

    split_n = (tile_size + num_jobs - 1) // num_jobs
    start_offset = pid * split_n
    i0 = tl.arange(0, BLOCK_SIZE)

    for i in range(0, split_n, BLOCK_SIZE):
        offset = start_offset + i + i0
        mask = offset < tile_size
        sorted_data = tl.load(sorted_data_ptr + offset, mask=mask)
        pre_sum_data = tl.load(pre_sum_ptr + offset, mask=mask)

        # data_out: scatter_(to=pre_sum_data, sorted_data)
        tl.store(data_out_ptr + pre_sum_data, sorted_data, mask=mask)

        # inverse_indices: scatter_(to=sorted_indices, pre_sum_data)
        if return_inverse:
            sorted_indices = tl.load(sorted_indices_ptr + offset, mask=mask)
            tl.store(inverse_indices_ptr + sorted_indices, pre_sum_data, mask=mask)

        # idx: mark positions of unique values in idx_ptr
        if return_counts:
            ne_result = tl.load(ne_result_ptr + offset, mask=mask)
            idx_mask = ((offset == 0) | ne_result.to(tl.int1)) & mask
            tl.store(idx_ptr + pre_sum_data, offset, mask=idx_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2**k}, num_stages=s, num_warps=1)
        for k in range(7, 14, 1)
        for s in [1, 3]
    ],
    key=[
        "tile_size",
    ],
)
@triton.jit
def get_output_counts_kernel(
    idx_ptr: tl.tensor,
    idx_next_ptr: tl.tensor,
    counts_ptr: tl.tensor,  # out
    tile_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    split_n = (tile_size + num_jobs - 1) // num_jobs
    start_offset = pid * split_n

    i0 = tl.arange(0, BLOCK_SIZE)

    for i in range(0, split_n, BLOCK_SIZE):
        offset = start_offset + i + i0
        mask = offset < tile_size
        # load idx
        idx = tl.load(idx_ptr + offset, mask=mask)
        # load idx_next
        idx_next = tl.load(idx_next_ptr + offset, mask=mask)
        # diff
        counts = idx_next - idx
        # store counts
        tl.store(counts_ptr + offset, counts, mask=mask)


def sorted_unique_flat(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    num_tasks = sorted_data.numel()
    grid = lambda meta: (
        min(triton.cdiv(num_tasks, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )

    # allocate tensor
    ne_out = torch.empty_like(sorted_data, dtype=torch.bool)
    data_out = torch.empty_like(sorted_data)
    if return_inverse:
        inverse_indices = torch.empty_like(sorted_data, dtype=torch.int64)
    else:
        inverse_indices = None
    if return_counts:
        idx = torch.empty_like(sorted_data, dtype=torch.int64)
    else:
        idx = None
    sorted_data_2 = torch.empty_like(sorted_data)
    sorted_data_2[1:] = sorted_data[:-1]

    # launch kernel
    with torch_device_fn.device(sorted_data.device.index):
        get_ne_kernel[grid](
            sorted_data,
            sorted_data_2,
            ne_out,
            tile_size=num_tasks,
        )
        pre_sum = ne_out.cumsum(axis=0)
        get_unique_out_kernel[grid](
            sorted_data,
            sorted_indices,
            ne_out,
            pre_sum,
            idx,
            data_out,
            inverse_indices,
            return_inverse,
            return_counts,
            tile_size=num_tasks,
        )

    out_size = pre_sum[-1].item() + 1
    counts = None
    if return_counts:
        idx = idx[:out_size]
        sorted_data_size = len(sorted_data)
        idx_next = torch.roll(idx, -1)
        idx_next[-1] = sorted_data_size
        counts = torch.zeros_like(idx)
        with torch_device_fn.device(sorted_data.device.index):
            get_output_counts_kernel[grid](
                idx,
                idx_next,
                counts,  # out
                tile_size=out_size,
            )
    return data_out[:out_size], inverse_indices, counts


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    sorted_data, sorted_indices = torch.sort(in0.ravel(), stable=False)
    data_out, inverse_indices, counts = sorted_unique_flat(
        sorted_data, sorted_indices, return_inverse, return_counts
    )
    return (
        data_out,
        inverse_indices if inverse_indices is None else inverse_indices.view_as(in0),
        counts,
    )
