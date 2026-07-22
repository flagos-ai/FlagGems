import argparse
import time

import torch
import torch_npu  # noqa: F401
import triton

import flag_gems
from benchmark.ascendc_baseline import load_nonzero_static
from flag_gems.ops.nonzero_static import (
    _nonzero_static_count_groups_kernel,
    _nonzero_static_count_kernel,
    _nonzero_static_expand_coordinates_kernel,
    _nonzero_static_write_small_counts_kernel,
    _nonzero_static_write_sparse_groups_fallback_kernel,
    _nonzero_static_write_sparse_groups_kernel,
    _nonzero_static_write_strided_kernel,
)
from flag_gems.runtime.backend._ascend.ops.cumsum import cumsum
from flag_gems.runtime.backend._ascend.ops.nonzero_static import (
    ASCEND_SCAN_GROUP_SIZE,
    ASCEND_SMALL_COUNTS_MAX_BLOCKS,
    ASCEND_SPARSE_COUNT_GROUP_BLOCKS,
    ASCEND_SPARSE_GROUP_SELECT_MAX_NNZ,
    _get_block_size,
    _get_sparse_scan_group_size,
    _use_small_linear_output,
    _use_sparse_groups,
)
from flag_gems.runtime.backend._ascend.utils import CORE_NUM

CASES = [
    ((16384,), 0.1, 1024),
    ((262144,), 0.1, 4096),
    ((1048576,), 0.1, 4096),
    ((1024, 4096), 0.001, 1024),
    ((1024, 4096), 0.1, 4096),
    ((1048576,), 0.001, 1024),
    ((128, 4096), 0.01, 4096),
    ((32, 128, 128), 0.01, 4096),
    ((262144,), 0.001, 1024),
    ((32, 1024), 0.1, 1024),
    ((16, 64, 64), 0.1, 1024),
]


def bench(fn, warmup, rep):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / rep


def make_input(shape, ratio, dtype):
    mask = torch.rand(shape, device="npu") < ratio
    x = torch.zeros(shape, dtype=dtype, device="npu")
    x[mask] = 1
    return x


def profile_case(shape, ratio, size, dtype, baseline, warmup, rep):
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    x = make_input(shape, ratio, dtype)
    numel = x.numel()
    ndim = x.dim()
    block_size = _get_block_size(x, size)
    num_blocks = triton.cdiv(numel, block_size)
    program_count = min(num_blocks, CORE_NUM)
    total_out = size * ndim
    shape4 = list(shape) + [1] * (4 - ndim)
    use_sparse = _use_sparse_groups(x, size)
    use_small_linear = _use_small_linear_output(x, size)
    is_bfloat16 = dtype == torch.bfloat16
    kernel_x = x.view(torch.uint16) if is_bfloat16 else x
    use_small = (
        num_blocks <= ASCEND_SMALL_COUNTS_MAX_BLOCKS
        and total_out <= num_blocks * block_size
    )
    out = torch.empty((size, ndim), dtype=torch.int64, device="npu")
    linear_out = (
        torch.empty((size,), dtype=torch.int64, device="npu")
        if use_small_linear
        else out
    )

    if use_sparse:
        num_count_groups = triton.cdiv(num_blocks, ASCEND_SPARSE_COUNT_GROUP_BLOCKS)
        group_tasks = triton.cdiv(num_count_groups, program_count)
        queue_capacity_per_worker = group_tasks * ASCEND_SPARSE_COUNT_GROUP_BLOCKS
        fallback_capacity = program_count * queue_capacity_per_worker
        fallback_counts_offset = num_count_groups
        fallback_ids_offset = fallback_counts_offset + program_count
        fallback_prefix_offset = fallback_ids_offset + fallback_capacity
        workspace = torch.empty(
            fallback_prefix_offset + fallback_capacity,
            dtype=torch.int64,
            device="npu",
        )
        prefix_block_size = triton.next_power_of_2(num_count_groups)

        def count_fn():
            _nonzero_static_count_groups_kernel[(program_count,)](
                kernel_x,
                workspace,
                numel,
                num_blocks,
                num_count_groups,
                IS_COMPLEX=False,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
                COUNT_GROUP_BLOCKS=ASCEND_SPARSE_COUNT_GROUP_BLOCKS,
            )

    else:
        counts = torch.empty(num_blocks, dtype=torch.int64, device="npu")

        def count_fn():
            _nonzero_static_count_kernel[(program_count,)](
                kernel_x,
                counts,
                numel,
                num_blocks,
                IS_COMPLEX=False,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
            )

    count_fn()
    torch.npu.synchronize()
    if not use_sparse:
        prefix = cumsum(counts, dim=0)
        torch.npu.synchronize()

    if use_sparse:
        active_scan_group_size = _get_sparse_scan_group_size(x)

        def write_fn():
            _nonzero_static_write_sparse_groups_kernel[(program_count,)](
                kernel_x,
                workspace,
                out,
                size,
                numel,
                num_blocks,
                num_count_groups,
                ndim,
                shape4[0],
                shape4[1],
                shape4[2],
                shape4[3],
                -1,
                total_out,
                IS_COMPLEX=False,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
                SCAN_GROUP_SIZE=active_scan_group_size,
                SELECT_MAX_NNZ=ASCEND_SPARSE_GROUP_SELECT_MAX_NNZ,
                PREFIX_BLOCK_SIZE=prefix_block_size,
                COUNT_GROUP_BLOCKS=ASCEND_SPARSE_COUNT_GROUP_BLOCKS,
                QUEUE_CAPACITY_PER_WORKER=queue_capacity_per_worker,
                FALLBACK_COUNTS_OFFSET=fallback_counts_offset,
                FALLBACK_IDS_OFFSET=fallback_ids_offset,
                FALLBACK_PREFIX_OFFSET=fallback_prefix_offset,
            )
            _nonzero_static_write_sparse_groups_fallback_kernel[(program_count,)](
                kernel_x,
                workspace,
                out,
                size,
                numel,
                num_blocks,
                ndim,
                shape4[0],
                shape4[1],
                shape4[2],
                shape4[3],
                IS_COMPLEX=False,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
                SCAN_GROUP_SIZE=active_scan_group_size,
                SELECT_MAX_NNZ=ASCEND_SPARSE_GROUP_SELECT_MAX_NNZ,
                QUEUE_CAPACITY_PER_WORKER=queue_capacity_per_worker,
                FALLBACK_COUNTS_OFFSET=fallback_counts_offset,
                FALLBACK_IDS_OFFSET=fallback_ids_offset,
                FALLBACK_PREFIX_OFFSET=fallback_prefix_offset,
            )

        path = "sparse_group_prefix"
    elif use_small:
        active_scan_group_size = ASCEND_SCAN_GROUP_SIZE
        prefix_block_size = triton.next_power_of_2(num_blocks)
        prefix_step_size = triton.next_power_of_2(program_count)

        def write_fn():
            _nonzero_static_write_small_counts_kernel[(program_count,)](
                kernel_x,
                counts,
                linear_out,
                out,
                size,
                numel,
                num_blocks,
                ndim,
                shape4[0],
                shape4[1],
                shape4[2],
                shape4[3],
                -1,
                total_out,
                IS_COMPLEX=False,
                IS_BFLOAT16=is_bfloat16,
                BLOCK_SIZE=block_size,
                PREFIX_BLOCK_SIZE=prefix_block_size,
                PREFIX_STEP_SIZE=prefix_step_size,
                SCAN_GROUP_SIZE=ASCEND_SCAN_GROUP_SIZE,
                STORE_LINEAR=use_small_linear,
            )
            if use_small_linear:
                expand_block_size = min(1024, triton.next_power_of_2(size))
                _nonzero_static_expand_coordinates_kernel[
                    (triton.cdiv(size, expand_block_size), ndim)
                ](
                    linear_out,
                    counts,
                    out,
                    size,
                    num_blocks,
                    ndim,
                    shape4[1],
                    shape4[2],
                    shape4[3],
                    -1,
                    PREFIX_BLOCK_SIZE=prefix_block_size,
                    BLOCK_SIZE=expand_block_size,
                )

        path = "small_linear" if use_small_linear else "small"
    else:
        active_scan_group_size = ASCEND_SCAN_GROUP_SIZE

        def write_fn():
            _nonzero_static_write_strided_kernel[(program_count,)](
                x,
                prefix,
                counts,
                out,
                size,
                numel,
                num_blocks,
                ndim,
                shape4[0],
                shape4[1],
                shape4[2],
                shape4[3],
                -1,
                total_out,
                IS_COMPLEX=False,
                BLOCK_SIZE=block_size,
                SCAN_GROUP_SIZE=ASCEND_SCAN_GROUP_SIZE,
            )

        path = "prefix"

    count_ms = bench(count_fn, warmup, rep)
    prefix_ms = 0.0 if use_sparse else bench(lambda: cumsum(counts, dim=0), warmup, rep)
    write_ms = bench(write_fn, warmup, rep)
    fallback_blocks = (
        int(
            workspace[fallback_counts_offset : fallback_counts_offset + program_count]
            .sum()
            .item()
        )
        if use_sparse
        else 0
    )
    total_ms = bench(
        lambda: flag_gems.nonzero_static(x, size=size, fill_value=-1),
        warmup,
        rep,
    )
    baseline_ms = bench(lambda: baseline(x, size, -1), warmup, rep)

    print(
        f"dtype={dtype} shape={shape} ratio={ratio} size={size} "
        f"block={block_size} scan={active_scan_group_size} "
        f"select={ASCEND_SPARSE_GROUP_SELECT_MAX_NNZ} "
        f"blocks={num_blocks} programs={program_count} "
        f"path={path} fallback_blocks={fallback_blocks} "
        f"count={count_ms:.6f}ms prefix={prefix_ms:.6f}ms "
        f"write={write_ms:.6f}ms total={total_ms:.6f}ms "
        f"baseline={baseline_ms:.6f}ms speedup={baseline_ms / total_ms:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--dtype",
        choices=("all", "float16", "bfloat16"),
        default="all",
    )
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    baseline = load_nonzero_static()
    dtypes = {
        "all": (torch.float16, torch.bfloat16),
        "float16": (torch.float16,),
        "bfloat16": (torch.bfloat16,),
    }[args.dtype]
    cases = CASES if args.case_index is None else (CASES[args.case_index],)
    for dtype in dtypes:
        for shape, ratio, size in cases:
            profile_case(
                shape,
                ratio,
                size,
                dtype,
                baseline,
                args.warmup,
                args.rep,
            )


if __name__ == "__main__":
    main()
