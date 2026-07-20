import argparse
import time

import torch
import torch_npu  # noqa: F401
import triton

import flag_gems
from benchmark.ascendc_baseline import load_nonzero_static
from flag_gems.ops.nonzero_static import (
    _nonzero_static_count_kernel,
    _nonzero_static_write_small_counts_kernel,
    _nonzero_static_write_strided_kernel,
)
from flag_gems.runtime.backend._ascend.ops.cumsum import cumsum
from flag_gems.runtime.backend._ascend.ops.nonzero_static import (
    ASCEND_BLOCK_SIZE,
    ASCEND_SMALL_COUNTS_MAX_BLOCKS,
)
from flag_gems.runtime.backend._ascend.utils import CORE_NUM

CASES = [
    ((16384,), 0.1, 1024),
    ((262144,), 0.1, 4096),
    ((1048576,), 0.1, 4096),
    ((1024, 4096), 0.001, 1024),
    ((1024, 4096), 0.1, 4096),
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
    x = make_input(shape, ratio, dtype)
    numel = x.numel()
    ndim = x.dim()
    block_size = ASCEND_BLOCK_SIZE
    num_blocks = triton.cdiv(numel, block_size)
    program_count = min(num_blocks, CORE_NUM)
    total_out = size * ndim
    shape4 = list(shape) + [1] * (4 - ndim)
    counts = torch.empty(num_blocks, dtype=torch.int64, device="npu")
    out = torch.empty((size, ndim), dtype=torch.int64, device="npu")

    def count_fn():
        _nonzero_static_count_kernel[(program_count,)](
            x,
            counts,
            numel,
            num_blocks,
            IS_COMPLEX=False,
            BLOCK_SIZE=block_size,
        )

    count_fn()
    torch.npu.synchronize()
    prefix = cumsum(counts, dim=0)
    torch.npu.synchronize()

    use_small = (
        num_blocks <= ASCEND_SMALL_COUNTS_MAX_BLOCKS
        and total_out <= num_blocks * block_size
    )

    if use_small:
        prefix_block_size = triton.next_power_of_2(num_blocks)

        def write_fn():
            _nonzero_static_write_small_counts_kernel[(num_blocks,)](
                x,
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
                PREFIX_BLOCK_SIZE=prefix_block_size,
            )

        path = "small"
    else:

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
            )

        path = "prefix"

    count_ms = bench(count_fn, warmup, rep)
    prefix_ms = bench(lambda: cumsum(counts, dim=0), warmup, rep)
    write_ms = bench(write_fn, warmup, rep)
    total_ms = bench(
        lambda: flag_gems.nonzero_static(x, size=size, fill_value=-1),
        warmup,
        rep,
    )
    baseline_ms = bench(lambda: baseline(x, size, -1), warmup, rep)

    print(
        f"dtype={dtype} shape={shape} ratio={ratio} size={size} "
        f"block={block_size} blocks={num_blocks} programs={program_count} "
        f"path={path} count={count_ms:.6f}ms prefix={prefix_ms:.6f}ms "
        f"write={write_ms:.6f}ms total={total_ms:.6f}ms "
        f"baseline={baseline_ms:.6f}ms speedup={baseline_ms / total_ms:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()
    baseline = load_nonzero_static()
    for dtype in (torch.float16, torch.bfloat16):
        for shape, ratio, size in CASES:
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
