import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base

# (shape, n) cases grouped by performance dimension.
#   matrix size sweep at fixed n=5 (3 matmuls: 2 squarings + 1 accumulate)
#   exponent sweep at 32×32 (matmul count grows as O(log n))
#   batched matrices at n=5
BENCH_CASES = [
    # exponent sweep (32×32)
    *[((32, 32), n) for n in (0, 2, 3, 8, 16, 32, 64)],
    # batched matrices
    *[(s, n) for s in ((8, 2, 2), (16, 64, 64), (2, 1024, 1024)) for n in (2, 8, 31)],
    *[
        (s, n)
        for s in ((2, 2), (8, 8), (64, 64), (256, 256), (1024, 1024))
        for n in (2, 8, 31)
    ],
]

if flag_gems.vendor_name != "ascend":
    BENCH_CASES += [
        *[
            (s, n)
            for s in ((2, 2), (8, 8), (64, 64), (256, 256), (1024, 1024))
            for n in (-2, -8, -31)
        ],
    ]

if flag_gems.runtime.device.support_fp64:
    DTYPES_ALL = [torch.float32, torch.float64]
else:
    DTYPES_ALL = [torch.float32]


def matrix_power_input_fn(shape, dtype, device):
    """Yield (input, n) for every (shape, n) case matching this shape."""
    for s, n in BENCH_CASES:
        if s == shape:
            yield torch.randn(shape, dtype=dtype, device=device), n


class MatrixPowerBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Benchmark exactly the (shape, n) table; ignore the yaml config.
        self.shapes = list(dict.fromkeys(s for s, _ in BENCH_CASES))


@pytest.mark.linalg_matrix_power
def test_linalg_matrix_power():
    bench = MatrixPowerBenchmark(
        op_name="linalg_matrix_power",
        torch_op=torch.ops.aten.linalg_matrix_power,
        input_fn=matrix_power_input_fn,
        dtypes=DTYPES_ALL,
    )
    # Pre-warm every (shape, n) kernel variant before bench.run() so the
    # benchmark's per-op iteration-count probe (which measures the first few
    # calls) sees the steady-state latency instead of a one-time Triton kernel
    # compile.  Without this, the probe reports a multi-ms first-call latency
    # and the benchmark falls back to ~3 repetitions, which is too few to
    # measure a fast operator accurately.
    warm = flag_gems.device
    for shape, n in BENCH_CASES:
        flag_gems.linalg_matrix_power(torch.randn(shape, device=warm), n)
    torch_device_fn.synchronize()
    bench.run()
