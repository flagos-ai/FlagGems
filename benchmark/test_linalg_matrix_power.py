import pytest
import torch

import flag_gems

from . import base

BENCH_CASES = [
    *[((32, 32), n) for n in (0, 2, 3, 8, 16, 32, 64)],
    *[
        (s, n)
        for s in ((8, 4, 4), (16, 64, 64), (2, 1024, 1024))
        for n in (2, 8, 31, 32)
    ],
    *[(s, n) for s in ((8, 8), (256, 256), (1024, 1024)) for n in (2, 8, 31, 32)],
]

if flag_gems.vendor_name != "ascend":
    BENCH_CASES += [
        *[(s, n) for s in ((8, 8), (256, 256), (1024, 1024)) for n in (-2, -8, -31)],
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
        self.shapes = list(dict.fromkeys(s for s, _ in BENCH_CASES))


@pytest.mark.linalg_matrix_power
def test_linalg_matrix_power():
    bench = MatrixPowerBenchmark(
        op_name="linalg_matrix_power",
        torch_op=torch.ops.aten.linalg_matrix_power,
        input_fn=matrix_power_input_fn,
        dtypes=DTYPES_ALL,
    )
    bench.run()
