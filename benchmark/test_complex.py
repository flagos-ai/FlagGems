import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base


class ComplexBenchmark(base.GenericBenchmark):
    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, bench_fn_args, latency):
        real = bench_fn_args[0]
        io_amount = shape_utils.size_in_bytes(real) * 4
        return io_amount * 1e-9 / (latency * 1e-3)


def _complex_input_fn(shape, dtype, device):
    num_elements = 1
    for s in shape:
        num_elements *= s

    max_elements = 536870912 if dtype == torch.float64 else 1073741824

    if num_elements >= max_elements:
        return

    real = torch.randn(shape, dtype=dtype, device=device)
    imag = torch.randn(shape, dtype=dtype, device=device)
    yield real, imag


@pytest.mark.complex
def test_benchmark_complex():
    bench = ComplexBenchmark(
        op_name="complex",
        torch_op=torch.complex,
        input_fn=_complex_input_fn,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(flag_gems.complex)
    bench.run()
