import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base


class _NansumBenchmark(base.GenericBenchmark):
    """Benchmark for nansum with GBPS metric."""

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, bench_fn_args, latency):
        inp = bench_fn_args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


def _nansum_input_fn(shape, dtype, device):
    num_elements = 1
    for s in shape:
        num_elements *= s

    max_elements = 536870912 if dtype == torch.float64 else 1073741824
    if num_elements >= max_elements:
        return

    x = torch.randn(shape, dtype=dtype, device=device) * 10
    mask = torch.rand(shape, device=device) > 0.7
    x[mask] = float("nan")

    yield (x,)


@pytest.mark.nansum
def test_benchmark_nansum():
    bench = _NansumBenchmark(
        op_name="nansum",
        torch_op=torch.nansum,
        input_fn=_nansum_input_fn,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(flag_gems.nansum)
    bench.run()


@pytest.mark.nansum_dim
def test_benchmark_nansum_dim():
    bench = _NansumBenchmark(
        op_name="nansum_dim",
        torch_op=lambda x: torch.nansum(x, dim=-1),
        input_fn=_nansum_input_fn,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(lambda x: flag_gems.nansum(x, dim=-1))
    bench.run()
