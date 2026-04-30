import pytest
import torch

from flag_gems.ops.median import median_dim

from . import base, consts


def _median_values(inp, dim, keepdim=False):
    return torch.median(inp, dim, keepdim).values


def _median_values_gems(inp, dim, keepdim=False):
    return median_dim(inp, dim, keepdim).values


@pytest.mark.median_dim
def test_perf_median_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="median.dim",
        torch_op=_median_values,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(_median_values_gems)
    bench.run()
