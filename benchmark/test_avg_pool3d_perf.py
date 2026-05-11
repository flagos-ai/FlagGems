from typing import Generator

import pytest
import torch

from . import base, consts, utils


def avg_pool3d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    # Common case
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # With count_include_pad=False
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": False,
            "divisor_override": None,
        }
        # With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": True,
            "count_include_pad": True,
            "divisor_override": None,
        }
        # With divisor_override
        if shape[-3] >= 2 and shape[-2] >= 2 and shape[-1] >= 2:
            yield inp, {
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": 3,
            }


class AvgPool3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes_5d = [
            (4, 3, 16, 56, 56),
            (8, 64, 8, 28, 28),
            (16, 128, 4, 14, 14),
            (32, 256, 4, 7, 7),
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.avg_pool3d
def test_perf_avg_pool3d():
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.ops.aten.avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.avg_pool3d_backward
def test_perf_avg_pool3d_backward():
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.ops.aten.avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
