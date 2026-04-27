from typing import Generator

import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


class AvgPool2dBenchmark(utils.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_4d = [
            (4, 3, 224, 224),  # Typical input image size
            (16, 64, 56, 56),  # Early ResNet layer output
            (32, 128, 28, 28),  # Mid ResNet layer output
            (64, 256, 14, 14),  # Later ResNet layer output
            (128, 512, 7, 7),  # Final ResNet layer output
        ]

        for shape in shapes_4d:
            yield from self.input_fn(shape, cur_dtype, self.device)


def avg_pool2d_input_fn(shape, dtype, device):
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

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
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
        if shape[-2] >= 2 and shape[-1] >= 2:
            yield inp, {
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": 3,
            }


@pytest.mark.avg_pool2d
def test_avg_pool2d():
    bench = AvgPool2dBenchmark(
        input_fn=avg_pool2d_input_fn,
        op_name="avg_pool2d",
        torch_op=torch.ops.aten.avg_pool2d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.skip(reason="Test case fails due to missing parameter self.")
@pytest.mark.avg_pool2d_backward
def test_avg_pool2d_backward():
    if flag_gems.vendor_name == "mthreads":
        dtypes = [torch.float32]
    else:
        dtypes = (attr_utils.FLOAT_DTYPES,)

    bench = AvgPool2dBenchmark(
        input_fn=avg_pool2d_input_fn,
        op_name="avg_pool2d_backward",
        torch_op=torch.ops.aten.avg_pool2d_backward,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()
