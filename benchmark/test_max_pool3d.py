from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


def max_pool3d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Non-cubic kernel/stride/padding
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": (2, 3, 3),
                "stride": (1, 2, 2),
                "padding": (0, 1, 1),
                "dilation": 1,
                "ceil_mode": False,
            }
        # With dilation (effective kernel = (3-1)*2+1 = 5, need dim+2*pad >= 5)
        if shape[-3] >= 4 and shape[-2] >= 4 and shape[-1] >= 4:
            yield inp, {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 2,
                "ceil_mode": False,
            }
        # With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": True,
        }


class MaxPool3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        # Representative 5-D (N, C, D, H, W) tensors covering typical 3D-CNN
        # feature-map sizes from shallow/large to deep/small.
        shapes_5d = [
            (4, 3, 16, 56, 56),
            (8, 64, 8, 28, 28),
            (16, 128, 4, 14, 14),
            (32, 256, 2, 7, 7),
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.max_pool3d
def test_max_pool3d():
    bench = MaxPool3dBenchmark(
        input_fn=max_pool3d_input_fn,
        op_name="max_pool3d",
        torch_op=torch.max_pool3d,
        gems_op=flag_gems.max_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
