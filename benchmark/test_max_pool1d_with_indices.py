from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class MaxPool1dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes_3d = [
            (16, 64, 1024),
            (32, 128, 512),
            (64, 256, 256),
            (8, 32, 4096),
            (128, 512, 128),
        ]

        for shape in shapes_3d:
            yield from self.input_fn(shape, dtype, self.device)


def max_pool1d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)

    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # With dilation
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


@pytest.mark.max_pool1d_with_indices
def test_max_pool1d_with_indices():
    bench = MaxPool1dBenchmark(
        op_name="max_pool1d_with_indices",
        input_fn=max_pool1d_input_fn,
        torch_op=torch.nn.functional.max_pool1d_with_indices,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.max_pool1d_with_indices)

    bench.run()
