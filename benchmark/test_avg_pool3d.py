from typing import Generator

import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


class AvgPool3dBenchmark(utils.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for config in AVGPOOL3D_BENCH_CONFIGS:
            yield from self.input_fn(config, cur_dtype, self.device)


AVGPOOL3D_BENCH_CONFIGS = [
    ((8, 16, 16, 32, 32), 2, 2, 0, False, True, None),
    ((8, 32, 16, 32, 32), 3, 2, 1, False, True, None),
    ((4, 32, 24, 40, 40), 3, 2, 1, False, False, None),
    (
        (4, 64, 32, 32, 32),
        (2, 3, 3),
        (1, 2, 2),
        (0, 1, 1),
        False,
        True,
        None,
    ),
    ((2, 64, 32, 64, 64), 3, 2, 1, True, True, None),
    ((2, 32, 32, 64, 64), 2, 1, 0, False, True, 4),
]


def avg_pool3d_input_fn(config, dtype, device):
    (
        shape,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ) = config
    inp = utils.generate_tensor_input(shape, dtype, device)

    yield inp, {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
    }


@pytest.mark.avg_pool3d
def test_avg_pool3d():
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.ops.aten.avg_pool3d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.avg_pool3d)
    bench.run()
