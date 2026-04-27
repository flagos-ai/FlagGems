from typing import Generator

import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class AvgPool3dBenchmark(utils.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_5d = [
            (1, 1, 16, 16, 16),
            (2, 3, 32, 32, 32),
            (4, 16, 16, 32, 32),
            (8, 32, 8, 16, 16),
        ]
        for shape in shapes_5d:
            yield from self.input_fn(shape, cur_dtype, self.device)


def avg_pool3d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"kernel_size": 2, "stride": 2}

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        yield inp, {"kernel_size": 3, "stride": 1, "padding": 1}
        yield inp, {"kernel_size": 2, "stride": 2, "count_include_pad": False}


@pytest.mark.avg_pool3d
def test_avg_pool3d():
    bench = AvgPool3dBenchmark(
        op_name="avg_pool3d",
        input_fn=avg_pool3d_input_fn,
        torch_op=torch.nn.functional.avg_pool3d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
