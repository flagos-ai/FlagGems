import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReflectionPad2dBenchmark(utils.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33, 33), (1, 1, 1, 1)),
            ((2, 4, 32, 64), (2, 3, 2, 3)),
            ((8, 16, 64, 64), (3, 5, 3, 5)),
            ((32, 64, 128, 256), (0, 4, 0, 4)),
            ((16, 32, 64, 128), (1, 1, 1, 1)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.reflection_pad2d
def test_reflection_pad2d():
    bench = ReflectionPad2dBenchmark(
        op_name="reflection_pad2d",
        torch_op=torch.ops.aten.reflection_pad2d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
