import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReflectionPad1dBenchmark(utils.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33), (1, 1)),
            ((2, 4, 64), (3, 5)),
            ((8, 16, 256), (8, 8)),
            ((32, 64, 2048), (3, 5)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.reflection_pad1d
def test_reflection_pad1d():
    bench = ReflectionPad1dBenchmark(
        op_name="reflection_pad1d",
        torch_op=torch.ops.aten.reflection_pad1d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
