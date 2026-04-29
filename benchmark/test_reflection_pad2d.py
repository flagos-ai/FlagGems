import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReflectionPad2dBenchmark(base.Benchmark):
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

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.reflection_pad2d
def test_reflection_pad2d():
    bench = ReflectionPad2dBenchmark(
        op_name="reflection_pad2d",
        torch_op=torch.ops.aten.reflection_pad2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
