import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReflectionPad3dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((2, 4, 8, 16, 16), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 16, 32, 32), (2, 3, 2, 3, 1, 1)),
            ((2, 4, 16, 32, 64), (1, 1, 2, 2, 2, 2)),
            ((1, 1, 32, 64, 128), (0, 4, 0, 4, 0, 4)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.reflection_pad3d
def test_reflection_pad3d():
    bench = ReflectionPad3dBenchmark(
        op_name="reflection_pad3d",
        torch_op=torch.ops.aten.reflection_pad3d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
