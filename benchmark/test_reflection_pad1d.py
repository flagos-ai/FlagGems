import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReflectionPad1dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33), (1, 1)),
            ((2, 4, 64), (3, 5)),
            ((8, 16, 256), (8, 8)),
            ((32, 64, 2048), (3, 5)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.reflection_pad1d
def test_reflection_pad1d():
    bench = ReflectionPad1dBenchmark(
        op_name="reflection_pad1d",
        torch_op=torch.ops.aten.reflection_pad1d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
