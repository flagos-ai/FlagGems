import pytest
import torch

from . import base, consts


class Rot90Benchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((64, 64), 1, (0, 1)),
            ((128, 128), 2, (0, 1)),
            ((256, 256), 1, (0, 1)),
            ((512, 512), 2, (0, 1)),
            ((1024, 1024), 1, (0, 1)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for (shape, k, dims) in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            yield x, k, dims


@pytest.mark.rot90
def test_rot90():
    bench = Rot90Benchmark(
        op_name="rot90",
        torch_op=lambda x, k, d: torch.ops.aten.rot90(x, k, d),
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
