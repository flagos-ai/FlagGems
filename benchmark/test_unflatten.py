import pytest
import torch

from . import base, consts


class UnflattenBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((2, 64, 32, 32), 1, (8, 8)),
            ((4, 128, 64, 64), 1, (16, 8)),
            ((8, 256, 128, 128), 2, (2, 128)),
            ((16, 512, 256, 256), 1, (32, 16)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for (shape, dim, sizes) in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            yield x, dim, sizes


@pytest.mark.unflatten
def test_unflatten():
    bench = UnflattenBenchmark(
        op_name="unflatten",
        torch_op=lambda x, d, s: torch.ops.aten.unflatten(x, d, s),
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
