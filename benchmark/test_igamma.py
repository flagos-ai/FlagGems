import pytest
import torch

from . import base, consts


class IgammaBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            a = torch.randn(shape, dtype=dtype, device=self.device).abs() + 0.5
            x = torch.randn(shape, dtype=dtype, device=self.device).abs()
            yield a, x


@pytest.mark.igamma
def test_igamma():
    bench = IgammaBenchmark(
        op_name="igamma",
        torch_op=torch.igamma,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
