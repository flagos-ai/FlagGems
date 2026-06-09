import pytest
import torch

from . import base, consts

# rot90 only supports 2D tensors, use square and rectangular shapes for benchmarking
ROT90_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (100, 200),
    (200, 400),
    (400, 800),
]


class Rot90Benchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ROT90_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, 1, [0, 1]  # k=1, dims=[0, 1]


@pytest.mark.rot90
def test_rot90():
    bench = Rot90Benchmark(
        op_name="rot90",
        torch_op=torch.rot90,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
