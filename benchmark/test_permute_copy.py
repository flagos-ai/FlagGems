import pytest
import torch

from . import base, consts

# Shapes covering 3D and 4D tensors
CUSTOM_SHAPES = [
    (2, 3, 4),
    (16, 128, 64),
    (32, 64, 128),
    (2, 3, 4, 5),
    (4, 16, 32, 64),
]


class PermuteCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CUSTOM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Move last dim to front, 0 to back
            ndim = len(shape)
            dims = tuple([ndim - 1] + list(range(1, ndim - 1)) + [0])
            yield x, dims


@pytest.mark.permute_copy
def test_permute_copy():
    bench = PermuteCopyBenchmark(
        op_name="permute_copy",
        torch_op=torch.permute_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
