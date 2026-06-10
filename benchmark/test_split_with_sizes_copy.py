import pytest
import torch

from . import base, consts

# Shapes covering 1D to 4D tensors with various dimension sizes
SPLIT_WITH_SIZES_COPY_SHAPES = [
    (10,),
    (10, 4),
    (10, 4, 8),
    (10, 4, 8, 16),
    (16, 32),
    (8, 64, 128),
    (1, 8192),
    (32, 50257),
]


class SplitWithSizesCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPLIT_WITH_SIZES_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Generate split sizes that sum to the first dimension
            dim_size = shape[0]
            split_sizes = [
                dim_size // 4,
                dim_size // 4,
                dim_size - 2 * (dim_size // 4),
            ]
            yield inp, split_sizes, 0  # dim=0


@pytest.mark.split_with_sizes_copy
def test_split_with_sizes_copy():
    bench = SplitWithSizesCopyBenchmark(
        op_name="split_with_sizes_copy",
        torch_op=torch.split_with_sizes_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
