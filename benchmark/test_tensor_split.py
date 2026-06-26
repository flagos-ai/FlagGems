from typing import Generator

import pytest
import torch

from . import base, consts


class TensorSplitBenchmark(base.Benchmark):
    """Benchmark for tensor_split operator."""

    def set_shapes(self, shape_file_path=None):
        # Various shapes covering 1D to 4D tensors for split benchmarking
        self.shapes = [
            (64,),
            (256,),
            (1024,),
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (8, 16, 32),
            (16, 32, 64),
            (32, 64, 128),
            (4, 8, 16, 32),
            (8, 16, 32, 64),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            # Split into 3 sections
            sections = 3
            yield inp, sections


@pytest.mark.tensor_split
def test_tensor_split():
    bench = TensorSplitBenchmark(
        op_name="tensor_split",
        torch_op=torch.tensor_split,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
