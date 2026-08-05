from typing import Generator, List

import pytest
import torch

import flag_gems

from . import base, consts

MESHGRID_SHAPES = [
    [32, 32],
    [1024, 1024],
    [4096, 4096],
    [32, 32, 32],
    [1024, 1024, 1024],
    [4096, 4096, 3],
    [32, 32, 32, 32],
    [128, 128, 128, 128],
    [65504, 1, 1, 1],
]

INDEXING_MODES = ["ij", "xy"]


def generate_tensors(shapes: List[int], dtype, device) -> List[torch.Tensor]:
    return [
        torch.linspace(0, size, size, device=device, dtype=dtype) for size in shapes
    ]


class MeshgridBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(MESHGRID_SHAPES)
        if flag_gems.vendor_name == "ascend":
            self.shapes.insert(0, [256, 256, 16])

    def get_input_iter(self, cur_dtype) -> Generator:
        for shapes in self.shapes:
            tensors = generate_tensors(shapes, cur_dtype, self.device)
            for mode in INDEXING_MODES:
                yield tensors, {"indexing": mode}


@pytest.mark.meshgrid
def test_meshgrid():
    bench = MeshgridBenchmark(
        op_name="meshgrid",
        torch_op=torch.meshgrid,
        gems_op=flag_gems.meshgrid,
        dtype=consts.FLOAT_DTYPES,
    )
    bench.run()
 
