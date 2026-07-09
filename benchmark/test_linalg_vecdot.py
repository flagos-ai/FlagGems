import pytest
import torch

from . import base


class VecdotBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (10,),
            (100,),
            (1000,),
            (10000,),
            (2, 10),
            (2, 100),
            (2, 1000),
            (4, 10),
            (4, 100),
            (4, 1000),
            (8, 10),
            (8, 100),
            (8, 1000),
            (16, 10),
            (16, 100),
            (16, 1000),
            (32, 10),
            (32, 100),
            (32, 1000),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            y = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (x, y)


@pytest.mark.linalg_vecdot
def test_linalg_vecdot():
    bench = VecdotBenchmark(
        op_name="linalg_vecdot",
        torch_op=torch.linalg.vecdot,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
