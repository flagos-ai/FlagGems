import pytest
import torch

import flag_gems

from . import base

VECDOT_DTYPES = (
    [torch.float32]
    if flag_gems.vendor_name in ["iluvatar", "ascend"]
    else [torch.float32, torch.float64]
)

VECDOT_SHAPES = [
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


class VecdotBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = VECDOT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            y = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (x, y)


@pytest.mark.linalg_vecdot
def test_linalg_vecdot_benchmark():
    bench = VecdotBenchmark(
        op_name="linalg_vecdot",
        torch_op=torch.linalg.vecdot,
        dtypes=VECDOT_DTYPES,
    )
    bench.gems_op = flag_gems.linalg_vecdot
    bench.run()


@pytest.mark.linalg_vecdot_out
def test_linalg_vecdot_out_benchmark():
    bench = VecdotBenchmark(
        op_name="linalg_vecdot_out",
        torch_op=torch.linalg.vecdot,
        dtypes=VECDOT_DTYPES,
    )
    bench.gems_op = flag_gems.linalg_vecdot_out
    bench.run()
