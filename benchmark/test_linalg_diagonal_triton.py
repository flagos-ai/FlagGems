import pytest
import torch
import flag_gems
from flag_gems.ops.linalg_diagonal import linalg_diagonal

from . import base, consts, utils


class DiagonalBenchmark(base.Benchmark):
    def get_input_iter(self, dtype):

        shapes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (128, 256, 256),       
            (64, 128, 128, 128),   
        ]
        for shape in shapes:
            A = utils.generate_tensor_input(shape, dtype, self.device)
            yield A,


@pytest.mark.linalg_diagonal
def test_linalg_diagonal():
    bench = DiagonalBenchmark(
        op_name="linalg_diagonal",
        torch_op=torch.linalg.diagonal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
