import pytest
import torch

from . import base, consts, utils


class TrilBenchmark(base.Benchmark):
    DEFAULT_DTYPES = consts.FLOAT_DTYPES
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (1024, 1024),
            (4096, 4096),
            (64, 512, 512),
            (256, 1024, 1024),
        ]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        diagonals = (0, -32, 32)
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            for diagonal in diagonals:
                yield inp, {"diagonal": diagonal}


@pytest.mark.tril
def test_tril():
    bench = TrilBenchmark(
        op_name="tril",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
