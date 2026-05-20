import pytest
import torch

from . import base, consts

LOGSUMEXP_SHAPES = [
    (256, 256),
    (1024, 1024),
    (4096, 4096),
]


class LogsumexpBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LOGSUMEXP_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, 1


@pytest.mark.special_logsumexp
def test_special_logsumexp():
    bench = LogsumexpBenchmark(
        op_name="special_logsumexp",
        torch_op=torch.special.logsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
