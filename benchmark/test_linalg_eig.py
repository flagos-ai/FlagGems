import pytest
import torch

from . import base, consts


class LinalgEigBenchmark(base.Benchmark):
    def set_more_shapes(self):
        self.shapes = [(4, 2, 2), (16, 2, 2), (64, 2, 2), (256, 2, 2)]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield {"A": inp}

    def get_golden_fn(self):
        return torch.linalg.eig

    def get_fn(self):
        return torch.linalg.eig


@pytest.mark.linalg_eig
def test_linalg_eig():
    bench = LinalgEigBenchmark(
        op_name="linalg_eig",
        torch_op=torch.linalg.eig,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
