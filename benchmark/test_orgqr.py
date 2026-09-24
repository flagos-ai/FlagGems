import pytest
import torch

import flag_gems

from . import base

ORGQR_SHAPES = [(16, 8), (32, 16), (64, 32), (128, 64), (128, 128)]


class OrgqrBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ORGQR_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            matrix = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield torch.geqrf(matrix)


@pytest.mark.orgqr
def test_orgqr():
    bench = OrgqrBenchmark(
        op_name="orgqr",
        torch_op=torch.orgqr,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(flag_gems.orgqr)
    bench.run()


@pytest.mark.orgqr_out
def test_orgqr_out():
    def torch_orgqr_out(input, tau):
        return torch.orgqr(input, tau, out=torch.empty_like(input))

    def gems_orgqr_out(input, tau):
        return flag_gems.orgqr_out(input, tau, out=torch.empty_like(input))

    bench = OrgqrBenchmark(
        op_name="orgqr_out",
        torch_op=torch_orgqr_out,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(gems_orgqr_out)
    bench.run()
