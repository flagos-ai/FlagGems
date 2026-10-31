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
            yield from self._inputs_for_shape(shape, cur_dtype)

    def _inputs_for_shape(self, shape, cur_dtype):
        matrix = torch.randn(shape, dtype=cur_dtype, device=self.device)
        yield torch.geqrf(matrix)


class _OrgqrCaseBenchmark(OrgqrBenchmark):
    def _inputs_for_shape(self, shape, cur_dtype):
        if cur_dtype == torch.float64 and flag_gems.vendor_name in (
            "kunlunxin",
            "thead",
            "enflame",
        ):
            pytest.skip(
                "Native orgqr reference cannot preserve float64 on this backend"
            )
        if (
            cur_dtype == torch.float64 and flag_gems.vendor_name == "iluvatar"
        ) or flag_gems.vendor_name == "thead":
            input, tau = torch.geqrf(torch.randn(shape, dtype=cur_dtype, device="cpu"))
            yield input.to(self.device), tau.to(self.device)
        else:
            yield from super()._inputs_for_shape(shape, cur_dtype)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return next(self._inputs_for_shape(shape, case.dtype))


@pytest.mark.orgqr
def test_orgqr():
    bench = _OrgqrCaseBenchmark(
        op_name="orgqr",
        torch_op=torch.orgqr,
        dtypes=[torch.float32, torch.float64],
    )
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
