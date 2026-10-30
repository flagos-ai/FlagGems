import pytest
import torch

import flag_gems

from . import base, consts


def _host_scalar_log1p(value):
    # Native host reference adaptation for the existing scalar extension;
    # this is not a vendor NVIDIA log1p primitive baseline.
    return torch.special.log1p(torch.tensor(value, dtype=torch.float32, device="cpu"))


def _scalar_gems_fallback(value):
    # Resolve only when called; an injected candidate still wins by op_name.
    return flag_gems.special_log1p(value)


class ScalarSpecialLog1pBenchmark(base.Benchmark):
    def __init__(self, value):
        self.value = value
        super().__init__(
            op_name="special_log1p",
            torch_op=_host_scalar_log1p,
            gems_op=_scalar_gems_fallback,
            dtypes=[torch.float32],
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = [()]

    def get_case_iter(self, dtype):
        yield self._case_from_plan(
            dtype,
            0,
            base.BenchmarkCasePlan(
                shape={}, params={"A": self.value}, builder_args=(self.value,)
            ),
        )

    def build_inputs(self, case):
        return case.builder_args[0].builder_args


@pytest.mark.special_log1p
def test_special_log1p():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_log1p",
        torch_op=torch.ops.aten.special_log1p,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_log1p_out
def test_special_log1p_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_log1p_out",
        torch_op=torch.ops.aten.special_log1p.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_log1p
@pytest.mark.parametrize("inp", [1.0, 5, -0.5])
def test_special_log1p_non_tensor(inp):
    bench = ScalarSpecialLog1pBenchmark(inp)
    bench.run()
