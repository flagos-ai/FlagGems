import pytest
import torch

from . import base, utils
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return (utils.generate_tensor_input(shape, dtype, device),)


class BesselY1Benchmark(base.UnaryPointwiseBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.special_bessel_y1
@pytest.mark.skip_native(
    vendors=["metax"],
    reason="Native bessel_y1 exceeds the MetaX per-thread private memory limit",
)
def test_special_bessel_y1():
    BesselY1Benchmark(
        op_name="special_bessel_y1",
        torch_op=torch.special.bessel_y1,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
