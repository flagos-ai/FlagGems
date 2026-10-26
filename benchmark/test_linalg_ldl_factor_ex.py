import pytest
import torch

from . import base


LDL_FACTOR_EX_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]


class LinalgLdlFactorExBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_FACTOR_EX_SHAPES
        self.shape_desc = "N, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"input": list(dims)}, params={}, builder_args=(dims,))


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    n = shape[-1]
    a = torch.randn(shape, dtype=dtype, device=device)
    return (a @ a.mT + torch.eye(n, dtype=dtype, device=device) * 0.1,)


@pytest.mark.linalg_ldl_factor_ex
@pytest.mark.ldl_factor_ex
def test_linalg_ldl_factor_ex():
    LinalgLdlFactorExBenchmark(
        op_name="ldl_factor_ex",
        torch_op=torch.linalg.ldl_factor_ex,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
