import pytest
import torch

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _tensor_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"self": list(dims), "other": list(dims)}, params={}, builder_args=(dims, "tensor"))


def _scalar_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"self": list(dims)}, params={"other": 0.5}, builder_args=(dims, "scalar"))


def _build_inputs_fn(plan, dtype, device):
    shape, kind = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if kind == "scalar":
        return inp, plan.params["other"]
    other = utils.generate_tensor_input(shape, dtype, device)
    other = other.sign().masked_fill_(other == 0, 1) * (other.abs() + 0.5)
    return inp, other


@pytest.mark.true_divide
@pytest.mark.true_divide_tensor
def test_true_divide_tensor_dispatch():
    OperatorBenchmark(op_name="true_divide_tensor", torch_op=torch.ops.aten.true_divide.Tensor, case_fn=_tensor_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()


@pytest.mark.div_scalar
def test_true_divide_scalar():
    OperatorBenchmark(op_name="div_scalar", torch_op=torch.true_divide, case_fn=_scalar_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()
