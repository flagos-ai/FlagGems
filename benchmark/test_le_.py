import pytest
import torch

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _input_fn_scalar(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 0


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"other": 0},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    return next(_input_fn_scalar(plan.builder_args[0], dtype, device))


@pytest.mark.le_
def test_le_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="le_",
        torch_op=lambda a, b: torch.ops.aten.le_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.le_scalar_
def test_le_scalar_():
    bench = OperatorBenchmark(
        op_name="le_scalar_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=lambda a, b: torch.ops.aten.le_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
