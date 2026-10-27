import pytest
import torch

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


def _input_fn(shape, cur_dtype, device):
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    # Use dim=0 for 1D, dim=1 for 2D+
    if len(shape) == 1:
        yield inp, 0
    elif len(shape) >= 2:
        yield inp, 1


@pytest.mark.unsqueeze
def test_unsqueeze():
    bench = base.GenericBenchmark(
        op_name="unsqueeze",
        input_fn=_input_fn,
        torch_op=torch.unsqueeze,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _inplace_case_fn(shape, dtype):
    if not shape:
        return
    dim = 0 if len(shape) == 1 else 1
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dim": dim},
        builder_args=(shape,),
    )


def _inplace_build_inputs(plan, dtype, device):
    return next(_input_fn(plan.builder_args[0], dtype, device))


@pytest.mark.unsqueeze_
def test_unsqueeze_():
    bench = OperatorBenchmark(
        op_name="unsqueeze_",
        case_fn=_inplace_case_fn,
        build_inputs_fn=_inplace_build_inputs,
        torch_op=torch.Tensor.unsqueeze_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
