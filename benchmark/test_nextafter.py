import pytest
import torch

from . import base, consts
from .test_nextafter_ import NextafterInplaceBenchmark


def _legacy_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"inputs": [shape, shape]}, params={}, builder_args=(shape,)
    )


def _legacy_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return (
        base.generate_tensor_input(shape, dtype, device),
        base.generate_tensor_input(shape, dtype, device),
    )


@pytest.mark.nextafter
def test_nextafter():
    bench = base.BinaryPointwiseBenchmark(
        op_name="nextafter",
        torch_op=torch.nextafter,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.nextafter_
def test_nextafter_():
    bench = NextafterInplaceBenchmark(
        op_name="nextafter_",
        torch_op=torch.Tensor.nextafter_,
        case_fn=_legacy_case_fn,
        build_inputs_fn=_legacy_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
