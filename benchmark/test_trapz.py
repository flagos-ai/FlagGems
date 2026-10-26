import pytest
import torch

from . import base, consts


def trapz_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    dim = len(shape) - 1
    yield inp, {"dx": 1.0, "dim": dim}


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(shape={"input": shape}, builder_args=(shape,))


def _build_inputs(plan, dtype, device):
    return next(trapz_input_fn(plan.builder_args[0], dtype, device))


@pytest.mark.trapz
def test_trapz():
    bench = base.GenericBenchmark(
        op_name="trapz",
        torch_op=torch.trapezoid,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
