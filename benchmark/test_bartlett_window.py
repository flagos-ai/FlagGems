import pytest
import torch

from . import base


BARTLETT_DTYPES = [torch.float32, torch.float64]


def _case_fn(shape, dtype):
    del dtype
    window_length = shape[0] if isinstance(shape, (list, tuple)) else shape
    yield base.BenchmarkCasePlan(
        shape={"window_length": window_length},
        params={"periodic": True},
        builder_args=(window_length,),
    )


def _build_inputs_fn(plan, dtype, device):
    (window_length,) = plan.builder_args
    if base.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame cannot preserve native Float64 window storage")
    return window_length, {
        "periodic": plan.params["periodic"],
        "dtype": dtype,
        "device": device,
    }


@pytest.mark.bartlett_window
def test_bartlett_window():
    base.GenericBenchmark(
        op_name="bartlett_window",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.bartlett_window,
        dtypes=BARTLETT_DTYPES,
    ).run()
