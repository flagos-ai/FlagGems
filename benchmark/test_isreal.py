import pytest
import torch

from . import base, consts, utils


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=float_dtype, device=device)
        imag = torch.randn(shape, dtype=float_dtype, device=device)
        imag.reshape(-1)[::2] = 0
        return (torch.complex(real, imag).to(dtype),)
    if dtype.is_floating_point:
        return (torch.randn(shape, dtype=dtype, device=device),)
    return (utils.generate_tensor_input(shape, dtype, device),)


@pytest.mark.isreal
def test_isreal():
    bench = base.GenericBenchmark(
        op_name="isreal",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.isreal,
        dtypes=consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES,
    )
    bench.run()
