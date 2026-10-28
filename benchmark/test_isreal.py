import pytest
import torch

import flag_gems

from . import base, consts


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"input": list(dims)}, params={}, builder_args=(dims,))


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=float_dtype, device=device)
        imag = torch.randn(shape, dtype=float_dtype, device=device)
        imag.reshape(-1)[::2] = 0
        return (torch.complex(real, imag).to(dtype),)
    if not dtype.is_floating_point:
        low, high = (0, 2) if dtype == torch.bool else (-8, 8)
        return (torch.randint(low, high, shape, dtype=dtype, device=device),)
    return (torch.randn(shape, dtype=dtype, device=device),)


@pytest.mark.isreal
def test_isreal():
    bench = base.GenericBenchmark(
        op_name="isreal",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.isreal,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.set_gems(flag_gems.isreal)
    bench.run()
