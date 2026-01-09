# HEAVISIDE_ operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.heaviside_ import heaviside_ as gems_heaviside_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import gems_assert_close, TO_CPU


except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.heaviside_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("values_kind", ["same", "scalar", "row", "col"])
@pytest.mark.parametrize("zero_fraction", [0.0, 0.2])
def test_heaviside__tensor(shape, dtype, values_kind, zero_fraction):
    self_input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if zero_fraction > 0.0:
        mask = torch.rand(shape, device=flag_gems.device) < zero_fraction
        self_input = self_input.masked_fill(mask, 0.0)

    if values_kind == "same":
        values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif values_kind == "scalar":
        values_tensor = torch.randn((), dtype=dtype, device=flag_gems.device)
    elif values_kind == "row":
        values_tensor = torch.randn((1, shape[1]), dtype=dtype, device=flag_gems.device)
    elif values_kind == "col":
        values_tensor = torch.randn((shape[0], 1), dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self_input)
    ref_values = to_reference(values_tensor)

    ref_out = torch.ops.aten.heaviside_(ref_self, ref_values)

    act_self = self_input.clone()
    act_values = values_tensor.clone()
    with flag_gems.use_gems():
        act_out = gems_heaviside_(act_self, act_values)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.heaviside_
def test_perf_aten_heaviside_():
    # Define input generation logic matching the operator arguments
    def heaviside__input_fn(shape, dtype, device):
        self_input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield self_input, values_tensor

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=heaviside__input_fn,
        op_name="heaviside_",
        torch_op=torch.ops.aten.heaviside_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
