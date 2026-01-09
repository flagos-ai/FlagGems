# LOGIT operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.logit import logit as gems_logit
from flag_gems.experimental_ops.logit import logit_out as gems_logit_out

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


@pytest.mark.logit
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [None, 1e-6, 1e-3])
def test_logit_tensor(shape, dtype, eps):
    if eps is None:
        base = (
            torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 0.998
            + 0.001
        )
    else:
        base = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
        flat = base.view(-1)
        if flat.numel() >= 2:
            flat[0] = 0.0
            flat[1] = 1.0
    input_tensor = base.to(dtype)

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.logit(ref_input, eps)

    with flag_gems.use_gems():
        act_out = gems_logit(input_tensor, eps)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logit
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [None, 1e-6, 1e-3])
def test_logit_out(shape, dtype, eps):
    if eps is None:
        base = (
            torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 0.998
            + 0.001
        )
    else:
        base = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
        flat = base.view(-1)
        if flat.numel() >= 2:
            flat[0] = 0.0
            flat[1] = 1.0
    input_tensor = base.to(dtype)

    ref_input = to_reference(input_tensor)
    ref_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.logit.out(ref_input, eps, out=ref_out_buf)  # noqa: F841

    with flag_gems.use_gems():
        act_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        act_out = gems_logit_out(input_tensor, eps, act_out_buf)  # noqa: F841

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)


@pytest.mark.logit
def test_perf_aten_logit():
    # Define input generation logic matching the operator arguments
    def logit_input_fn(shape, dtype, device):
        base = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
        input_tensor = base.to(dtype)
        yield input_tensor,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=logit_input_fn,
        op_name="logit",
        torch_op=torch.ops.aten.logit,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
