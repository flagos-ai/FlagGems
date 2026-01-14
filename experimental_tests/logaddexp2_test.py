# LOGADDEXP2 operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.logaddexp2 import logaddexp2 as gems_logaddexp2
from flag_gems.experimental_ops.logaddexp2 import logaddexp2_out as gems_logaddexp2_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import TO_CPU, gems_assert_close


except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.logaddexp2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp2_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_other = to_reference(other)

    ref_out = torch.ops.aten.logaddexp2(ref_self, ref_other)

    with flag_gems.use_gems():
        act_out = gems_logaddexp2(self, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logaddexp2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp2_out(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_other = to_reference(other)

    ref_out = torch.empty(shape, dtype=dtype, device=ref_self.device)
    torch.ops.aten.logaddexp2.out(ref_self, ref_other, out=ref_out)

    act_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        gems_logaddexp2_out(self, other, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logaddexp2
def test_perf_aten_logaddexp2():
    # Define input generation logic matching the operator arguments
    def logaddexp2_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=logaddexp2_input_fn,
        op_name="logaddexp2",
        torch_op=torch.ops.aten.logaddexp2,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
