# LOGADDEXP operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.logaddexp import logaddexp as gems_logaddexp
from flag_gems.experimental_ops.logaddexp import logaddexp_out as gems_logaddexp_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

try:
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


@pytest.mark.logaddexp
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_other = to_reference(other)
    ref_out = torch.ops.aten.logaddexp(ref_self, ref_other)

    with flag_gems.use_gems():
        act_out = gems_logaddexp(self, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp_out(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_self = to_reference(self)
    ref_other = to_reference(other)

    ref_out_buf = torch.empty(shape, dtype=dtype, device=ref_self.device)
    ref_out = torch.ops.aten.logaddexp.out(ref_self, ref_other, out=ref_out_buf)

    act_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_logaddexp_out(self, other, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logaddexp
def test_perf_aten_logaddexp():
    # Define input generation logic matching the operator arguments
    def logaddexp_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=logaddexp_input_fn,
        op_name="logaddexp",
        torch_op=torch.ops.aten.logaddexp,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
