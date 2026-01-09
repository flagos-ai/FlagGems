# ERF_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.erf_ import erf_ as gems_erf_  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


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


@pytest.mark.erf_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontig", [False, True])
def test_erf__tensor(shape, dtype, noncontig):
    device = "cuda"  # noqa: F841
    if noncontig:
        base = torch.randn(
            (shape[1], shape[0]), device=flag_gems.device, dtype=torch.float32
        )
        input_tensor = base.transpose(0, 1).to(dtype)
    else:
        input_tensor = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    if dtype in (torch.float16, torch.bfloat16):
        input_tensor = input_tensor.clamp(-3, 3)

    ref_input = to_reference(input_tensor)
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.erf_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_erf_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.erf_
def test_perf_aten_erf_():
    # Define input generation logic matching the operator arguments
    def erf__input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if dtype in (torch.float16, torch.bfloat16):
            inp = inp.clamp(-3, 3)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=erf__input_fn,
        op_name="erf_",
        torch_op=torch.ops.aten.erf_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
