# THRESHOLD_ operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.threshold_ import (  # noqa: E402
    threshold_ as gems_threshold_,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
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


@pytest.mark.threshold_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("threshold", [0.0, 0.25, -0.75, 0])
@pytest.mark.parametrize("value", [0.0, -0.5, 1.0, 2])
def test_threshold__tensor_inplace(shape, dtype, threshold, value):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.threshold_(ref_input, threshold, value)

    with flag_gems.use_gems():
        act_out = gems_threshold_(act_input, threshold, value)

    gems_assert_close(act_out, ref_out, dtype=dtype)
