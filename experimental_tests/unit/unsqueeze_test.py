# UNSQUEEZE operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.unsqueeze import (  # noqa: E402
    unsqueeze as gems_unsqueeze,
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


@pytest.mark.unsqueeze
@pytest.mark.parametrize(
    "shape_dim",
    [
        ((), 0),
        ((), -1),
        ((2, 3), -3),
        ((2, 3), -1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), 2),
        ((128, 256), -3),
        ((128, 256), -1),
        ((128, 256), 0),
        ((128, 256), 2),
        ((512, 512), -3),
        ((512, 512), 0),
        ((512, 512), 2),
        ((64, 32, 16), -4),
        ((64, 32, 16), -1),
        ((64, 32, 16), 0),
        ((64, 32, 16), 2),
        ((64, 32, 16), 3),
        ((4,), -2),
        ((4,), -1),
        ((4,), 0),
        ((4,), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unsqueeze_tensor(shape_dim, dtype):
    shape, dim = shape_dim
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.unsqueeze(ref_input, dim)

    with flag_gems.use_gems():
        act_out = gems_unsqueeze(input_tensor, dim)

    gems_assert_close(act_out, ref_out, dtype=dtype)
