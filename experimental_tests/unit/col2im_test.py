# col2im (fold) operator accuracy tests

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.col2im import col2im as gems_col2im

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def _compute_L(H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW):
    """Compute number of sliding positions L for given col2im parameters."""
    eff_kh = dH * (kH - 1) + 1
    eff_kw = dW * (kW - 1) + 1
    H_col = (H_out + 2 * pH - eff_kh) // sH + 1
    W_col = (W_out + 2 * pW - eff_kw) // sW + 1
    return H_col * W_col


@pytest.mark.col2im
@pytest.mark.parametrize(
    "N, C, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW",
    [
        # Basic 3×3 kernel, stride=1, padding=1
        (1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1),
        # Rectangular kernel, stride=2, asymmetric padding
        (2, 8, 16, 12, 3, 2, 2, 2, 1, 0, 1, 1),
        # Dilation > 1, asymmetric stride/padding
        (3, 6, 14, 15, 3, 3, 2, 1, 2, 1, 2, 1),
        # Single spatial element (1×1 kernel)
        (2, 3, 4, 4, 1, 1, 1, 1, 0, 0, 1, 1),
        # Larger spatial size
        (1, 16, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_col2im(N, C, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW, dtype):
    L = _compute_L(H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW)
    assert L > 0, "Invalid parameters: L must be positive"

    inp = torch.randn((N, C * kH * kW, L), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True)

    output_size = (H_out, W_out)
    kernel_size = (kH, kW)
    stride_val = (sH, sW)
    padding_val = (pH, pW)
    dilation_val = (dH, dW)

    ref_out = torch.nn.functional.fold(
        ref_inp,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation_val,
        padding=padding_val,
        stride=stride_val,
    )

    with flag_gems.use_gems():
        res_out = gems_col2im(
            inp, output_size, kernel_size, dilation_val, padding_val, stride_val
        )

    gems_assert_close(res_out, ref_out, dtype=dtype)


@pytest.mark.col2im
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_col2im_2d_input(dtype):
    """Test with 2D input (no batch dimension)."""
    C, kH, kW = 3, 3, 3
    H_out, W_out = 8, 8
    sH, sW, pH, pW, dH, dW = 1, 1, 1, 1, 1, 1
    L = _compute_L(H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW)

    inp = torch.randn((C * kH * kW, L), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True)

    output_size = (H_out, W_out)
    kernel_size = (kH, kW)

    # Reference: add batch dim → fold → squeeze
    ref_out = torch.nn.functional.fold(
        ref_inp.unsqueeze(0),
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=(dH, dW),
        padding=(pH, pW),
        stride=(sH, sW),
    ).squeeze(0)

    with flag_gems.use_gems():
        res_out = gems_col2im(
            inp, output_size, kernel_size, (dH, dW), (pH, pW), (sH, sW)
        )

    gems_assert_close(res_out, ref_out, dtype=dtype)
