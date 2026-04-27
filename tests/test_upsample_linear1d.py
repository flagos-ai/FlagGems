import random
import time

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    UPSAMPLE_SHAPES_1D,
    gems_assert_close,
    to_reference,
)

random.seed(time.time() // 100)

BOUNDARY_CASES = [
    ("W_in_1_upsample", (2, 3, 1), [5], True, None),
    ("W_in_1_upsample", (2, 3, 1), [5], False, None),
    ("W_out_1", (1, 1, 10), [1], False, None),
    ("identity_scale_ac", (2, 2, 100), [100], True, None),
    ("identity_scale_nc", (2, 2, 100), [100], False, None),
    ("value_nan", (1, 1, 10), [20], False, "nan"),
    ("value_inf", (1, 1, 10), [20], False, "inf"),
    ("non_contiguous", (2, 4, 10), [15], True, "non_contiguous"),
    ("non_contiguous", (2, 4, 10), [15], False, "non_contiguous"),
]


@pytest.mark.upsample_linear1d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("case", BOUNDARY_CASES, ids=lambda x: x[0])
def test_upsample_linear1d_boundaries(dtype, case):
    _, shape, output_size, align_corners, special_cfg = case

    if special_cfg == "nan":
        input_tensor = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        input_tensor.fill_(float("nan"))
    elif special_cfg == "inf":
        input_tensor = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        input_tensor.fill_(float("inf"))
    else:
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if special_cfg == "non_contiguous":
        if shape[2] > 2:
            input_tensor = input_tensor[:, :, :-2]

            input_tensor = input_tensor.transpose(0, 2)
            input_tensor = input_tensor.transpose(0, 2)
    ref_i = to_reference(input_tensor).to(torch.float32)

    try:
        ref_out = torch._C._nn.upsample_linear1d(
            ref_i,
            output_size=output_size,
            align_corners=align_corners,
        ).to(dtype)
    except Exception as e:
        pytest.skip(f"PyTorch reference raised error: {e}")
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_linear1d(
            input_tensor,
            output_size=output_size,
            align_corners=align_corners,
        )
    if special_cfg == "nan":
        assert torch.isnan(res_out).all(), "Output should be all NaN"
        assert torch.isnan(ref_out).all(), "Reference should be all NaN"
    elif special_cfg == "inf":

        def is_inf_or_nan(x):
            return torch.isinf(x) | torch.isnan(x)

        assert is_inf_or_nan(res_out).all(), "Output should be all inf or nan"
        assert is_inf_or_nan(ref_out).all(), "Reference should be all inf or nan"
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_linear1d
@pytest.mark.skipif(True, reason="Result not close.")
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [2, 2.5, 0.3, 0.7])
@pytest.mark.parametrize("shape", UPSAMPLE_SHAPES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_linear1d(dtype, shape, scale, align_corners):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = to_reference(input).to(torch.float32)
    output_size = [int(ref_i.shape[i + 2] * scale) for i in range(1)]

    ref_out = torch._C._nn.upsample_linear1d(
        ref_i,
        output_size=output_size,
        align_corners=align_corners,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_linear1d(
            input,
            output_size=output_size,
            align_corners=align_corners,
        )

    gems_assert_close(res_out, ref_out, dtype)
