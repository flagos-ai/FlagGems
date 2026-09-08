import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES = [
    (4, 4, 4),
    (8, 8, 8),
    (3, 3, 3),
    (5, 5, 5),
    # Upsampling cases where each input maps to more than 2 outputs per
    # dimension. These exercise the dynamic MAX_OUT_D/H/W loop bound in
    # the kernel; the previous static_range(0, 2) would miss contributions.
    (7, 7, 7),
    (8, 8, 8),
]


# Define shapes for 3D adaptive average pooling
ADAPTIVE_AVG_POOL3D_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 7, 7, 7),
    (1, 2, 10, 10, 10),
    # Small inputs paired with the upsampling output sizes above to expose
    # the static_range(0, 2) bug: ceil(out/in) >= 3 forces the kernel to
    # iterate beyond 2 output positions per input.
    (1, 1, 3, 3, 3),
    (1, 1, 2, 2, 2),
]


@pytest.mark.adaptive_avg_pool3d_backward
@pytest.mark.parametrize("shape", ADAPTIVE_AVG_POOL3D_SHAPES)
@pytest.mark.parametrize("output_size", ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_backward(shape, output_size, dtype):
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        (*shape[:-3], *output_size), device=flag_gems.device, dtype=dtype
    )
    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    # Reference implementation (high-precision upcast)
    ref_grad = torch.ops.aten._adaptive_avg_pool3d_backward.default(
        ref_grad_output, ref_inp
    )

    # GEMS implementation
    with flag_gems.use_gems():
        gems_grad = torch.ops.aten._adaptive_avg_pool3d_backward.default(
            grad_output, inp
        )

    utils.gems_assert_close(
        gems_grad,
        ref_grad,
        dtype,
        reduce_dim=output_size[0] * output_size[1] * output_size[2],
    )
