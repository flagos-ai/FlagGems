# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

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
@pytest.mark.underscore_adaptive_avg_pool3d_backward
@pytest.mark.parametrize("shape", ADAPTIVE_AVG_POOL3D_SHAPES)
@pytest.mark.parametrize("output_size", ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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
    gems_grad = flag_gems._adaptive_avg_pool3d_backward(grad_output, inp)

    utils.gems_assert_close(
        gems_grad,
        ref_grad,
        dtype,
        reduce_dim=output_size[0] * output_size[1] * output_size[2],
    )


@pytest.mark.adaptive_avg_pool3d_backward
@pytest.mark.underscore_adaptive_avg_pool3d_backward
@pytest.mark.parametrize("shape", ADAPTIVE_AVG_POOL3D_SHAPES)
@pytest.mark.parametrize("output_size", ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_adaptive_avg_pool3d_backward_grad_input(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        (*shape[:-3], *output_size), device=flag_gems.device, dtype=dtype
    )
    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    # Reference implementation (high-precision upcast)
    ref_grad = torch.ops.aten.adaptive_avg_pool3d_backward.grad_input(
        ref_grad_output, ref_inp, grad_input=torch.empty_like(ref_inp)
    )

    # GEMS implementation, writing into a caller-provided buffer
    buf = torch.empty_like(inp)
    gems_grad = flag_gems.adaptive_avg_pool3d_backward_grad_input(
        grad_output, inp, grad_input=buf
    )

    # Out semantics: the result must be the provided buffer itself
    assert gems_grad.data_ptr() == buf.data_ptr()

    utils.gems_assert_close(
        gems_grad,
        ref_grad,
        dtype,
        reduce_dim=output_size[0] * output_size[1] * output_size[2],
    )
