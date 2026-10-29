# Copyright 2026 FlagOS Contributors
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

from .accuracy_utils import FLOAT_DTYPES, TO_CPU, gems_assert_close, to_reference

MAXPOOL3D_BACKWARD_CONFIGS = [
    # (shape, kernel_size, stride, padding, dilation, ceil_mode)
    # Classic 3x3x3 kernel, stride 2, padding 1
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    # Non-cubic kernel and stride
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    # ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    # Typical 3D CNN shape
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    # Non-symmetric padding
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    # Small input
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
    # Large batch
    ((8, 16, 8, 8, 8), 3, 1, 1, 1, False),
]


@pytest.mark.max_pool3d_with_indices_backward
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    MAXPOOL3D_BACKWARD_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_pool3d_with_indices_backward(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    ascend_dilated_forward = (
        flag_gems.vendor_name == "ascend"
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and tuple(shape) == (1, 1, 9, 9, 9)
        and (kernel_size, stride, padding, dilation, ceil_mode) == (2, 1, 0, 2, False)
    )
    if ascend_dilated_forward and not TO_CPU:
        pytest.skip("Ascend native max_pool3d forward only supports dilation=1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, upcast=True)

    # Forward pass to get indices (reference)
    ref_out = torch.nn.functional.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    # Native forward supplies indices for the backward candidate under test.
    # The proven Ascend dilation=2 gap uses CPU input preparation only in the
    # CPU-reference mode; the backward candidate still executes on the target.
    forward_inp = inp.cpu() if ascend_dilated_forward else inp
    res_out, res_indices = torch.nn.functional.max_pool3d(
        forward_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    forward_shape = res_out.shape
    if ascend_dilated_forward:
        cpu_indices = res_indices
        assert cpu_indices.dtype == torch.int64
        res_out = res_out.to(inp.device)
        # Ascend's native backward requires int32 indices. This fixed grid is
        # within int32 range; prove that the CPU-native indices survive exactly.
        res_indices = cpu_indices.to(device=inp.device, dtype=torch.int32)
        assert torch.equal(res_indices.cpu().to(cpu_indices.dtype), cpu_indices)
    assert res_out.device == inp.device and res_out.dtype == inp.dtype
    assert res_indices.device == inp.device
    assert res_out.shape == res_indices.shape == forward_shape

    # Generate gradient
    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = to_reference(out_grad, upcast=True)

    # Reference backward via autograd
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    # Gems backward via direct call
    expected = (inp.device, inp.dtype, inp.shape)
    res_in_grad = flag_gems.max_pool3d_with_indices_backward(
        out_grad,
        inp,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        res_indices,
    )

    assert isinstance(res_in_grad, torch.Tensor)
    assert (res_in_grad.device, res_in_grad.dtype, res_in_grad.shape) == expected

    # max_pool3d backward is scatter-based (each output grad maps to exactly
    # one input position), so no significant accumulation tolerance needed.
    gems_assert_close(res_in_grad, ref_in_grad, dtype)
