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

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

COL2IM_CONFIGS = [
    # (batch, channels, kernel_size, output_size, stride, padding, dilation)
    (2, 3, (2, 2), (4, 5), (1, 1), (0, 0), (1, 1)),
    (1, 2, (3, 3), (8, 8), (2, 2), (0, 0), (1, 1)),
    (2, 4, (2, 2), (6, 6), (1, 1), (1, 1), (1, 1)),
    (1, 2, (2, 2), (8, 8), (1, 1), (0, 0), (2, 2)),
    (2, 3, (3, 3), (10, 12), (2, 2), (1, 1), (2, 2)),
]

if QUICK_MODE:
    COL2IM_CONFIGS = COL2IM_CONFIGS[:2]


@pytest.mark.col2im
@pytest.mark.parametrize("config", COL2IM_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_col2im(config, dtype):
    batch, channels, kernel_size, output_size, stride, padding, dilation = config
    kernel_h, kernel_w = kernel_size
    output_h, output_w = output_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    L_h = (output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    L_w = (output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    L = L_h * L_w

    inp = torch.randn(
        batch, channels * kernel_h * kernel_w, L, device=flag_gems.device, dtype=dtype
    )
    ref_inp = to_reference(inp, True)

    ref_out = torch.ops.aten.col2im(
        ref_inp, output_size, kernel_size, dilation, padding, stride
    )
    res_out = flag_gems.col2im(inp, output_size, kernel_size, dilation, padding, stride)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=kernel_h * kernel_w)


@pytest.mark.col2im_backward
@pytest.mark.parametrize(
    "shape,output_size,kernel_size,dilation,padding,stride",
    [
        ((1, 9, 4), (4, 4), (3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 9, 1024), (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)),
        # Non-unit stride exercises the overlapping-window scatter.
        ((1, 9, 225), (32, 32), (3, 3), (1, 1), (0, 0), (2, 2)),
        ((2, 16, 3844), (128, 128), (4, 4), (2, 2), (1, 1), (2, 2)),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_col2im_backward(
    shape, output_size, kernel_size, dilation, padding, stride, dtype
):
    """Drive the gradient through the autograd of the forward ``col2im``."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = to_reference(inp.detach()).requires_grad_(True)
    args = (output_size, kernel_size, dilation, padding, stride)

    ref_out = torch.ops.aten.col2im(ref_inp, *args)
    out_grad = torch.randn(ref_out.shape, dtype=dtype, device=flag_gems.device)

    (res_in_grad,) = torch.autograd.grad(flag_gems.col2im(inp, *args), inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, to_reference(out_grad))

    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=1)


@pytest.mark.col2im_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_col2im_backward_expanded_grad(dtype):
    """An expanded upstream gradient is non-contiguous and must not be read linearly."""
    inp = torch.randn((1, 9, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.detach()).requires_grad_(True)
    args = ((4, 4), (3, 3), (1, 1), (0, 0), (1, 1))

    ref_out = torch.ops.aten.col2im(ref_inp, *args)
    out_grad = torch.randn(
        (ref_out.shape[0], 1, *ref_out.shape[2:]), dtype=dtype, device=flag_gems.device
    ).expand(ref_out.shape)

    (res_in_grad,) = torch.autograd.grad(
        flag_gems.col2im(inp.requires_grad_(), *args), inp, out_grad
    )
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, to_reference(out_grad))
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=1)
