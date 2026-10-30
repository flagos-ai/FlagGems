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
#
import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)


@pytest.mark.cudnn_batch_norm_backward
@pytest.mark.parametrize(
    "shape", [(7, 3), (3, 5, 17), (2, 7, 9, 11), (2, 3, 5, 7, 9), (3, 4, 2051)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last"])
def test_cudnn_batch_norm_backward(shape, dtype, layout):
    _check_backward(shape, dtype, layout)


def _check_backward(shape, dtype, layout, offset=0):
    if layout == "channels_last" and len(shape) not in (4, 5):
        pytest.skip("channels-last format requires rank 4 or 5")
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if layout == "channels_last":
        x = torch.randn(
            (shape[0], *shape[2:], shape[1]), dtype=dtype, device=flag_gems.device
        )
        x = x.permute(0, len(shape) - 1, *range(1, len(shape) - 1))
    if offset:
        x += offset
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    w = torch.randn(shape[1], dtype=param_dtype, device=flag_gems.device)
    grad = torch.randn_like(x)
    xr, wr = [t.detach().cpu().double().requires_grad_() for t in (x, w)]
    br = torch.zeros_like(wr, requires_grad=True)
    # Use the exact saved statistics that produced the reference gradient.
    # Recomputing mean/variance uses a different reduction order and can diverge
    # from the CPU BatchNorm's saved values for large-offset FP64 inputs.
    y, mean, inv = torch.ops.aten.native_batch_norm.default(
        xr, wr, br, None, None, True, 0.1, 1e-5
    )
    expected = torch.autograd.grad(y, (xr, wr, br), grad.cpu().double())
    actual = flag_gems.cudnn_batch_norm_backward(
        x,
        grad,
        w,
        None,
        None,
        mean.to(device=x.device, dtype=param_dtype),
        inv.to(device=x.device, dtype=param_dtype),
        1e-5,
        torch.empty(0, dtype=torch.uint8, device=x.device),
    )
    assert actual[0].dtype == dtype
    assert actual[1].dtype == actual[2].dtype == param_dtype
    tol = (
        3e-3 if dtype == torch.float16 else (3e-5 if dtype == torch.float32 else 1e-10)
    )
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu().double(), e, rtol=tol, atol=tol)
    assert math.prod(shape) // shape[1] > 1


@pytest.mark.cudnn_batch_norm_backward
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="NVIDIA FP64 split reduction"
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "amd",
    reason="ROCm torch has no aten.cudnn_batch_norm reference (MIOpen backend)",
)
@pytest.mark.parametrize(
    "shape",
    [(1, 3, 2048), (1, 3, 2049), (16, 16, 1024), (16, 16, 4098), (16, 8, 128, 128)],
)
@pytest.mark.parametrize("offset", [0, 1024])
def test_cudnn_batch_norm_backward_fp64_split(shape, offset):
    _check_backward(shape, torch.float64, "contiguous", offset)
