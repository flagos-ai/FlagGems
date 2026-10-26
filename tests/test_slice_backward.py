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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

SLICE_BACKWARD_SHAPES = [
    (128, 256),
    (1024, 1024),
    (512, 1024, 512),
    (16, 8192, 4096),
    (8, 4096, 11008),
    (4, 32, 4096, 128),
    (32, 256, 256, 128),
]

random.seed(time.time() // 100)


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", SLICE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("start", [0, 16])
@pytest.mark.parametrize("end", [64, 128])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_slice_backward(shape, dim, start, end, step, dtype):
    device = flag_gems.device

    ndim = len(shape)
    dim = dim % ndim
    size = shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    valid_shape = list(shape)

    slice_len = (end - start + step - 1) // step
    valid_shape[dim] = slice_len

    grad_output = torch.randn(valid_shape, dtype=dtype, device=device)

    ref_grad_output = utils.to_reference(grad_output)
    ref_out = torch.ops.aten.slice_backward(
        ref_grad_output, shape, dim, start, end, step
    )

    res_out = flag_gems.slice_backward(grad_output, shape, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "layout",
    [
        "expand_scalar",
        "expand_row",
        "transpose",
        "strided_offset",
        "offset",
        "contiguous",
    ],
)
@pytest.mark.parametrize(
    "dim,start,end,step", [(0, 0, 2, 1), (1, 1, 7, 2), (-1, -7, 8, 3), (0, 2, 2, 1)]
)
def test_slice_backward_gradient_layout(dtype, layout, dim, start, end, step):
    shape = (4, 8)
    sliced = torch.empty(shape)[
        tuple(
            slice(start, end, step) if d == dim % 2 else slice(None) for d in range(2)
        )
    ]
    rows, cols = sliced.shape
    device = flag_gems.device
    if layout == "expand_scalar":
        grad = torch.tensor(2, dtype=dtype, device=device).expand(rows, cols)
    elif layout == "expand_row":
        grad = (
            torch.arange(cols, dtype=dtype, device=device)
            .reshape(1, cols)
            .expand(rows, cols)
        )
    elif layout == "transpose":
        grad = (
            torch.arange(rows * cols, dtype=dtype, device=device)
            .reshape(cols, rows)
            .t()
        )
    elif layout == "strided_offset":
        grad = torch.arange(2 * rows * cols + 1, dtype=dtype, device=device)[
            1::2
        ].reshape(rows, cols)
    elif layout == "offset":
        grad = torch.arange(rows * cols + 1, dtype=dtype, device=device)[1:].reshape(
            rows, cols
        )
    else:
        grad = torch.arange(rows * cols, dtype=dtype, device=device).reshape(rows, cols)

    ref_out = torch.ops.aten.slice_backward(grad.cpu(), shape, dim, start, end, step)
    res_out = flag_gems.slice_backward(grad, shape, dim, start, end, step)
    assert res_out.shape == shape
    assert res_out.dtype == dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", SLICE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_slice_backward_oob_end(shape, dtype):
    # Regression test: end > dim_size caused out-of-bounds write in kernel.
    device = flag_gems.device
    dim = 1 % len(shape)
    dim_size = shape[dim]
    start = 0
    end = dim_size + 100  # intentionally out of bounds
    step = 1

    # grad_output shape matches what PyTorch would produce (clamped slice)
    valid_shape = list(shape)
    valid_shape[dim] = dim_size
    grad_output = torch.randn(valid_shape, dtype=dtype, device=device)
    ref_grad_output = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.slice_backward(
        ref_grad_output, shape, dim, start, end, step
    )
    res_out = flag_gems.slice_backward(grad_output, shape, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", SLICE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_slice_backward_oob_start(shape, dtype):
    # Regression test: start > dim_size caused out-of-bounds write in kernel.
    device = flag_gems.device
    dim = 1 % len(shape)
    dim_size = shape[dim]
    start = dim_size + 50  # intentionally out of bounds
    end = dim_size + 100
    step = 1

    # grad_output is empty since clamped slice is empty
    valid_shape = list(shape)
    valid_shape[dim] = 0
    grad_output = torch.randn(valid_shape, dtype=dtype, device=device)
    ref_grad_output = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.slice_backward(
        ref_grad_output, shape, dim, start, end, step
    )
    res_out = flag_gems.slice_backward(grad_output, shape, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)
