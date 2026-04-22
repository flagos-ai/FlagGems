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

    res_out = flag_gems.ops.slice_backward(grad_output, shape, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", SLICE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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
    res_out = flag_gems.ops.slice_backward(grad_output, shape, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", SLICE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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
    res_out = flag_gems.ops.slice_backward(grad_output, shape, dim, start, end, step)

    utils.gems_assert_equal(res_out, ref_out)
