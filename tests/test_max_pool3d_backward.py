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

"""Backward accuracy for ``max_pool3d`` through the public FlagGems entry."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

# Same window set as the value/indices backward test, so the two backward
# entry points are compared over identical geometry.
MAXPOOL3D_BACKWARD_CONFIGS = [
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
    ((8, 16, 8, 8, 8), 3, 1, 1, 1, False),
]


def _native_grad(inp, out_grad, params):
    """Reference input gradient from the native ATen autograd node."""
    ref_inp = to_reference(inp, upcast=True).detach().requires_grad_(True)
    ref_out = torch.nn.functional.max_pool3d(ref_inp, *params)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_inp, to_reference(out_grad, upcast=True)
    )
    return ref_grad


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    MAXPOOL3D_BACKWARD_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_pool3d_backward(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    params = (kernel_size, stride, padding, dilation, ceil_mode)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out_grad = torch.randn(
        _pooled_shape(shape, params), dtype=dtype, device=flag_gems.device
    )

    res_out = flag_gems.max_pool3d(inp, *params)
    (res_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    ref_out = torch.nn.functional.max_pool3d(
        to_reference(inp, upcast=True).detach(), *params
    )
    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_grad, _native_grad(inp, out_grad, params), dtype)


def _pooled_shape(shape, params):
    kernel_size, stride, padding, dilation, ceil_mode = params
    out = []
    for size, k, s, p, d in zip(
        shape[2:],
        _triple(kernel_size),
        _triple(stride if stride is not None else kernel_size),
        _triple(padding),
        _triple(dilation),
    ):
        effective = (k - 1) * d + 1
        numerator = size + 2 * p - effective
        if ceil_mode:
            value = (numerator + s - 1) // s + 1
            if (value - 1) * s >= size + p:
                value -= 1
        else:
            value = numerator // s + 1
        out.append(value)
    return tuple(shape[:2]) + tuple(out)


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(value)


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_pool3d_backward_channels_last_3d(dtype):
    """The scatter indexes NCDHW storage, so the wrapper must stay correct for
    a channels_last_3d input whose ``zeros_like`` would allocate NDHWC."""
    shape = (2, 8, 8, 16, 16)
    params = (3, 2, 1, 1, False)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).to(
        memory_format=torch.channels_last_3d
    )
    inp.requires_grad_(True)
    out_grad = torch.randn(
        _pooled_shape(shape, params), dtype=dtype, device=flag_gems.device
    )

    res_out = flag_gems.max_pool3d(inp, *params)
    (res_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    ref_grad = _native_grad(inp, out_grad, params)
    # ATen preserves the input's channels_last_3d layout for the gradient while
    # the FlagGems wrapper documents an NCDHW-contiguous result.
    assert res_grad.is_contiguous()
    gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_pool3d_backward_tied_maxima(dtype):
    """Equal maxima inside one window are ambiguous in the input gradient.

    ATen keeps the first maximum, so a block of identical values must still
    produce exactly one unit of gradient per pooled element.
    """
    inp = torch.ones((2, 4, 5, 5, 5), dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    params = (2, 2, 0, 1, False)
    res_out = flag_gems.max_pool3d(inp, *params)
    out_grad = torch.ones_like(res_out)
    (res_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    assert res_grad.sum().item() == res_out.numel()
    assert torch.count_nonzero(res_grad).item() == res_out.numel()


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_pool3d_backward_unselected_inputs(dtype):
    """Stride larger than the kernel leaves input voxels unselected; those
    positions must receive exactly zero instead of a stale value."""
    inp = torch.randn((2, 2, 9, 9, 9), dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    params = (2, 3, 0, 1, False)
    res_out = flag_gems.max_pool3d(inp, *params)
    out_grad = torch.randn_like(res_out)
    (res_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    ref_grad = _native_grad(inp, out_grad, params)
    gems_assert_close(res_grad, ref_grad, dtype)
    assert res_grad.shape == inp.shape
    assert res_grad.dtype == inp.dtype
    assert torch.count_nonzero(res_grad) <= res_out.numel()


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_pool3d_backward_expanded_grad(dtype):
    """An expanded upstream gradient must still scatter one value per index."""
    shape = (2, 2, 6, 7, 8)
    params = (2, 2, 0, 1, False)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    pooled = _pooled_shape(shape, params)
    base = torch.randn(
        (shape[0], shape[1], 1, 1, 1), dtype=dtype, device=flag_gems.device
    )
    out_grad = base.expand(pooled)
    assert not out_grad.is_contiguous()

    res_out = flag_gems.max_pool3d(inp, *params)
    (res_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    ref_grad = _native_grad(inp, out_grad, params)
    gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize("dtype", [torch.float64])
def test_accuracy_max_pool3d_backward_float64(dtype):
    """float64 keeps float64 accumulation instead of being widened to fp32."""
    if not utils.fp64_is_supported:
        pytest.skip("float64 is unsupported on this backend")
    shape = (1, 2, 6, 7, 8)
    params = (2, 2, 0, 1, False)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out_grad = torch.randn(
        _pooled_shape(shape, params), dtype=dtype, device=flag_gems.device
    )

    res_out = flag_gems.max_pool3d(inp, *params)
    (res_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    assert res_grad.dtype == dtype
    ref_grad = _native_grad(inp, out_grad, params)
    gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.max_pool3d_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_pool3d_with_indices_backward_entry(dtype):
    """The explicit backward entry must agree with the autograd node on the
    same saved indices, and must not depend on the values of ``self``."""
    shape = (2, 4, 6, 9, 11)
    params = (3, 2, 1, 1, False)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True).detach().requires_grad_(True)
    ref_out, ref_indices = torch.nn.functional.max_pool3d(
        ref_inp, *params, return_indices=True
    )
    out_grad = torch.randn(ref_out.shape, dtype=dtype, device=flag_gems.device)

    res_indices = torch.ops.aten.max_pool3d_with_indices(
        inp, [3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1], False
    )[1]
    res_grad = flag_gems.max_pool3d_with_indices_backward(
        out_grad, inp, *params, res_indices
    )

    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_inp, to_reference(out_grad, upcast=True)
    )
    gems_assert_close(res_grad, ref_grad, dtype)
    utils.gems_assert_equal(res_indices, ref_indices)

    # The gradient must be produced from the saved indices alone. The scatter
    # uses atomics, so repeated runs may order a window's contributions
    # differently; compare within the dtype tolerance instead of bitwise.
    other = torch.randn_like(inp)
    res_grad_other = flag_gems.max_pool3d_with_indices_backward(
        out_grad, other, *params, res_indices
    )
    gems_assert_close(res_grad, to_reference(res_grad_other), dtype)
