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
import unittest.mock

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg
from .conftest import TO_CPU

# Custom shapes: (N, C, H, W); the spatial (H, W) extent is the reduction
# dimension, so the shapes also cover the degenerate 1x1 grid and large
# reduction planes.
if cfg.QUICK_MODE:
    # Single minimal case for --quick CI runs.
    AFFINE_GRID_BACKWARD_SHAPES = [
        (2, 3, 4, 5),
    ]
else:
    # (N, C, H, W) with degenerate 1x1 grids, odd and non-square H, W values.
    AFFINE_GRID_BACKWARD_SHAPES = [
        (1, 1, 1, 1),
        (1, 3, 1, 8),
        (2, 3, 8, 1),
        (2, 3, 4, 5),
        (3, 7, 16, 24),
        (2, 3, 64, 64),
        (1, 5, 127, 129),
        (2, 3, 256, 256),
    ]

# cudnn does not support bfloat16 for this operator (CUDNN_STATUS_BAD_PARAM);
# float16 and float32 are supported (cudnn computes in float32 internally).
CUDNN_AFFINE_GRID_BACKWARD_DTYPES = [torch.float16, torch.float32]


def _make_grad(shape, dtype):
    N, H, W = shape[0], shape[2], shape[3]
    grad = torch.randn((N, H, W, 2), dtype=dtype, device=flag_gems.device)
    # to_reference upcasts to float64, matching the reference out-buffer dtype.
    ref_grad = utils.to_reference(grad, True)
    return grad, ref_grad


def _normalized_coords(size, dtype, device):
    """cudnn's align_corners coordinate for a spatial axis of length `size`.

    cudnn clamps the denominator to 1, so a unit dimension maps to -1.0 rather
    than aten's 0.0. This is the convention the operator under test follows.
    """
    if size == 1:
        return torch.tensor([-1.0], dtype=dtype, device=device)
    idx = torch.arange(size, dtype=torch.float64, device=device)
    return (idx * (2.0 / (size - 1)) - 1.0).to(dtype)


def _manual_backward(ref_grad, N, C, H, W):
    """Closed-form Jacobian of cudnn's forward grid w.r.t. theta.

    The grid is linear in theta::

        grid[n, h, w, 0] = theta[n, 0, 0] * w_norm + theta[n, 0, 1] * h_norm
                         + theta[n, 0, 2]
        grid[n, h, w, 1] = theta[n, 1, 0] * w_norm + theta[n, 1, 1] * h_norm
                         + theta[n, 1, 2]

    so the vector-Jacobian product is just the reduction of `ref_grad` against
    the normalized coordinates. Computing it here rather than through
    autograd keeps the reference independent of
    `cudnn_affine_grid_generator_backward`, which is what `derivatives.yaml`
    binds to the cudnn forward's backward.
    """
    dtype = ref_grad.dtype
    device = ref_grad.device
    h_norm = _normalized_coords(H, dtype, device)
    w_norm = _normalized_coords(W, dtype, device)

    out = torch.zeros((N, 2, 3), dtype=dtype, device=device)
    for ch in range(2):
        out[:, ch, 0] = (ref_grad[..., ch] * w_norm[None, None, :]).sum(dim=(1, 2))
        out[:, ch, 1] = (ref_grad[..., ch] * h_norm[None, :, None]).sum(dim=(1, 2))
        out[:, ch, 2] = ref_grad[..., ch].sum(dim=(1, 2))
    return out


@pytest.mark.cudnn_affine_grid_generator_backward
@pytest.mark.parametrize("shape", AFFINE_GRID_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_BACKWARD_DTYPES)
def test_cudnn_affine_grid_generator_backward(shape, dtype):
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = shape
    grad, ref_grad = _make_grad(shape, dtype)

    ref_out = torch.ops.aten.cudnn_affine_grid_generator_backward(ref_grad, N, C, H, W)
    res_out = flag_gems.cudnn_affine_grid_generator_backward(grad, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_affine_grid_generator_backward_out
@pytest.mark.parametrize("shape", AFFINE_GRID_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_BACKWARD_DTYPES)
def test_cudnn_affine_grid_generator_backward_out(shape, dtype):
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = shape
    grad, ref_grad = _make_grad(shape, dtype)
    # to_reference upcasts to float64, so the reference out buffer must match it.
    ref_out = torch.empty((N, 2, 3), dtype=ref_grad.dtype, device=ref_grad.device)
    out = torch.empty((N, 2, 3), dtype=dtype, device=flag_gems.device)

    torch.ops.aten.cudnn_affine_grid_generator_backward(
        ref_grad, N, C, H, W, out=ref_out
    )
    res_out = flag_gems.cudnn_affine_grid_generator_backward_out(
        grad, N, C, H, W, out=out
    )

    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_affine_grid_generator_backward
@pytest.mark.parametrize("shape", [(2, 3, 16, 20), (2, 3, 33, 17)])
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_BACKWARD_DTYPES)
def test_cudnn_affine_grid_generator_backward_extreme(shape, dtype):
    """All-zero and constant-gradient inputs, where the exact sums are known."""
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = shape
    for fill in (0.0, 1.0):
        grad = torch.full((N, H, W, 2), fill, dtype=dtype, device=flag_gems.device)
        ref_grad = utils.to_reference(grad, True)

        ref_out = torch.ops.aten.cudnn_affine_grid_generator_backward(
            ref_grad, N, C, H, W
        )
        res_out = flag_gems.cudnn_affine_grid_generator_backward(grad, N, C, H, W)

        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_affine_grid_generator_backward
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_BACKWARD_DTYPES)
def test_cudnn_affine_grid_generator_backward_non_contiguous(dtype):
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = 2, 3, 16, 12
    base = torch.randn((N, H, W, 4), dtype=dtype, device=flag_gems.device)
    grad = base[..., ::2]
    assert not grad.is_contiguous()

    ref_grad = utils.to_reference(grad, True)
    ref_out = torch.ops.aten.cudnn_affine_grid_generator_backward(ref_grad, N, C, H, W)
    res_out = flag_gems.cudnn_affine_grid_generator_backward(grad, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_affine_grid_generator_backward
@pytest.mark.parametrize("shape", [(2, 3, 16, 12), (2, 3, 128, 128)])
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_BACKWARD_DTYPES)
def test_cudnn_affine_grid_generator_backward_out_non_contiguous(shape, dtype):
    """A strided out tensor, on both the direct and the chunked reduction path."""
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = shape
    grad = torch.randn((N, H, W, 2), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad, True)

    # Strided out buffers carved from a wider [N, 2, 6] tensor.
    ref_buf = torch.zeros((N, 2, 6), dtype=ref_grad.dtype, device=ref_grad.device)
    out_buf = torch.zeros((N, 2, 6), dtype=dtype, device=flag_gems.device)
    ref_out = ref_buf[:, :, ::2]
    out = out_buf[:, :, ::2]
    assert not out.is_contiguous()

    torch.ops.aten.cudnn_affine_grid_generator_backward(
        ref_grad, N, C, H, W, out=ref_out
    )
    res_out = flag_gems.cudnn_affine_grid_generator_backward_out(
        grad, N, C, H, W, out=out
    )

    assert res_out.data_ptr() == out.data_ptr()
    # The untouched gaps of the buffer must not have been written to.
    assert torch.count_nonzero(out_buf[:, :, 1::2]) == 0
    utils.gems_assert_close(out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_affine_grid_generator_backward
@pytest.mark.parametrize(
    "shape", [(1, 3, 1, 1), (1, 3, 1, 8), (2, 3, 8, 1), (2, 3, 4, 5)]
)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_BACKWARD_DTYPES)
def test_cudnn_affine_grid_generator_backward_unit_dim(shape, dtype):
    """Unit spatial dimensions, checked against cudnn's *forward*, not against
    its backward.

    For a unit dimension cudnn and native aten disagree: cudnn clamps the
    align_corners denominator to 1 and places the single sample at -1, while
    aten's ``affine_grid_generator`` places it at 0. Comparing only against
    ``cudnn_affine_grid_generator_backward`` cannot detect a kernel that
    follows the wrong convention, so the reference here is an independent
    closed form of the Jacobian rather than another path through PyTorch.

    Deriving ``theta.grad`` from ``cudnn_affine_grid_generator`` would not be
    independent: ``derivatives.yaml`` binds that backward to
    ``cudnn_affine_grid_generator_backward``, i.e. the very operator under
    test.
    """
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = shape
    grad = torch.randn((N, H, W, 2), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad, True)
    ref_out = _manual_backward(ref_grad, N, C, H, W)

    res_out = flag_gems.cudnn_affine_grid_generator_backward(grad, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_affine_grid_generator_backward
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_BACKWARD_DTYPES)
def test_cudnn_affine_grid_generator_backward_out_dispatch(dtype):
    """The `.out` overload must be reachable through dispatch.

    `.out` ships only a `CompositeExplicitAutograd` kernel, so a device-key
    registration that never lists the `.out` key leaves the whole overload on
    the native implementation: the call succeeds and returns correct values,
    which is why the direct-call tests cannot notice. Calling the real schema
    under `only_enable()` exercises the same registrar `use_gems()` uses, and
    the result is compared against the non-`.out` operator.
    """
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = 2, 3, 8, 6
    grad = torch.randn((N, H, W, 2), dtype=dtype, device=flag_gems.device)
    expected = flag_gems.cudnn_affine_grid_generator_backward(grad, N, C, H, W)

    # The native `.out` kernel also produces the right values, so a value
    # comparison alone passes whether or not the registration exists. Count
    # calls on the implementation the registrar actually binds instead; the spy
    # is installed by swapping the entry in `FULL_CONFIG_BY_FUNC`, the table
    # `only_enable` reads when it builds its registration list (same pattern as
    # `test_nested_tensor_from_tensor_list.py`).
    calls = []
    real_out_op = flag_gems.cudnn_affine_grid_generator_backward_out

    def spy(*args, **kwargs):
        calls.append(args)
        return real_out_op(*args, **kwargs)

    mapping = dict(flag_gems.FULL_CONFIG_BY_FUNC)
    mapping["cudnn_affine_grid_generator_backward_out"] = [
        ("cudnn_affine_grid_generator_backward.out", spy)
    ]

    out = torch.empty_like(expected)
    # A private Library keeps a previous parametrization's registration (which
    # would still hold the first spy) from leaking into this case.
    lib = torch.library.Library("aten", "IMPL")
    try:
        with unittest.mock.patch.object(flag_gems, "FULL_CONFIG_BY_FUNC", mapping):
            flag_gems.only_enable(
                lib=lib,
                include=[
                    "cudnn_affine_grid_generator_backward",
                    "cudnn_affine_grid_generator_backward_out",
                ],
            )
            torch.ops.aten.cudnn_affine_grid_generator_backward.out(
                grad, N, C, H, W, out=out
            )
    finally:
        lib._destroy()

    assert len(calls) == 1, "the .out overload did not dispatch to FlagGems"
    utils.gems_assert_close(out, expected, dtype)
