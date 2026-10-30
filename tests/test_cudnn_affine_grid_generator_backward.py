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
    follows the wrong convention, so the reference here is the gradient of
    cudnn's own forward grid. That keeps the pair self-consistent and pins the
    convention on an implementation this kernel does not share code with.
    """
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator_backward is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = shape
    grad = torch.randn((N, H, W, 2), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad, True)

    # d/dtheta sum(cudnn_forward(theta) * grad) at theta = identity. The grid
    # is linear in theta, so the gradient is the constant coefficient of grad.
    theta = torch.zeros(N, 2, 3, dtype=ref_grad.dtype, device=ref_grad.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta.requires_grad_(True)
    grid = torch.ops.aten.cudnn_affine_grid_generator(theta, N, C, H, W)
    (grid * ref_grad).sum().backward()
    ref_out = theta.grad.detach()

    res_out = flag_gems.cudnn_affine_grid_generator_backward(grad, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)
