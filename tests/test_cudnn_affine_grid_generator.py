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

# Custom shapes: (N, H, W, C); small edge cases plus odd/prime H, W values.
if cfg.QUICK_MODE:
    # Single minimal case for --quick CI runs.
    CUDNN_AFFINE_GRID_SHAPES = [
        (1, 2, 2, 3),
    ]
else:
    # (N, H, W, C) with degenerate, odd and non-square H, W combinations.
    CUDNN_AFFINE_GRID_SHAPES = [
        (1, 2, 2, 3),
        (1, 1, 5, 2),
        (2, 3, 1, 4),
        (2, 8, 8, 3),
        (4, 16, 16, 1),
        (3, 7, 11, 5),
        (2, 32, 32, 3),
    ]

# cuDNN computes this operator in float32 internally and rounds back to the
# input dtype, so bfloat16 is rejected by the reference kernel; float16,
# float32 and float64 are all supported.
CUDNN_AFFINE_GRID_FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
]


def _reference(theta, N, C, H, W):
    return torch.cudnn_affine_grid_generator(theta, N, C, H, W)


@pytest.mark.cudnn_affine_grid_generator
@pytest.mark.parametrize("shape", CUDNN_AFFINE_GRID_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator(shape, dtype):
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, H, W, C = shape
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)
    ref_theta = utils.to_reference(theta, True)

    ref_out = _reference(ref_theta, N, C, H, W)
    res_out = flag_gems.cudnn_affine_grid_generator(theta, N, C, H, W)

    assert res_out.shape == (N, H, W, 2)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator
@pytest.mark.parametrize("shape", CUDNN_AFFINE_GRID_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_identity(shape, dtype):
    """With theta=identity the grid is the normalized coordinate lattice.

    cuDNN normalizes in the align_corners=True sense
    (``2 * i / (size - 1) - 1``) and collapses size-1 dimensions to -1.0.
    """
    N, H, W, C = shape
    theta = torch.zeros((N, 2, 3), dtype=dtype, device=flag_gems.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0

    # The identity reference is built with plain torch ops, so it runs wherever
    # the reference tensor lives (CPU in quick-cpu mode, GPU otherwise).
    ref_device = utils.to_reference(theta).device

    def coords(size):
        # cuDNN does the arithmetic in float32 and rounds to the output dtype.
        if size == 1:
            return torch.full((1,), -1.0, dtype=torch.float32, device=ref_device)
        idx = torch.arange(size, dtype=torch.float32, device=ref_device)
        return idx * (2.0 / (size - 1)) - 1.0

    norm_x = coords(W)
    norm_y = coords(H)
    ref_out = torch.stack(
        [
            norm_x[None, None, :].expand(N, H, W),
            norm_y[None, :, None].expand(N, H, W),
        ],
        dim=-1,
    ).to(dtype)

    res_out = flag_gems.cudnn_affine_grid_generator(theta, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator_out
@pytest.mark.parametrize("shape", CUDNN_AFFINE_GRID_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_out(shape, dtype):
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, H, W, C = shape
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)
    ref_theta = utils.to_reference(theta, True)
    # to_reference upcasts to float64, so the reference out buffer must match it.
    ref_out = torch.empty((N, H, W, 2), dtype=ref_theta.dtype, device=ref_theta.device)
    res_out = torch.empty((N, H, W, 2), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.cudnn_affine_grid_generator.out(
        ref_theta, N, C, H, W, out=ref_out
    )
    res_out = flag_gems.cudnn_affine_grid_generator_out(theta, N, C, H, W, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_non_contiguous(dtype):
    """theta with non-contiguous strides must produce the same grid."""
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    # Fixed shape (N, C, H, W) so theta can be made non-contiguous.
    N, C, H, W = 3, 4, 9, 13
    theta = torch.randn((N, 3, 2), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    assert not theta.is_contiguous()

    ref_theta = utils.to_reference(theta, True)
    ref_out = _reference(ref_theta, N, C, H, W)
    res_out = flag_gems.cudnn_affine_grid_generator(theta, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator
@pytest.mark.parametrize("shape", CUDNN_AFFINE_GRID_SHAPES)
def test_cudnn_affine_grid_generator_fp64_precision(shape):
    """float64 must be computed in float64, not narrowed to float32.

    The default ``gems_assert_close`` budget (atol=1e-4) hides the narrowing,
    so compare against a float64 eager reference with a strict tolerance.
    """
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, H, W, C = shape
    theta = torch.randn((N, 2, 3), dtype=torch.float64, device=flag_gems.device)
    ref_out = _reference(theta, N, C, H, W)
    res_out = flag_gems.cudnn_affine_grid_generator(theta, N, C, H, W)

    assert res_out.dtype == torch.float64
    # float32 narrowing shows up at ~1e-7; float64 arithmetic stays near 1e-16.
    torch.testing.assert_close(res_out, ref_out, atol=1e-14, rtol=1e-14)


@pytest.mark.cudnn_affine_grid_generator_out
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_out_resizes(dtype):
    """A wrong-shaped or zero-element `out` is resized, as ATen's out does."""
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = 2, 3, 16, 16
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)
    ref_theta = utils.to_reference(theta, True)
    ref_out = torch.empty((N, H, W, 2), dtype=ref_theta.dtype, device=ref_theta.device)
    ref_out = torch.ops.aten.cudnn_affine_grid_generator.out(
        ref_theta, N, C, H, W, out=ref_out
    )

    for out in (
        torch.empty(0, dtype=dtype, device=flag_gems.device),
        torch.empty((N, H, W, 2), dtype=dtype, device=flag_gems.device)[..., :1],
    ):
        res_out = flag_gems.cudnn_affine_grid_generator_out(theta, N, C, H, W, out=out)
        assert res_out.shape == (N, H, W, 2)
        assert res_out.data_ptr() == out.data_ptr()
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator_out
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_out_non_contiguous(dtype):
    """A non-contiguous `out` is written in place, not copied through."""
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = 2, 3, 16, 16
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)
    ref_theta = utils.to_reference(theta, True)
    ref_out = torch.empty((N, H, W, 2), dtype=ref_theta.dtype, device=ref_theta.device)
    ref_out = torch.ops.aten.cudnn_affine_grid_generator.out(
        ref_theta, N, C, H, W, out=ref_out
    )

    # Wider backing buffer sliced to [..., :2]: the elements after the view
    # must not be touched, which catches an offset computed from a contiguous
    # layout instead of `out`'s strides.
    backing = torch.zeros((N, H, W, 4), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.cudnn_affine_grid_generator_out(
        theta, N, C, H, W, out=backing[..., :2]
    )
    assert res_out.data_ptr() == backing[..., :2].data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)
    assert torch.count_nonzero(backing[..., 2:]) == 0

    # Transposed view: neither axis is unit-stride, so both coordinates are
    # stored with out's own strides.
    transposed = torch.zeros((N, 2, W, H), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.cudnn_affine_grid_generator_out(
        theta, N, C, H, W, out=transposed.transpose(1, 3)
    )
    assert res_out.data_ptr() == transposed.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator_out
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_out_rejects(dtype):
    """out must match theta's dtype and device, like ATen."""
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, C, H, W = 2, 3, 4, 5
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)

    other_dtype = torch.float16 if dtype != torch.float16 else torch.float32
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_affine_grid_generator_out(
            theta,
            N,
            C,
            H,
            W,
            out=torch.empty((N, H, W, 2), dtype=other_dtype, device=flag_gems.device),
        )

    with pytest.raises(RuntimeError):
        flag_gems.cudnn_affine_grid_generator_out(
            theta, N, C, H, W, out=torch.empty((N, H, W, 2), dtype=dtype)
        )
