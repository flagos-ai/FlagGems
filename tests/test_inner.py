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

# torch.inner contracts the last dimension of both operands and keeps the
# leading dimensions per-side (no cross broadcasting), so the shape cases cover
# the 1-D dot product, the 2-D matrix product, batched operands, and the
# degenerate 0-D / matvec dispatches.
INNER_SHAPES = [
    ((1024,), (1024,)),
    ((128, 64), (256, 64)),
    ((64, 512), (128, 512)),
    ((8, 16, 256), (32, 256)),
    ((4, 8, 32), (3, 5, 32)),
    ((256, 128), (128,)),
    ((1, 4096), (7, 4096)),
    ((64,), (4, 64)),
    ((2, 3, 4), (2, 3, 4)),
]


@pytest.mark.inner
@pytest.mark.parametrize("input_shape,other_shape", INNER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_inner(input_shape, other_shape, dtype):
    # The contraction length drives the accumulated round-off, so the reference
    # is computed in higher precision and the reduction tolerance is sized to
    # the contracted dimension.
    k = input_shape[-1]
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_other = utils.to_reference(other, upcast=True)

    res_out = flag_gems.inner(inp, other)

    ref_out = torch.inner(ref_inp, ref_other).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=k)


@pytest.mark.inner
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_inner_1d_dot(dtype):
    # 1-D x 1-D is the scalar dot product and returns a 0-D tensor.
    inp = torch.randn(4096, dtype=dtype, device=flag_gems.device)
    other = torch.randn(4096, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.inner(inp, other)

    ref_out = torch.inner(
        utils.to_reference(inp, upcast=True),
        utils.to_reference(other, upcast=True),
    ).to(dtype)

    assert res_out.shape == torch.Size([])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=4096)


@pytest.mark.inner
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_inner_scalar(dtype):
    # A 0-D operand is an outer-product scale: no contraction dimension exists.
    scalar = torch.randn((), dtype=dtype, device=flag_gems.device)
    other = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.inner(scalar, other)

    ref_out = torch.inner(
        utils.to_reference(scalar, upcast=True),
        utils.to_reference(other, upcast=True),
    ).to(dtype)

    assert res_out.shape == other.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inner
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_inner_matvec(dtype):
    # 2-D x 1-D contracts to a 1-D vector (matrix-vector product).
    inp = torch.randn((256, 128), dtype=dtype, device=flag_gems.device)
    other = torch.randn(128, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.inner(inp, other)

    ref_out = torch.inner(
        utils.to_reference(inp, upcast=True),
        utils.to_reference(other, upcast=True),
    ).to(dtype)

    assert res_out.shape == torch.Size([256])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=128)


@pytest.mark.inner
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_inner_empty_contraction(dtype):
    # A zero-length contraction dimension is a well-defined empty reduction
    # whose result is all zeros.
    inp = torch.randn((3, 0), dtype=dtype, device=flag_gems.device)
    other = torch.randn((4, 0), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.inner(inp, other)

    ref_out = torch.inner(
        utils.to_reference(inp),
        utils.to_reference(other),
    )

    assert res_out.shape == torch.Size([3, 4])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.inner
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_inner_2d(dtype):
    # A 2-D x 2-D inner product is the plain matrix product with the second
    # operand transposed.
    inp = torch.randn((128, 64), dtype=dtype, device=flag_gems.device)
    other = torch.randn((256, 64), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.inner(inp, other)

    ref_out = torch.inner(
        utils.to_reference(inp, upcast=True),
        utils.to_reference(other, upcast=True),
    ).to(dtype)

    assert res_out.shape == torch.Size([128, 256])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


@pytest.mark.inner
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_inner_noncontiguous(dtype):
    # The last-dimension strides need not be 1; a transposed operand exercises
    # the strided load path.
    inp = torch.randn((64, 128), dtype=dtype, device=flag_gems.device).t()
    other = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.inner(inp, other)

    ref_out = torch.inner(
        utils.to_reference(inp, upcast=True),
        utils.to_reference(other, upcast=True),
    ).to(dtype)

    assert res_out.shape == torch.Size([128, 32])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


@pytest.mark.inner
def test_inner_dtype_mismatch():
    inp = torch.randn((4, 8), dtype=torch.float32, device=flag_gems.device)
    other = torch.randn((4, 8), dtype=torch.float64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.inner(inp, other)


@pytest.mark.inner
def test_inner_contraction_mismatch():
    inp = torch.randn((4, 8), dtype=torch.float32, device=flag_gems.device)
    other = torch.randn((4, 9), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.inner(inp, other)
