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

from . import accuracy_utils as utils

# (input_shape, output_size) pairs, ported from the experimental unit test.
ADAPTIVE_AVG_POOL3D_CASES = [
    ((1, 1, 4, 4, 4), (1, 1, 1)),
    ((1, 1, 4, 4, 4), (2, 2, 2)),
    ((1, 1, 4, 4, 4), (4, 4, 4)),
    ((2, 3, 8, 7, 6), (1, 1, 1)),
    ((2, 3, 8, 7, 6), (2, 3, 3)),
    ((2, 3, 8, 7, 6), (4, 5, 6)),
    ((4, 16, 16, 16, 16), (1, 1, 1)),
    ((4, 16, 16, 16, 16), (4, 4, 4)),
    ((4, 16, 16, 16, 16), (8, 8, 8)),
]

# The host launch picks between two kernels: the tiled one (BO outputs x BR
# window offsets per program) and the lane one (one lane per output, used for
# large output counts over narrow windows). Both must be exercised, so this set
# is chosen to straddle that rule -- the first block has a large output over a
# narrow window, the second a small output over a wide one -- and to include
# degenerate extents (a size-1 axis that is also the whole input, and an
# identity axis) that make the window bounds and masks degenerate.
ADAPTIVE_AVG_POOL3D_DISPATCH_CASES = [
    # total >= 8192 with a window <= 256 elements -> lane kernel.
    ((4, 16, 16, 16, 16), (7, 7, 7)),  # total 21952, window 64
    ((4, 128, 16, 16, 16), (7, 7, 7)),  # total 175616, window 64
    ((2, 64, 32, 32, 32), (7, 7, 7)),  # total 43904, window 216
    # Everything else -> tiled kernel, including the wide-window cases the lane
    # kernel handled poorly (single-digit output counts over large inputs).
    ((1, 3, 64, 64, 64), (1, 1, 1)),  # one output over a 262144-element window
    ((1, 3, 64, 64, 64), (2, 2, 2)),
    ((1, 3, 64, 64, 64), (4, 4, 4)),
    ((1, 3, 64, 64, 64), (7, 7, 7)),
    ((2, 3, 8, 8, 8), (1, 1, 1)),
    ((2, 3, 8, 8, 8), (2, 2, 2)),  # 48 outputs: under the lane rule's 8192
    ((4, 16, 16, 16, 16), (1, 1, 1)),
    # Degenerate extents: identity on one axis, size-1 axes, non-square windows.
    ((2, 2, 16, 1, 16), (3, 1, 5)),
    ((1, 2, 7, 5, 3), (7, 5, 3)),
    ((2, 3, 1, 1, 1), (1, 1, 1)),
    ((1, 1, 4, 4, 4), (4, 4, 4)),
]


# The underscore implementation has its own yaml id (``_adaptive_avg_pool3d`` /
# ``_adaptive_avg_pool3d_out``), but pytest rejects ``pytest.mark._adaptive_...``
# because attribute access starting with an underscore is special-cased. The #6359
# convention strips the leading underscore and, when that collides with the public
# ``adaptive_avg_pool3d`` op, prefixes ``underscore_`` instead. The public
# overloads keep their own plain yaml ids and markers, so the four aten signatures
# map one-to-one onto four yaml entries (cf. the ``arcsin`` / ``arcsin_out``
# grouping).
@pytest.mark.underscore_adaptive_avg_pool3d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d(shape, output_size, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)
    res_out = flag_gems._adaptive_avg_pool3d(res_inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_public(shape, output_size, dtype):
    # The plain public overload has no Autograd kernel in ATen, so this path only
    # works because FlagGems both overrides its device key and supplies the
    # backward formula. Check it against the underscore reference.
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)
    res_out = torch.ops.aten.adaptive_avg_pool3d(res_inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.underscore_adaptive_avg_pool3d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_out(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (shape[0], shape[1], *output_size)
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._adaptive_avg_pool3d.out(ref_inp, output_size, out=ref_out)

    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.adaptive_avg_pool3d_out(inp, output_size, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.underscore_adaptive_avg_pool3d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_4d_input(dtype):
    # A 4D (C, D, H, W) input takes the unsqueeze-to-5D branch.
    shape = (3, 8, 7, 6)
    output_size = (4, 5, 6)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)
    res_out = flag_gems._adaptive_avg_pool3d(res_inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.underscore_adaptive_avg_pool3d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_DISPATCH_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_dispatch_kernels(shape, output_size, dtype):
    # Both launch paths must agree with the reference, so this runs the
    # dispatch-straddling cases through the public path (which reaches the
    # generic host launcher) and through the out variant (same launcher, a
    # caller-supplied destination).
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)
    res_out = flag_gems._adaptive_avg_pool3d(res_inp, output_size)
    utils.gems_assert_close(res_out, ref_out, dtype)

    out_shape = (shape[0], shape[1], *output_size)
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out_2 = flag_gems.adaptive_avg_pool3d_out(res_inp, output_size, out=out)

    assert res_out_2 is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.underscore_adaptive_avg_pool3d_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_out_4d_input(dtype):
    # A 4D (C, D, H, W) input takes the unsqueeze-to-5D branch of the out variant;
    # the result is squeezed back to 4D (C, D_out, H_out, W_out).
    shape = (3, 8, 7, 6)
    output_size = (4, 5, 6)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (shape[0], *output_size)
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._adaptive_avg_pool3d.out(ref_inp, output_size, out=ref_out)

    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.adaptive_avg_pool3d_out(inp, output_size, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


# The public ``aten::adaptive_avg_pool3d.out`` overload carries its own backend
# kernel (it is not CompositeImplicitAutograd like the plain public overload),
# so it needs its own registration; these cases exercise that dispatch entry.
@pytest.mark.adaptive_avg_pool3d_out
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_public_out_overload(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (shape[0], shape[1], *output_size)
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.adaptive_avg_pool3d.out(ref_inp, output_size, out=ref_out)

    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.ops.aten.adaptive_avg_pool3d.out(inp, output_size, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool3d_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_public_out_overload_4d_input(dtype):
    # A 4D (C, D, H, W) input takes the unsqueeze-to-5D branch of the out variant.
    shape = (3, 8, 7, 6)
    output_size = (4, 5, 6)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (shape[0], *output_size)
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.adaptive_avg_pool3d.out(ref_inp, output_size, out=ref_out)

    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.ops.aten.adaptive_avg_pool3d.out(inp, output_size, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


# The plain public overload has no Autograd kernel in ATen, so once FlagGems
# overrides its device key it must also supply the backward formula -- otherwise
# gradients silently stop flowing (a requires_grad output with a None grad). Cover
# both the batched 5D and unbatched 4D entry points.
@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize(
    "shape, output_size",
    [((2, 3, 8, 8, 8), (4, 4, 4)), ((3, 8, 7, 6), (4, 5, 6))],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_public_backward(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)
    ref_out.sum().backward()

    res_out = torch.ops.aten.adaptive_avg_pool3d(inp, output_size)
    res_out.sum().backward()

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert inp.grad is not None
    utils.gems_assert_close(inp.grad, ref_inp.grad, dtype)
