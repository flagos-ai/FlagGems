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

DEVICE = flag_gems.device

# torch.gradient does not support uint8 and bool inputs; every other dtype is
# promoted like the reference (integral -> default float dtype).
_TEST_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
if flag_gems.runtime.device.support_fp64:
    _TEST_DTYPES.append(torch.float64)

# every grad dim needs at least edge_order + 1 points
GRADIENT_SHAPES = [
    (1024,),
    (16, 32),
    (8, 16, 24),
    (2, 4, 8, 12),
    (15, 33),  # odd sizes
]


def _make_input(shape, dtype, device):
    torch.manual_seed(42)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device) * 3
    return torch.randint(-100, 100, shape, dtype=dtype, device=device)


@pytest.mark.gradient
@pytest.mark.parametrize("shape", GRADIENT_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient_array(shape, dtype, edge_order):
    # exercises aten::gradient.array (no spacing)
    if min(shape) < edge_order + 1:
        pytest.skip("dimension smaller than edge_order + 1")
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.gradient.array(
        ref_inp, dim=list(range(ref_inp.dim())), edge_order=edge_order
    )
    res_out = flag_gems.gradient(inp, edge_order=edge_order)

    assert len(res_out) == ref_inp.dim()
    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("shape", GRADIENT_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("spacing", [None, 1.0, 2.5])
def test_gradient_scalarint(shape, dtype, spacing):
    # exercises aten::gradient.scalarint (Scalar? spacing, int? dim)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)
    dim = ref_inp.dim() - 1

    ref_out = torch.ops.aten.gradient.scalarint(ref_inp, spacing=spacing, dim=dim)
    if spacing is None:
        res_out = flag_gems.gradient(inp, dim=dim)
    else:
        res_out = flag_gems.gradient(inp, spacing=spacing, dim=dim)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("shape", GRADIENT_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_gradient_scalararray(shape, dtype):
    # exercises aten::gradient.scalararray (Scalar spacing + int[] dim)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)
    dims = list(range(ref_inp.dim()))

    ref_out = torch.ops.aten.gradient.scalararray(ref_inp, spacing=2.0, dim=dims)
    res_out = flag_gems.gradient(inp, spacing=2.0, dim=dims)

    assert len(res_out) == len(dims)
    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_scalarrayint(dtype):
    # exercises aten::gradient.scalarrayint (Scalar[] spacing, dim=None)
    shape = (4, 8, 12)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.gradient.scalarrayint(ref_inp, spacing=[1.0, 2.0, 0.5])
    res_out = flag_gems.gradient(inp, spacing=[1.0, 2.0, 0.5])

    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_gradient_tensorarray(dtype):
    # exercises aten::gradient.tensorarray / tensorarrayint (1-D coordinate
    # spacing tensors, uniform and non-uniform, both edge orders)
    shape = (16, 32)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)
    dim = 1

    xs = torch.linspace(0, 1, shape[dim], dtype=dtype, device=DEVICE)
    ref_out = torch.ops.aten.gradient.tensorarray(
        ref_inp, spacing=[utils.to_reference(xs)], dim=[dim], edge_order=2
    )
    res_out = flag_gems.gradient(inp, spacing=[xs], dim=[dim], edge_order=2)
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)

    # non-uniform coordinates
    xs_nu = torch.cumsum(torch.rand(shape[dim], dtype=dtype, device=DEVICE) + 0.5, 0)
    xs_nu = xs_nu - xs_nu[0]
    ref_out = torch.ops.aten.gradient.tensorarray(
        ref_inp, spacing=[utils.to_reference(xs_nu)], dim=[dim], edge_order=1
    )
    res_out = flag_gems.gradient(inp, spacing=[xs_nu], dim=[dim], edge_order=1)
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)

    ref_out = torch.ops.aten.gradient.tensorarray(
        ref_inp, spacing=[utils.to_reference(xs_nu)], dim=[dim], edge_order=2
    )
    res_out = flag_gems.gradient(inp, spacing=[xs_nu], dim=[dim], edge_order=2)
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)


@pytest.mark.gradient
def test_gradient_tensorarrayint_promotion():
    # tensor spacing promotes the output dtype like the reference:
    # promote_types(self, coords)
    x32 = torch.randn(4, 16, dtype=torch.float32, device=DEVICE)
    xs64 = torch.linspace(0, 1, 16, dtype=torch.float64, device=DEVICE)
    res = flag_gems.gradient(x32, spacing=[xs64], dim=[1])
    assert res[0].dtype == torch.result_type(x32, xs64)

    xs16 = torch.linspace(0, 1, 16, dtype=torch.float16, device=DEVICE)
    res = flag_gems.gradient(x32, spacing=[xs16], dim=[1])
    assert res[0].dtype == torch.result_type(x32, xs16)


@pytest.mark.gradient
def test_gradient_output_order_and_negative_dim():
    # outputs follow the order of `dim`, negative dims are wrapped
    inp = _make_input((4, 8, 12), torch.float32, DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.gradient.array(ref_inp, dim=[2, 0])
    res_out = flag_gems.gradient(inp, dim=[2, 0])
    utils.gems_assert_close(res_out[0], ref_out[0], torch.float32)
    utils.gems_assert_close(res_out[1], ref_out[1], torch.float32)

    ref_out = torch.ops.aten.gradient.array(ref_inp, dim=[-1])
    res_out = flag_gems.gradient(inp, dim=[-1])
    utils.gems_assert_close(res_out[0], ref_out[0], torch.float32)


@pytest.mark.gradient
def test_gradient_non_contiguous():
    inp = _make_input((8, 16, 12), torch.float32, DEVICE)
    inp_t = inp.transpose(1, 2)
    ref = torch.ops.aten.gradient.array(utils.to_reference(inp_t.contiguous()), dim=[1])
    res = flag_gems.gradient(inp_t, dim=[1])
    utils.gems_assert_close(res[0], ref[0], torch.float32)


@pytest.mark.gradient
def test_gradient_int_promotion():
    # integral inputs are promoted by true division with a Scalar
    inp = torch.arange(48, device=DEVICE).reshape(4, 12)
    res = flag_gems.gradient(inp, dim=[1])
    ref = torch.ops.aten.gradient.array(utils.to_reference(inp), dim=[1])
    assert res[0].dtype == ref[0].dtype == torch.result_type(inp, 1.0)
    utils.gems_assert_close(res[0], ref[0], torch.float32)


@pytest.mark.gradient
def test_gradient_quadratic_exactness():
    # edge_order=2 is exact on quadratics: y = t**2 with unit spacing
    t = torch.arange(64, dtype=torch.float64, device=DEVICE)
    y = t**2
    res = flag_gems.gradient(y, spacing=1.0, dim=0, edge_order=2)[0]
    ref = torch.ops.aten.gradient.scalarint(
        utils.to_reference(y), spacing=1.0, dim=0, edge_order=2
    )[0]
    utils.gems_assert_close(res, ref, torch.float64)
    assert torch.allclose(res, 2 * t, rtol=1e-12, atol=1e-12)


@pytest.mark.gradient
def test_gradient_error_paths():
    inp = _make_input((4, 8, 12), torch.float32, DEVICE)
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, dim=[1, 1])  # duplicate dim
    with pytest.raises(IndexError):
        flag_gems.gradient(inp, dim=[3])  # out of range
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, edge_order=3)  # unsupported edge_order
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp.byte(), dim=[0])  # uint8 rejected
    # eo=2 requires every grad dim >= 3
    with pytest.raises(RuntimeError):
        flag_gems.gradient(
            _make_input((2, 8), torch.float32, DEVICE), dim=[0], edge_order=2
        )
