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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.neg
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.neg(ref_inp)
    res_out = flag_gems.neg(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.neg_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_neg_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.neg_(ref_inp)
    res_out = flag_gems.neg_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.fixture
def hygon_neg_module():
    if flag_gems.vendor_name != "hygon":
        pytest.skip("Hygon-specific neg path")
    module = importlib.import_module(flag_gems.neg.__module__)
    assert module.__name__.endswith("_hygon.ops.neg")
    return module


@pytest.mark.neg
@pytest.mark.parametrize(
    "shape", [(), (0,), (1,), (1023,), (1024,), (1025,), (3, 1024), (1024, 1024)]
)
def test_neg_hygon_contiguous(hygon_neg_module, monkeypatch, shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    before = utils.to_reference(inp.clone())
    reference = torch.neg(utils.to_reference(inp))
    assert hygon_neg_module._can_use_fast_path(inp)

    def unexpected_fallback(A):
        pytest.fail("contiguous FP32 neg used generic fallback")

    monkeypatch.setattr(hygon_neg_module, "_generic_neg", unexpected_fallback)
    result = flag_gems.neg(inp)
    utils.gems_assert_equal(result.cpu(), reference)
    utils.gems_assert_equal(inp, before)
    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    assert result.device == inp.device
    if inp.numel():
        assert result.data_ptr() != inp.data_ptr()


@pytest.mark.neg
def test_neg_hygon_special_bits(hygon_neg_module):
    # Include both signs of zeros, subnormals, max finite, infinities, quiet
    # and signaling NaNs with distinct payloads. Construct bits before H2D.
    unsigned = [
        0x00000000,
        0x80000000,
        0x00000001,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x3EAAAAAB,
        0xBEAAAAAB,
        0x3F800000,
        0xBF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0xFFC12345,
        0x7F800001,
        0xFF812345,
    ]
    bits = torch.tensor(
        [x if x < 2**31 else x - 2**32 for x in unsigned], dtype=torch.int32
    )
    inp = bits.view(torch.float32).to(flag_gems.device)
    result = flag_gems.neg(inp)
    expected = bits ^ -2147483648
    torch.testing.assert_close(result.cpu().view(torch.int32), expected, rtol=0, atol=0)
    utils.gems_assert_equal(
        result.cpu(), torch.neg(bits.view(torch.float32)), equal_nan=True
    )
    torch.testing.assert_close(inp.cpu().view(torch.int32), bits, rtol=0, atol=0)


@pytest.mark.neg
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand", "empty", "offset"])
def test_neg_hygon_layouts(hygon_neg_module, monkeypatch, layout):
    inp = torch.randn((8, 16), device=flag_gems.device, dtype=torch.float32)
    if layout == "transpose":
        inp = inp.t()
    elif layout == "slice":
        inp = inp[:, 1::2]
    elif layout == "expand":
        inp = inp[:1].expand(8, 16)
    elif layout == "empty":
        inp = inp[:, :0]
    else:
        inp = inp.reshape(-1)[3:]
    expected_fast = layout in ("empty", "offset")
    assert hygon_neg_module._can_use_fast_path(inp) == expected_fast
    calls = []
    original = hygon_neg_module._generic_neg

    def fallback(A):
        calls.append(A)
        return original(A)

    monkeypatch.setattr(hygon_neg_module, "_generic_neg", fallback)
    reference = torch.neg(utils.to_reference(inp))
    result = flag_gems.neg(inp)
    # Both contiguous and strided FP32 routes now preserve the sign bit.
    assert not calls
    utils.gems_assert_equal(result, reference)


@pytest.mark.neg
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float64, torch.int32, torch.int64]
)
def test_neg_hygon_dtype_fallback(hygon_neg_module, monkeypatch, dtype):
    inp = torch.tensor([-13, -1, 0, 1, 7], device=flag_gems.device, dtype=dtype)
    assert not hygon_neg_module._can_use_fast_path(inp)
    calls = []
    original = hygon_neg_module._generic_neg

    def fallback(A):
        calls.append(A)
        return original(A)

    monkeypatch.setattr(hygon_neg_module, "_generic_neg", fallback)
    result = flag_gems.neg(inp)
    assert len(calls) == 1
    assert calls[0] is inp
    assert result.dtype == dtype
    utils.gems_assert_equal(result, torch.neg(utils.to_reference(inp)))


@pytest.mark.neg
@pytest.mark.parametrize(
    "kind", ["cpu", "grad", "negative_view", "conjugate_view", "subclass", "bool"]
)
def test_neg_hygon_fallback_boundary(hygon_neg_module, monkeypatch, kind):
    inp = torch.randn((8,), device=flag_gems.device)
    if kind == "cpu":
        inp = inp.cpu()
    elif kind == "grad":
        inp.requires_grad_(True)
    elif kind == "negative_view":
        inp = torch._neg_view(inp)
    elif kind == "conjugate_view":
        inp = inp.to(torch.complex64).conj()
    elif kind == "bool":
        inp = inp.to(torch.bool)
    else:

        class NegTensor(torch.Tensor):
            pass

        inp = inp.as_subclass(NegTensor)
    # Check delegation with object identity. Unsupported generic domains are
    # not silently advertised as newly supported by this specialization.
    sentinel = object()
    seen = []

    def fallback(A):
        seen.append(A)
        return sentinel

    monkeypatch.setattr(hygon_neg_module, "_generic_neg", fallback)
    assert not hygon_neg_module._can_use_fast_path(inp)
    assert flag_gems.neg(inp) is sentinel
    assert len(seen) == 1
    assert seen[0] is inp


@pytest.mark.neg
def test_neg_hygon_autograd(hygon_neg_module):
    inp = torch.randn((1025,), device=flag_gems.device, requires_grad=True)
    assert not hygon_neg_module._can_use_fast_path(inp)
    # ATen Autograd owns the derivative; direct flag_gems.neg calls do not
    # promise to build an autograd graph.
    flag_gems.enable()
    result = torch.neg(inp)
    (grad,) = torch.autograd.grad(result, inp, torch.ones_like(result))
    utils.gems_assert_equal(result.cpu(), torch.neg(utils.to_reference(inp.detach())))
    utils.gems_assert_equal(grad, utils.to_reference(-torch.ones_like(inp)))


@pytest.mark.neg
def test_neg_hygon_aten_route(hygon_neg_module, monkeypatch):
    calls = []
    original = hygon_neg_module.neg

    def routed(A):
        calls.append(A)
        return original(A)

    monkeypatch.setattr(flag_gems, "neg", routed)
    inp = torch.randn((1025,), device=flag_gems.device)
    reference = torch.neg(utils.to_reference(inp))
    flag_gems.enable()
    result = torch.neg(inp)
    assert len(calls) == 1
    utils.gems_assert_equal(result.cpu(), reference)
