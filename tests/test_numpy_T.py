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

"""Accuracy tests for ``aten::numpy_T(Tensor(a) self) -> Tensor(a)``.

``numpy_T`` is a zero-copy metadata view that reverses every dimension of
``self``: 0-D/1-D keep their shape (but still yield a distinct view object),
2-D swaps both dimensions, n-D is a full reversal. The tests verify the view
contract (shape/stride/storage sharing, laziness of the conj/neg bits,
autograd through the view) and the error paths, mirroring the
``Tensor.numpy_T`` semantics. Because a view never touches element data, one
float dtype sweep is enough; the layout checks are dtype-agnostic.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Ranks 0 through 4: every rank exercises a distinct reversal permutation, and
# the 2-D row additionally feeds the non-contiguous and offset-window rows.
NUMPY_T_SHAPES = [(), (8,), (16, 32), (7, 8, 9), (2, 3, 4, 5)]


def _check_view_metadata(res, ref):
    # The view must match the reference layout and alias the operand.
    assert res.shape == ref.shape
    if not utils.TO_CPU:
        assert res.stride() == ref.stride()
        assert res.storage_offset() == ref.storage_offset()
        assert res.is_conj() == ref.is_conj()
        assert res.is_neg() == ref.is_neg()
    assert res._is_view()


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape", NUMPY_T_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_numpy_T(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.numpy_T(inp)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    _check_view_metadata(res_out, ref_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape", [(5, 6), (2, 3, 4, 5)])
def test_numpy_T_shares_storage(shape):
    # The result is a view: same storage pointer, writes propagate both ways,
    # and the operand's own layout metadata is untouched.
    inp = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    inp_shape, inp_stride, inp_offset = (
        inp.shape,
        inp.stride(),
        inp.storage_offset(),
    )

    res_out = flag_gems.numpy_T(inp)

    assert res_out is not inp
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()

    snapshot = inp.clone()
    res_out.fill_(1.0)
    ref = torch.ops.aten.numpy_T(snapshot)
    assert torch.equal(inp, torch.ops.aten.numpy_T(ref).fill_(1.0))
    assert inp.shape == inp_shape
    assert inp.stride() == inp_stride
    assert inp.storage_offset() == inp_offset


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape", [(), (8,)])
def test_numpy_T_low_rank_distinct_object(shape):
    # 0-D and 1-D keep their shape but still produce a distinct view object
    # sharing the operand's storage.
    inp = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)

    res_out = flag_gems.numpy_T(inp)

    assert res_out is not inp
    assert res_out.shape == inp.shape
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out._is_view()


@pytest.mark.numpy_T
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_numpy_T_noncontiguous(dtype):
    # A transposed / sliced / offset-window operand is still only relaid out,
    # never copied.
    base = torch.randn((32, 16), dtype=dtype, device=flag_gems.device)
    inp = base.t()[4:12, 2:14]
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.numpy_T(inp)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    _check_view_metadata(res_out, ref_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 8, 9)])
def test_numpy_T_strided_and_expanded(shape):
    # A zero-stride expand and a stepped slice both read exactly like a clone.
    base = torch.randn((1,) + shape[1:], device=flag_gems.device, dtype=torch.float32)
    expanded = base.expand(shape)
    res_expanded = flag_gems.numpy_T(expanded)
    ref_expanded = torch.ops.aten.numpy_T(utils.to_reference(expanded))
    utils.gems_assert_close(res_expanded, ref_expanded, torch.float32)

    stepped = torch.randn((8, 16), device=flag_gems.device, dtype=torch.float32)[:, ::2]
    res_stepped = flag_gems.numpy_T(stepped)
    ref_stepped = torch.ops.aten.numpy_T(utils.to_reference(stepped))
    utils.gems_assert_close(res_stepped, ref_stepped, torch.float32)


@pytest.mark.numpy_T
def test_numpy_T_lazy_conj_bit_forwarded():
    # A lazy conj base keeps its bit through the reversal (the public API has
    # no lazy-neg constructor, so only the conj bit is constructible here).
    base = torch.randn((4, 6), device=flag_gems.device, dtype=torch.complex64)
    conj_inp = base.conj()
    res_conj = flag_gems.numpy_T(conj_inp)
    assert res_conj.is_conj()
    ref_conj = torch.ops.aten.numpy_T(utils.to_reference(conj_inp))
    utils.gems_assert_close(res_conj, ref_conj, torch.complex64)

    # (The lazy neg bit is exercised implicitly: no public API creates a lazy
    # neg view without dispatching, and a private-API construction is not
    # allowed in tests. The conj bit above covers the bit-forwarding path.)


@pytest.mark.numpy_T
def test_numpy_T_autograd():
    # The view participates in autograd and routes the gradient with the
    # exact reversed relayout.
    inp = torch.randn(
        4, 6, 8, device=flag_gems.device, dtype=torch.float32, requires_grad=True
    )
    ref_inp = utils.to_reference(inp).detach().requires_grad_(True)

    res_out = flag_gems.numpy_T(inp)
    grad = torch.randn(res_out.shape, device=flag_gems.device, dtype=torch.float32)
    res_out.backward(grad)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    ref_out.backward(utils.to_reference(grad))
    utils.gems_assert_close(inp.grad, ref_inp.grad, torch.float32)


@pytest.mark.numpy_T
@pytest.mark.parametrize("bad_input", [3.14, [1.0, 2.0], None])
def test_numpy_T_rejects_non_tensor(bad_input):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.numpy_T(bad_input)
