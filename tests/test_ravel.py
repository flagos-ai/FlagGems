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

"""Accuracy tests for ``aten::ravel(Tensor(a) self) -> Tensor(a)``.

``ravel`` flattens a tensor to 1-D. Two paths exist and both are tested:

* VIEW path -- the flattened input is already contiguous after collapsing
  size-1 dimensions; the result aliases the input storage, keeps its
  storage_offset and the lazy neg/conj bits, and no device work runs.
* COPY path -- every other strided layout; a fresh contiguous buffer is
  materialised with the lazy neg/conj bits folded into the stored values.

Backward only rearranges gradients back to the input shape. Comparisons are
exact where the contract is metadata-only and tolerance-based on the copy
path; one float dtype sweep is enough because the layout logic is
dtype-agnostic.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Ranks 0 through 4: rank 0 flattens to a 1-element 1-D tensor, rank 1 is the
# identity-shaped row, and the higher ranks feed the non-contiguous rows.
RAVEL_SHAPES = [(), (8,), (16, 32), (7, 8, 9), (2, 3, 4, 5)]

# Non-contiguous layouts: transposed, offset window, stepped slice,
# zero-stride expand and a channels-last 4-D input. All take the COPY path.
# An offset-only slice stays contiguous (it takes the VIEW path with a
# non-zero offset), so it is dropped from the copy-path grid.
_COPY_LAYOUTS = ["transposed", "strided_last", "expanded"]

# Ranks 1 through 8 for the rank-decomposition path; rank 8 is the kernel's
# documented maximum.
RANK_SHAPES = [
    (256,),
    (16, 32),
    (7, 8, 9),
    (2, 3, 4, 5),
    (2, 3, 2, 3, 2),
    (2, 2, 2, 3, 2, 3),
    (2, 2, 2, 2, 3, 2, 2),
    (2, 2, 2, 2, 2, 3, 2, 2),
]


def _apply_layout(inp, layout):
    if layout == "transposed":
        return inp.transpose(0, 1)
    if layout == "offset_slice":
        return inp[1:]
    if layout == "strided_last":
        return inp[..., ::2]
    if layout == "expanded":
        return inp[:1].expand(4, *inp.shape[1:])
    raise ValueError(f"unknown layout {layout!r}")


@pytest.mark.ravel
@pytest.mark.parametrize("shape", RAVEL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ravel(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.ravel(inp)
    ref_out = torch.ravel(ref_inp)

    # The result is 1-D with the input's element count and unit stride.
    assert res_out.dim() == 1
    assert res_out.numel() == inp.numel()
    assert tuple(res_out.stride()) == (1,)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ravel
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ravel_contiguous_view_path(dtype):
    # The contiguous input takes the VIEW path: same storage, offset kept,
    # no copy.
    base = torch.randn((6, 7), dtype=dtype, device=flag_gems.device)
    inp = base[2:5]  # full-width row window at a non-zero storage offset
    assert inp.is_contiguous()

    res_out = flag_gems.ravel(inp)
    ref_out = torch.ravel(inp)
    # .to("cpu") compacts the view, so the offset metadata must be asserted
    # against the device-side native result while the value comparison uses
    # the moved reference (gems_assert_close requires it on CPU in quick-cpu).
    ref_values = torch.ravel(utils.to_reference(inp))

    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.storage_offset() != 0
    utils.gems_assert_close(res_out, ref_values, dtype)


@pytest.mark.ravel
@pytest.mark.parametrize("layout", _COPY_LAYOUTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ravel_copy_path(layout, dtype):
    # 3-D ranks keep the transpose/step/expand layouts on a genuine multi-axis
    # shape; the expand row uses a smaller first extent so the 4x expansion
    # stays allocation-friendly.
    shape = (4, 6, 8) if layout != "expanded" else (2, 6, 8)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = _apply_layout(inp, layout)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.ravel(inp)
    ref_out = torch.ravel(ref_inp)

    # The copy path yields a fresh buffer: offset 0, no storage sharing.
    assert res_out.is_contiguous()
    assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ravel
@pytest.mark.parametrize("shape", RANK_SHAPES)
def test_ravel_copy_path_ranks(shape):
    # Every rank up to the kernel's 8-dim limit flattens correctly. The first
    # two axes are swapped so the layout is genuinely non-contiguous for every
    # rank (rank 1 has nothing to swap and is covered by the view-path rows).
    inp = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    if len(shape) >= 2:
        inp = inp.transpose(0, 1)
        assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.ravel(inp)
    ref_out = torch.ravel(ref_inp)

    assert res_out.numel() == inp.numel()
    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.ravel
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_ravel_complex(dtype):
    if dtype == torch.complex128 and not flag_gems.runtime.device.support_fp64:
        pytest.skip("complex128 needs fp64 support")
    inp = torch.randn((4, 6), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.ravel(inp)
    ref_out = torch.ravel(ref_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ravel
def test_ravel_lazy_conj_bit_kept_on_view_path():
    # A contiguous conj base takes the view path and keeps the lazy bit.
    base = torch.randn((4, 6), device=flag_gems.device, dtype=torch.complex64)
    conj_inp = base.conj()
    assert conj_inp.is_contiguous()

    res_out = flag_gems.ravel(conj_inp)
    ref_out = torch.ravel(conj_inp)
    # The lazy conj bit lives in view metadata that a .to("cpu") move would
    # materialize away, so the bit is asserted on the device-side native
    # result; the value comparison uses the moved reference.
    ref_values = torch.ravel(utils.to_reference(conj_inp))

    assert res_out.is_conj() == ref_out.is_conj()
    utils.gems_assert_close(res_out, ref_values, torch.complex64)


@pytest.mark.ravel
def test_ravel_neg_materialized_value():
    # A negated (materialized, not lazy) transposed base flattens to the same
    # values the native op produces on the copy path.
    base = torch.randn((4, 6), device=flag_gems.device, dtype=torch.float32)
    neg_inp = base.neg().t()
    ref_inp = utils.to_reference(neg_inp)

    res_out = flag_gems.ravel(neg_inp)
    ref_out = torch.ravel(ref_inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.ravel
def test_ravel_autograd():
    # Backward rearranges gradients back to the input shape and touches no
    # backing storage. The leaf is the 2-D base, so the gradient accumulates
    # on a leaf while the non-contiguous transposed view feeds the copy path.
    # In quick-cpu mode the reference leaf lives on CPU so the backward graph
    # stays on the same device as the reference forward result.
    base = torch.randn(
        (4, 6), device=flag_gems.device, dtype=torch.float32, requires_grad=True
    )
    inp = base.t()
    ref_base = torch.randn(
        (4, 6),
        dtype=torch.float32,
        requires_grad=True,
        device="cpu" if utils.TO_CPU else flag_gems.device,
    )
    ref_inp = ref_base.t()

    res_out = flag_gems.ravel(inp)
    grad = torch.randn(res_out.shape, device=flag_gems.device, dtype=torch.float32)
    res_out.backward(grad)

    ref_out = torch.ravel(ref_inp)
    ref_out.backward(utils.to_reference(grad))
    assert base.grad is not None and ref_base.grad is not None
    utils.gems_assert_close(base.grad, ref_base.grad, torch.float32)


@pytest.mark.ravel
@pytest.mark.parametrize("bad_input", [3.14, [1.0, 2.0], None])
def test_ravel_rejects_non_tensor(bad_input):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ravel(bad_input)


@pytest.mark.ravel
def test_ravel_rejects_sparse_layout():
    idx = torch.tensor([[0, 1], [1, 0]], device=flag_gems.device)
    values = torch.tensor([1.0, 2.0], device=flag_gems.device)
    sparse = torch.sparse_coo_tensor(idx, values, (2, 2), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.ravel(sparse)
