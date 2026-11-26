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

"""Accuracy tests for ``aten::swapdims_(Tensor(a!) self, int, int) -> Tensor(a!)``.

``swapdims_`` is the in-place alias of ``transpose_``: it only rewrites the
operand's size/stride metadata. The tests therefore compare the mutated
operand's shape/stride/storage offset against ATen, check that ``self`` is
returned, that the version counter is bumped, that the storage is untouched
and shared with pre-existing aliases, and cover negative dims,
non-contiguous operands, scalar/empty tensors and the ATen error messages.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Dim pairs: first/last, middle pairs, negative dims, and the identity swap.
SWAPDIMS_DIM_PAIRS = [
    (0, 1),
    (0, -1),
    (-1, -2),
    (1, 2),
    (1, -1),
    (0, 0),
]

# 0D/1D exercise the degenerate (no-op) path, 2D/3D/4D the real swaps, the 5D
# entry covers deep negatives, and the larger entries catch stride arithmetic
# that small shapes would hide.
SWAPDIMS_SHAPES = [
    (),
    (5,),
    (2, 3),
    (128, 256),
    (2, 3, 4),
    (16, 128, 64),
    (2, 3, 4, 5),
    (8, 16, 32, 64),
    (16, 7, 57, 32, 29),
]


def _valid_axis_cases():
    """Yield only (shape, dim_pair) combos where the pair is in range.

    0D/1D tensors accept only dims -1 and 0 (both mean the only dim); for
    rank >= 2 every pair in the list is in range.  Filtering invalid
    combinations here instead of skipping inside the test keeps the
    parametrization free of no-op cases.
    """
    for shape in SWAPDIMS_SHAPES:
        ndim = len(shape)
        for dim_pair in SWAPDIMS_DIM_PAIRS:
            if ndim <= 1:
                if dim_pair[0] not in (0, -1) or dim_pair[1] not in (0, -1):
                    continue
            elif not all(-ndim <= d < ndim for d in dim_pair):
                continue
            yield (shape, dim_pair)


@pytest.mark.swapdims_
@pytest.mark.parametrize("shape, dim_pair", list(_valid_axis_cases()))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapdims_(shape, dim_pair, dtype):
    dim0, dim1 = dim_pair
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    ref_out = ref_inp.swapdims_(dim0, dim1)

    # The operand is mutated in place, ``self`` is returned, and the mutated
    # layout and values match ATen exactly.
    assert res_out is inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    utils.gems_assert_close(inp, ref_inp, dtype)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.swapdims_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapdims__negative_dims(dtype):
    # Full reversal via negative dims: -1/-2 and -3/-1 must normalize exactly
    # like ATen on every orbit of a 4D tensor.
    for dim0, dim1 in [(-1, -2), (-1, -4), (-3, -1), (2, -4), (-4, 0)]:
        inp = torch.randn((3, 4, 5, 6), dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp.clone())

        res_out = flag_gems.swapdims_(inp, dim0, dim1)
        ref_inp.swapdims_(dim0, dim1)

        assert res_out is inp
        assert inp.shape == ref_inp.shape
        assert inp.stride() == ref_inp.stride()
        utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.swapdims_
@pytest.mark.parametrize(
    "shape,dim_pair",
    [
        ((2, 3, 4), (0, 2)),
        ((8, 16, 32, 64), (0, -1)),
        ((128, 256), (1, 0)),
        ((3, 4, 5), (1, 2)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapdims__non_contiguous(shape, dim_pair, dtype):
    # Slicing a contiguous base yields non-contiguous strides; the in-place
    # swap must still only permute metadata. The slice is built on both
    # devices before to_reference so both operands share one layout.
    dim0, dim1 = dim_pair
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]

    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    ref_inp.swapdims_(dim0, dim1)

    assert res_out is inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.swapdims_
@pytest.mark.parametrize(
    "shape,dim_pair",
    [
        # 0-D has no dims to swap: every in-range pair is a no-op.
        ((), (0, 0)),
        ((), (-1, 0)),
        ((), (0, -1)),
        # 1-D: only dim 0/-1 exist, so any in-range pair is a no-op.
        ((7,), (0, 0)),
        ((7,), (-1, 0)),
        # Empty tensors keep their (possibly degenerate) strides.
        ((0,), (0, 0)),
        ((5, 0, 3), (0, 1)),
        ((5, 0, 3), (0, -1)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapdims__scalar_and_empty(shape, dim_pair, dtype):
    dim0, dim1 = dim_pair
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    ref_inp.swapdims_(dim0, dim1)

    assert res_out is inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    if ref_inp.numel() > 0:
        utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.swapdims_
@pytest.mark.parametrize(
    "shape,dim_pair",
    [
        ((2, 3, 4), (0, 2)),
        ((8, 16, 32, 64), (0, -1)),
        ((128, 256), (1, 0)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapdims__aliasing(shape, dim_pair, dtype):
    # The swap is applied to ``self``'s metadata, so an alias taken *before*
    # the call (a view of the operand) must observe the swapped layout, and
    # no element data may move.
    dim0, dim1 = dim_pair
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    storage_ptr = inp.untyped_storage().data_ptr()
    snapshot = inp.clone()

    res_alias = inp[0]
    ref_alias = ref_inp[0]

    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    ref_inp.swapdims_(dim0, dim1)

    assert res_out is inp
    assert res_alias.shape == ref_alias.shape
    assert res_alias.stride() == ref_alias.stride()
    utils.gems_assert_close(res_alias, ref_alias, dtype)
    assert inp.untyped_storage().data_ptr() == storage_ptr
    # The permuted view still exposes exactly the original elements.
    assert sorted(inp.reshape(-1).tolist()) == sorted(snapshot.reshape(-1).tolist())


@pytest.mark.swapdims_
@pytest.mark.parametrize("shape,dim_pair", [((4, 5), (0, 1)), ((2, 3, 4), (0, 2))])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapdims__version_counter(shape, dim_pair, dtype):
    # An in-place op must bump the version counter exactly like ATen so that
    # autograd's in-place checks stay sound; the identity swap also bumps it.
    dim0, dim1 = dim_pair
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    version_before = inp._version
    ref_version_before = ref_inp._version

    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    ref_inp.swapdims_(dim0, dim1)

    assert res_out is inp
    assert inp._version == version_before + 1
    assert inp._version - version_before == ref_inp._version - ref_version_before


@pytest.mark.swapdims_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapdims__version_counter_identity(dtype):
    # A no-op swap (same dim twice) still bumps the version counter in ATen.
    inp = torch.randn((3, 4, 5), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    version_before = inp._version
    ref_version_before = ref_inp._version

    res_out = flag_gems.swapdims_(inp, 1, 1)
    ref_inp.swapdims_(1, 1)

    assert res_out is inp
    assert ref_inp._version == ref_version_before + 1
    assert inp._version == version_before + 1


@pytest.mark.swapdims_
@pytest.mark.parametrize(
    "shape,dim_pair",
    [
        ((2, 3, 4), (0, 3)),
        ((2, 3, 4), (-4, 0)),
        ((2, 3), (0, 5)),
        ((2, 3), (-3, 0)),
        ((5,), (0, 1)),
        ((5,), (2, 0)),
        ((), (0, 1)),
        ((), (-2, 0)),
    ],
)
def test_swapdims__invalid_dims(shape, dim_pair):
    # The error type and message must match ATen exactly.
    dim0, dim1 = dim_pair
    inp = torch.randn(shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    with pytest.raises(IndexError) as ref_exc_info:
        ref_inp.swapdims_(dim0, dim1)
    with pytest.raises(IndexError) as res_exc_info:
        flag_gems.swapdims_(inp, dim0, dim1)
    assert str(res_exc_info.value) == str(ref_exc_info.value)


@pytest.mark.swapdims_
def test_swapdims__expanded_overlap():
    # Expanded (stride-0) tensors: the swap must still only permute strides
    # and keep the storage, matching ATen. The reference must also be an
    # expanded view -- cloning would materialize a contiguous tensor.
    inp = torch.zeros(1, 3, device=flag_gems.device).expand(4, 3)
    ref_inp = utils.to_reference(torch.zeros(1, 3, device=flag_gems.device)).expand(
        4, 3
    )
    data_ptr = inp.data_ptr()

    res_out = flag_gems.swapdims_(inp, 0, 1)
    ref_inp.swapdims_(0, 1)

    assert res_out is inp
    assert res_out.shape == ref_inp.shape
    assert res_out.stride() == ref_inp.stride()
    assert res_out.data_ptr() == data_ptr
    utils.gems_assert_close(res_out, ref_inp, torch.float32)


@pytest.mark.swapdims_
@pytest.mark.parametrize("shape", [(2, 3, 4), (128, 256)])
def test_swapdims__int_dtypes(shape):
    # swapdims_ is layout-only; verify on an integer dtype too.
    dim0, dim1 = 0, -1
    inp = torch.randint(-100, 100, shape, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    res_out = flag_gems.swapdims_(inp, dim0, dim1)
    ref_out = ref_inp.swapdims_(dim0, dim1)

    assert res_out is inp
    utils.gems_assert_equal(inp, ref_inp)
    utils.gems_assert_equal(res_out, ref_out)
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
