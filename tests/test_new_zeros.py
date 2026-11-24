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

"""Correctness tests for ``aten::new_zeros``.

``new_zeros`` is a factory: it allocates a brand new zero-filled buffer of the
requested size and never reads or mutates ``self``, which only supplies the
default dtype / device / layout / pinned metadata. Every comparison against the
native result is therefore exact (the only payload is the zero bit pattern) and
there is no gradient to check.

Native behaviour measured on the CUDA/NVIDIA backend (torch 2.8):
  * the size argument accepts a ``torch.Size``, a list/tuple (bool extents
    coerce to 0/1) and a 1-D integer tensor; a non-sequence raises, and a
    non-tensor ``self`` fails the schema check.
  * ``new_zeros.out`` returns the caller's buffer, resizes a buffer whose shape
    differs and writes through a strided view.
  * strided (dense), sparse COO and sparse CSR/CSC layouts are all accepted;
    the sparse factories are metadata-only constructions (COO holds zero
    stored entries, the compressed layouts materialise an all-zero
    ``crow_indices`` / ``ccol_indices`` pointer buffer).
  * rank-0 and zero-extent sizes are valid and produce exactly those shapes.
  * an empty self still supplies its dtype/device defaults.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import TO_CPU

gems_device = flag_gems.device

# dtype coverage: every family the factory must be able to inherit and override
# (the buffer is a pure zero fill, so every dtype works). Built from the shared
# constants plus the remaining families; the capability gates inside the
# constants drop the dtypes this backend does not declare.
_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
DTYPES = list(dict.fromkeys(_DTYPES))
if flag_gems.runtime.device.support_bf16:
    DTYPES.append(torch.bfloat16)
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)
if flag_gems.runtime.device.support_int64:
    DTYPES.append(torch.int64)

# Sizes: rank 0 through 3 plus the zero-extent and single-element degenerate
# rows. The largest (16, 128, 64, 60) block keeps the fill bandwidth-relevant
# while staying allocation-friendly.
SIZES = [
    (),  # rank-0
    (0,),  # zero extent
    (1,),  # single element
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (16, 7, 57),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
]

# Source shapes for the inherited-metadata rows; the source never contributes
# data, only dtype/device/layout defaults.
SOURCE_SHAPES = [(), (1,), (4, 5)]

SPARSE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc]


def _ref_new_zeros(src, size, **kwargs):
    """Native oracle, moved to the reference device in quick-cpu mode.

    gems_assert_equal asserts the reference is already on CPU when the suite
    runs with --ref=cpu, so the oracle must be built (or moved) there.
    """
    expected = torch.ops.aten.new_zeros.default(src, size, **kwargs)
    return expected.to("cpu") if TO_CPU else expected


def _assert_zero(result, expected_shape, expected_dtype):
    assert tuple(result.shape) == tuple(
        expected_shape
    ), f"shape {tuple(result.shape)} != {tuple(expected_shape)}"
    assert result.dtype == expected_dtype, f"dtype {result.dtype} != {expected_dtype}"
    if result.layout == torch.sparse_coo:
        # COO factories are metadata-only; the stored payload is empty.
        assert result._nnz() == 0
        return
    if result.layout in (torch.sparse_csr, torch.sparse_csc):
        # The compressed layouts materialise a pointer buffer that must be
        # zeroed exactly like the native factory.
        aux = (
            result.crow_indices()
            if result.layout == torch.sparse_csr
            else result.ccol_indices()
        )
        assert aux.numel() > 0 and int(aux.abs().sum()) == 0
        return
    assert result.numel() == 0 or bool(torch.all(result == 0))


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_new_zeros(size, dtype):
    """Size/dtype grid: the result carries the requested shape and every bit is 0."""
    src = torch.empty(SOURCE_SHAPES[1], dtype=dtype, device=gems_device)
    expected = _ref_new_zeros(src, list(size))

    result = flag_gems.new_zeros(src, list(size))

    _assert_zero(result, size, dtype)
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", SIZES)
def test_new_zeros_inherits_metadata(size):
    """Without overrides the result inherits self's dtype/device/layout."""
    src = torch.empty((2, 3), dtype=torch.float32, device=gems_device)
    expected = _ref_new_zeros(src, list(size))

    result = flag_gems.new_zeros(src, list(size))

    assert result.dtype == src.dtype
    assert result.device == src.device
    assert result.layout == src.layout
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros
@pytest.mark.parametrize("source_shape", SOURCE_SHAPES)
def test_new_zeros_source_is_not_read_or_mutated(source_shape):
    """The source only supplies defaults: nan payloads must not leak into the result."""
    src = torch.full(
        source_shape, float("nan"), dtype=torch.float32, device=gems_device
    )
    before = src.clone()

    result = flag_gems.new_zeros(src, [2, 3])

    expected = torch.zeros(2, 3, device=gems_device)
    utils.gems_assert_equal(result, expected.to("cpu") if TO_CPU else expected)
    before_ref = before.to("cpu") if TO_CPU else before
    utils.gems_assert_equal(src, before_ref, equal_nan=True)


@pytest.mark.new_zeros
@pytest.mark.parametrize("dtype", DTYPES)
def test_new_zeros_dtype_override(dtype):
    """An explicit dtype overrides the inherited one; values stay zero."""
    src = torch.empty((2,), dtype=torch.float32, device=gems_device)
    expected = _ref_new_zeros(src, [4, 2], dtype=dtype)

    result = flag_gems.new_zeros(src, [4, 2], dtype=dtype)

    _assert_zero(result, (4, 2), dtype)
    assert result.device == src.device
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", [(), (0,), (1,), (2, 3), (16, 128, 64, 60)])
def test_new_zeros_explicit_device(size):
    """An explicit device override keeps the result zero-filled on that device."""
    src = torch.empty((2, 3), dtype=torch.float32, device=gems_device)
    expected = _ref_new_zeros(src, list(size), device=gems_device)

    result = flag_gems.new_zeros(src, list(size), device=gems_device)

    assert result.device.type == torch.device(gems_device).type
    _assert_zero(result, size, src.dtype)
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros
@pytest.mark.parametrize("layout", SPARSE_LAYOUTS)
def test_new_zeros_sparse_layout(layout):
    """Sparse factories are accepted; the structural buffers must be zeroed."""
    src = torch.empty((2, 3), dtype=torch.float32, device=gems_device)
    size = [4, 5]

    expected = _ref_new_zeros(src, size, layout=layout)
    result = flag_gems.new_zeros(src, size, layout=layout)

    assert result.layout == layout
    _assert_zero(result, size, src.dtype)
    # The COO index/value buffers and the compressed pointer buffers must match
    # the native construction exactly.
    if layout == torch.sparse_coo:
        (
            utils.gems_assert_equal(result.indices(), expected.indices())
            if result._nnz()
            else None
        )
    else:
        aux_res = (
            result.crow_indices()
            if layout == torch.sparse_csr
            else result.ccol_indices()
        )
        aux_ref = (
            expected.crow_indices()
            if layout == torch.sparse_csr
            else expected.ccol_indices()
        )
        utils.gems_assert_equal(aux_res, aux_ref)


@pytest.mark.new_zeros
@pytest.mark.parametrize("size", [(), (0,), (1,), (2, 3)])
def test_new_zeros_size_argument_forms(size):
    """torch.Size, list, tuple and 1-D integer tensor size forms all work."""
    src = torch.empty((2, 3), dtype=torch.float32, device=gems_device)
    expected = _ref_new_zeros(src, list(size))

    for size_arg in (torch.Size(size), list(size), tuple(size), torch.tensor(size)):
        result = flag_gems.new_zeros(src, size_arg)
        _assert_zero(result, size, src.dtype)
        utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros
@pytest.mark.parametrize("extent", [True, False])
def test_new_zeros_bool_extent(extent):
    """A bool extent coerces to 0/1 exactly like the native schema."""
    src = torch.empty((2,), dtype=torch.float32, device=gems_device)
    expected = _ref_new_zeros(src, [extent, 2])

    result = flag_gems.new_zeros(src, [extent, 2])

    _assert_zero(result, (int(extent), 2), src.dtype)
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros_out
@pytest.mark.parametrize("size", [(), (0,), (2, 3), (1024, 1024)])
def test_new_zeros_out(size):
    """The out= overload returns and writes the caller's buffer."""
    src = torch.empty((2,), dtype=torch.float32, device=gems_device)
    expected = _ref_new_zeros(src, list(size))

    buf = torch.full(size, 7.0, dtype=torch.float32, device=gems_device)
    result = flag_gems.new_zeros(src, list(size), out=buf)

    assert result is buf
    _assert_zero(result, size, torch.float32)
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros_out
def test_new_zeros_out_resizes_buffer():
    """A too-small out buffer is resized, matching the native out overload."""
    src = torch.empty((2,), dtype=torch.float32, device=gems_device)
    expected = _ref_new_zeros(src, [3, 4])

    buf = torch.full((2,), 7.0, dtype=torch.float32, device=gems_device)
    result = flag_gems.new_zeros(src, [3, 4], out=buf)

    assert result is buf
    _assert_zero(result, (3, 4), torch.float32)
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros_out
def test_new_zeros_out_writes_through_strided_buffer():
    """Only the strided view is written; the parent's padding stays untouched."""
    src = torch.empty((2,), dtype=torch.float32, device=gems_device)
    parent = torch.full((4, 6), 7.0, dtype=torch.float32, device=gems_device)
    buf = parent[:, ::2]

    result = flag_gems.new_zeros(src, [4, 3], out=buf)

    assert result is buf
    _assert_zero(result, (4, 3), torch.float32)
    # The unwritten odd columns keep their sentinel value.
    expected_pad = torch.full((4, 3), 7.0, device=gems_device)
    utils.gems_assert_equal(
        parent[:, 1::2], expected_pad.to("cpu") if TO_CPU else expected_pad
    )


@pytest.mark.new_zeros
def test_new_zeros_empty_source_still_supplies_defaults():
    """A zero-extent self contributes its dtype/device defaults."""
    src = torch.empty(0, dtype=torch.int32, device=gems_device)
    expected = _ref_new_zeros(src, [2, 3])

    result = flag_gems.new_zeros(src, [2, 3])

    assert result.dtype == torch.int32
    _assert_zero(result, (2, 3), torch.int32)
    utils.gems_assert_equal(result, expected)


@pytest.mark.new_zeros
def test_new_zeros_rejects_non_tensor_self():
    """A non-tensor self fails the schema check."""
    with pytest.raises((RuntimeError, TypeError, AttributeError)):
        flag_gems.new_zeros([1, 2, 3], (2,))


@pytest.mark.new_zeros
def test_new_zeros_rejects_non_sequence_size():
    """A non-sequence size raises instead of being silently coerced."""
    src = torch.empty((2,), dtype=torch.float32, device=gems_device)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.new_zeros(src, 3.14)
