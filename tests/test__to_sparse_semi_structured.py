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

"""Correctness tests for ``aten::_to_sparse_semi_structured``."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Measured native geometry of the target build (each shape in the tables below was
# run through torch.ops.aten._to_sparse_semi_structured and returned the expected
# packed/meta pair):
#   float16 / bfloat16   rows % 32, columns % 32
#   int8                 rows % 16, columns % 64
#   float32              rows % 32, columns % 16
# Every positive shape below is native-verified. A single observation from an
# exploratory run of the *local* build left no isolated command/returncode/stderr
# artifact: fp16 widths that are odd multiples of 16 aborted the interpreter with
# malloc heap corruption. It is recorded as an unresolved native-oracle hazard, not
# as a contract and not as evidence that such shapes are unsupported; _PENDING_GEOMETRY
# names the exact shapes in that class and they are deliberately never executed. No
# other size is claimed unsafe, and no test re-probes a suspicious width.
#
# Backward is exempt on the tested native path (float16, (64, 64), requires_grad):
# the packed result carries grad_fn=<NotImplemented object>, the metadata result has
# no autograd history, and torch.autograd.grad raises "RuntimeError: derivative for
# aten::_to_sparse_semi_structured is not implemented". That is the measured
# behavior of this path rather than a claim about every dtype or vendor. There is no
# reference gradient to compare a candidate against, so the dimension is reported as
# an oracle gap; it is not replaced with a CPU substitute, since no CPU kernel is
# assumed to exist.
SUPPORTED_DTYPES = [torch.float16, torch.float32, torch.int8]
if utils.bf16_is_supported:
    SUPPORTED_DTYPES.insert(1, torch.bfloat16)

_FLOAT_DTYPES = [dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point]

# Half of every slot group is retained: 2:4 for the 16-bit/int8 dtypes, 1:2 for
# float32 (a second retained slot in the float32 pair is rejected natively with
# "does not match 1:2sparsity pattern").
_GROUP = {
    torch.float16: 4,
    torch.bfloat16: 4,
    torch.int8: 4,
    torch.float32: 2,
}

_PATTERN_KINDS = ("fixed", "shifted", "alternating", "blocked", "diagonal", "varied")
_LAYOUT_KINDS = ("transposed", "sliced", "expanded", "dilated")
_PATTERN_SHAPE = (64, 64)


def _assert_pair(res, ref):
    # The operator returns (packed, metadata). Both are exact payloads, so each
    # component uses the zero-tolerance comparison instead of a tolerance policy.
    assert len(res) == len(ref) == 2
    tu.assert_result_equal(res[0], ref[0])
    tu.assert_result_equal(res[1], ref[1])


def _axis(size, device):
    # int32 keeps the metadata/index tensors small; no int64 arange is needed.
    return torch.arange(size, device=device, dtype=torch.int32)


def _offset(kind, rows, cols, dtype, device):
    row = _axis(rows, device).view(rows, 1)
    col = _axis(cols, device).view(1, cols)
    group = _GROUP[dtype]
    block = col // group
    if kind == "fixed":
        return torch.zeros((1, 1), device=device, dtype=torch.int32)
    if kind == "shifted":
        return torch.ones((1, 1), device=device, dtype=torch.int32)
    if kind == "alternating":
        return row % 2
    if kind == "blocked":
        return block % 2
    # 'diagonal' and 'varied' place the retained pair per row/block.
    return (row + block) % 2


def _pattern_mask(kind, rows, cols, dtype, device):
    group = _GROUP[dtype]
    slot = _axis(cols, device).view(1, cols) % group
    start = _offset(kind, rows, cols, dtype, device)
    return (slot >= start) & (slot < start + group // 2)


def _pattern_values(kind, rows, cols, dtype, device):
    row = _axis(rows, device).view(rows, 1)
    col = _axis(cols, device).view(1, cols)
    if kind == "varied":
        # Distinct values inside every slot group, so metadata/compression cannot
        # pass by broadcasting one retained value over its neighbours.
        values = ((row * cols + col) % 99) + 1
    else:
        values = ((row * 3 + col) % 7) + 1
    return values.expand(rows, cols).to(dtype)


def _structured(dtype, values, mask, anchor):
    # A retained slot that holds an exact zero is indistinguishable from a pruned
    # slot for the native pattern count, so retained zeros are replaced by a nonzero
    # anchor. Pruned slots are written as an exact +0, hence a signed-zero payload is
    # deliberately not preserved and no signed-zero retention is claimed.
    device = values.device
    zero = torch.zeros((), dtype=dtype, device=device)
    filled = torch.where(
        values == zero, torch.full((), anchor, dtype=dtype, device=device), values
    )
    return torch.where(mask.expand(values.shape), filled, zero).contiguous()


def _pattern_fixture(kind, dtype, shape):
    rows, cols = shape
    device = flag_gems.device
    values = _pattern_values(kind, rows, cols, dtype, device)
    mask = _pattern_mask(kind, rows, cols, dtype, device)
    return _structured(dtype, values, mask, 1)


def _masked_input(dtype, shape, value_range):
    rows, cols = shape
    values = tu.make_input(dtype, shape, value_range)
    mask = _pattern_mask("fixed", rows, cols, dtype, values.device)
    return _structured(dtype, values, mask, 1)


def _special_input(scenario, dtype, shape):
    # The payload comes from the shared special-value contract (nan only / inf only
    # / mixed) and is sampled over a legal sparse layout; retained zeros are anchored
    # for the same reason as in _structured.
    rows, cols = shape
    device = flag_gems.device
    payload = tu.make_special_input(dtype, scenario)
    row = _axis(rows, device).view(rows, 1)
    col = _axis(cols, device).view(1, cols)
    values = payload[((row + col) % payload.numel()).expand(rows, cols)]
    mask = _pattern_mask("fixed", rows, cols, dtype, device)
    return _structured(dtype, values, mask, 1)


# ---------------------------------------------------------------------------
# Case tables
# ---------------------------------------------------------------------------
# Each dtype keeps its own measured shape set: the alignment units differ (float32
# uses 1:2 groups), so the tables are not shared between dtypes. The scale rows cover
# the minimum representable shape, small/large squares and the two large non-square
# scales from the regular-operator shape reference.
_FP16_SCALES = (
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
    (32, 256),
    (256, 32),
    (64, 512),
    (512, 64),
    (96, 96),
    # Legal beyond the original table: 96 and 64 each already appear as an accepted
    # row or column, and neither is an odd multiple of 16.
    (32, 96),
    (96, 64),
    (128, 256),
    (256, 256),
    (320, 320),
    (384, 128),
    (1024, 32),
    (32, 1024),
    (1024, 1024),
    (2048, 2880),
    (2048, 3840),
)

_INT8_SCALES = (
    (16, 64),
    (16, 128),
    (16, 512),
    (32, 64),
    (64, 64),
    (64, 128),
    (512, 64),
    (64, 512),
    (96, 192),
    (256, 256),
    (512, 512),
    (320, 320),
    (16, 1024),
    (1024, 64),
    (1024, 1024),
    (2048, 2880),
    (2048, 3840),
)

_FP32_SCALES = (
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 16),
    (64, 64),
    (32, 128),
    (256, 16),
    (64, 512),
    (512, 64),
    (96, 96),
    (128, 256),
    (256, 256),
    (320, 320),
    (384, 128),
    (1024, 32),
    (32, 1024),
    (1024, 1024),
    (2048, 2880),
    (2048, 3840),
)

_SHAPES = {
    torch.float16: _FP16_SCALES,
    torch.bfloat16: _FP16_SCALES,
    torch.int8: _INT8_SCALES,
    torch.float32: _FP32_SCALES,
}

# Quick mode keeps the smallest representable shape of every supported dtype.
_QUICK_SHAPES = {
    torch.float16: (32, 32),
    torch.bfloat16: (32, 32),
    torch.int8: (16, 64),
    torch.float32: (32, 16),
}

_EMPTY_SCALES = {
    torch.float16: ((0, 64), (32, 0)),
    torch.bfloat16: ((0, 64), (32, 0)),
    torch.int8: ((0, 64), (16, 0)),
    torch.float32: ((0, 64), (32, 0)),
}

_LAYOUT_BASES = {
    torch.float16: {
        "transposed": (128, 64),
        "sliced": (64, 256),
        "expanded": (1, 256),
        "dilated": (128, 128),
    },
    torch.bfloat16: {
        "transposed": (128, 64),
        "sliced": (64, 256),
        "expanded": (1, 256),
        "dilated": (128, 128),
    },
    torch.int8: {
        "transposed": (128, 64),
        "sliced": (64, 256),
        "expanded": (1, 256),
        "dilated": (128, 128),
    },
    torch.float32: {
        "transposed": (128, 64),
        "sliced": (64, 256),
        "expanded": (1, 256),
        "dilated": (128, 128),
    },
}

# Rank rejection: the operator takes a 2-D dense tensor. Pattern-count rejection: a
# fully-zero row under-fills the packed buffer and a fully-dense row over-fills it,
# checked for the dtype whose slot group the fixture uses. Shape rejection: (48, 64)
# and (80, 64) carry a legal 2:4 payload and fail the native row-alignment check
# before any packing kernel runs.
_INVALID_RANK_SHAPES = (
    (),
    (64,),
    (1, 1, 1),
    (32, 64, 2),
    (2, 32, 64, 2),
    (64, 64, 2, 2, 2),
)
_INVALID_GEOMETRY = ((48, 64), (80, 64))
# Recorded, deliberately not executed: these are the fp16 widths inside the unresolved
# hazard above, so neither acceptance nor rejection can be established from the
# evidence on hand. This is an open native-oracle gap - not claimed unsupported and
# not claimed covered - and it stays in the file so the gap remains visible.
_PENDING_GEOMETRY = ((32, 48), (32, 80))
_INVALID_PATTERN_BASE = (
    (torch.float16, (32, 64)),
    (torch.int8, (32, 64)),
    (torch.float32, (32, 32)),
)


def _shape_rows():
    rows = []
    for dtype in SUPPORTED_DTYPES:
        for shape in tu.selected_cases(_SHAPES[dtype], quick=[_QUICK_SHAPES[dtype]]):
            for value_range in tu.selected_ranges():
                rows.append((dtype, shape, value_range))
    return rows


def _invalid_pattern_rows():
    # bfloat16 is only a legal fixture where the device can construct it.
    base = list(_INVALID_PATTERN_BASE)
    if utils.bf16_is_supported:
        base.insert(1, (torch.bfloat16, (32, 64)))
    return [
        (dtype, shape, mode) for dtype, shape in base for mode in ("pruned", "dense")
    ]


def _unsupported_dtype_rows():
    # Only the types the active device can construct are negative cases; the ones the
    # device cannot build are left out instead of being declared unsupported.
    rows = [
        (torch.int32, (32, 64)),
        (torch.int16, (32, 64)),
        (torch.bool, (32, 64)),
        (torch.uint8, (32, 64)),
    ]
    if utils.int64_is_supported:
        rows.append((torch.int64, (32, 64)))
    if utils.fp64_is_supported:
        rows.append((torch.float64, (32, 64)))
    if utils.fp8_is_supported:
        rows.extend(((torch.float8_e4m3fn, (32, 64)), (torch.float8_e5m2, (32, 64))))
    return rows


SHAPE_ROWS = _shape_rows()
# Supplementary positive families are default-only; quick keeps the main shape/range
# grid and the complete negative set.
EMPTY_ROWS = tu.selected_cases(
    [(dtype, shape) for dtype in SUPPORTED_DTYPES for shape in _EMPTY_SCALES[dtype]],
    quick=[],
)
PATTERN_ROWS = tu.selected_cases(
    [(kind, dtype) for dtype in SUPPORTED_DTYPES for kind in _PATTERN_KINDS], quick=[]
)
LAYOUT_ROWS = tu.selected_cases(
    [(kind, dtype) for dtype in SUPPORTED_DTYPES for kind in _LAYOUT_KINDS], quick=[]
)
SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])
INVALID_PATTERN_ROWS = _invalid_pattern_rows()
UNSUPPORTED_DTYPE_ROWS = _unsupported_dtype_rows()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("dtype,shape,value_range", SHAPE_ROWS)
def test__to_sparse_semi_structured_shape_grid(dtype, shape, value_range):
    inp = _masked_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._to_sparse_semi_structured(inp)
    ref = torch.ops.aten._to_sparse_semi_structured(ref_inp)

    _assert_pair(res, ref)
    assert res[0].device == inp.device
    assert res[1].device == inp.device
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("dtype,shape", EMPTY_ROWS)
def test__to_sparse_semi_structured_empty(dtype, shape):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._to_sparse_semi_structured(inp)
    ref = torch.ops.aten._to_sparse_semi_structured(ref_inp)

    _assert_pair(res, ref)
    assert res[0].device == inp.device
    assert res[1].device == inp.device
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("kind,dtype", PATTERN_ROWS)
def test__to_sparse_semi_structured_pattern(kind, dtype):
    inp = _pattern_fixture(kind, dtype, _PATTERN_SHAPE)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._to_sparse_semi_structured(inp)
    ref = torch.ops.aten._to_sparse_semi_structured(ref_inp)

    _assert_pair(res, ref)
    assert res[0].device == inp.device
    assert res[1].device == inp.device
    tu.assert_result_equal(inp, ref_inp)


def _layout_view(kind, parent):
    if kind == "transposed":
        return parent.t()
    if kind == "sliced":
        return parent[:, ::2]
    if kind == "expanded":
        return parent.expand(parent.shape[1] // 4, parent.shape[1])
    return parent[::2, :]


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("kind,dtype", LAYOUT_ROWS)
def test__to_sparse_semi_structured_layout(kind, dtype):
    parent = _pattern_fixture("varied", dtype, _LAYOUT_BASES[dtype][kind])
    inp = _layout_view(kind, parent)
    ref_parent = tu.to_reference(parent)
    ref_inp = _layout_view(kind, ref_parent)

    res = flag_gems._to_sparse_semi_structured(inp)
    ref = torch.ops.aten._to_sparse_semi_structured(ref_inp)

    _assert_pair(res, ref)
    assert res[0].device == inp.device
    assert res[1].device == inp.device
    # The view must keep its parent allocation, geometry and offset.
    assert inp.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
    assert inp.storage_offset() == ref_inp.storage_offset()
    assert inp.stride() == ref_inp.stride()
    tu.assert_result_equal(inp, ref_inp)
    # The whole parent must survive the op, padding included.
    tu.assert_result_equal(parent, ref_parent)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("dtype,scenario", SPECIAL_ROWS)
def test__to_sparse_semi_structured_special_values(dtype, scenario):
    inp = _special_input(scenario, dtype, _PATTERN_SHAPE)
    ref_inp = tu.to_reference(inp)

    res = flag_gems._to_sparse_semi_structured(inp)
    ref = torch.ops.aten._to_sparse_semi_structured(ref_inp)

    _assert_pair(res, ref)
    assert res[0].device == inp.device
    assert res[1].device == inp.device
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("shape", _INVALID_RANK_SHAPES)
def test__to_sparse_semi_structured_invalid_rank(shape):
    inp = tu.make_input(torch.float16, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_semi_structured(inp)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("dtype,shape", [(torch.float16, s) for s in _INVALID_GEOMETRY])
def test__to_sparse_semi_structured_invalid_geometry(dtype, shape):
    # A legal 2:4 payload, so the only remaining violation is row alignment and the
    # rejection cannot come from the retained-nonzero count.
    inp = _pattern_fixture("fixed", dtype, shape)

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_semi_structured(inp)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("dtype,shape,mode", INVALID_PATTERN_ROWS)
def test__to_sparse_semi_structured_invalid_pattern(dtype, shape, mode):
    device = flag_gems.device
    if mode == "pruned":
        inp = torch.zeros(shape, dtype=dtype, device=device)
    else:
        inp = torch.ones(shape, dtype=dtype, device=device)

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_semi_structured(inp)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("dtype,shape", UNSUPPORTED_DTYPE_ROWS)
def test__to_sparse_semi_structured_unsupported_dtype(dtype, shape):
    inp = tu.make_input(dtype, shape, ["0", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._to_sparse_semi_structured(inp)
