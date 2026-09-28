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
from . import test_utils as tu

# multilabel_margin_loss scores an (N, C) matrix, or a single C-class row, against an
# int64 target of the same shape whose rows list active class ids terminated by -1.
# reduction follows at::Reduction: 0 is per-row, 1 the mean, 2 the sum. The native
# kernel takes rank <= 2 only, so the three high-rank spec shapes are folded to rank 2
# with the element count preserved: (20, 320, 15) -> (6400, 15),
# (16, 128, 64, 60) -> (131072, 60), (16, 7, 57, 32, 29) -> (204288, 29). Rank 0 and
# rank 1 are the single class-row form. Broadcast is exempt because input and target
# must have exactly the same shape: the native kernel rejects (4, 8) against (3, 8)
# with 'inconsistent target size: [3, 8] for input of size: [4, 8]'.
SPEC_SHAPES = [
    (),
    (1,),
    (256,),
    (1024, 1024),
    (6400, 15),
    (131072, 60),
    (204288, 29),
]
SHAPES = tu.selected_cases(SPEC_SHAPES, quick=[(2, 19)])
RANGES = tu.selected_ranges()
# tu.make_input takes symbolic range keys resolved per dtype, so the tests that do not
# sweep ranges reuse the first shared range ([-1, 1]) instead of literal bounds.
MAIN_RANGE = RANGES[0]
REDUCTIONS = tu.selected_cases([0, 1, 2], quick=[1])

# Probed on the NVIDIA CUDA backend: int8, uint8, int32, int64, float8_e4m3fn and
# float8_e5m2 raise 'multilabel_margin_loss_forward_kernel not implemented for
# Char / Byte / Int / Long / Float8_e4m3fn / Float8_e5m2', so only these floating
# types have a native forward kernel.
SUPPORTED_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    SUPPORTED_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    SUPPORTED_DTYPES.append(torch.float64)

# Kernel availability was measured on one vendor, so the input-dtype negatives are
# scoped statically to it, together with the probed dtype capability flags.
MEASURED_VENDOR = "nvidia"
ON_MEASURED_VENDOR = (
    getattr(flag_gems.runtime.device, "vendor_name", None) == MEASURED_VENDOR
)
UNSUPPORTED_DTYPES = []
if ON_MEASURED_VENDOR:
    UNSUPPORTED_DTYPES += [torch.int8, torch.uint8, torch.int32]
    if utils.int64_is_supported:
        UNSUPPORTED_DTYPES.append(torch.int64)
    if utils.fp8_is_supported:
        UNSUPPORTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]

TARGET_PATTERNS = [
    "empty",
    "single",
    "half",
    "full",
    "near_full",
    "permuted",
    "ragged",
    "payload",
]
MAIN_PATTERN = "half"
# Supplementary target patterns are a default-only dimension.
PATTERN_CASES = tu.selected_cases(TARGET_PATTERNS, quick=[])

VIEW_LAYOUTS = ["contiguous", "offset", "row_strided", "col_strided", "transposed"]
LAYOUT_PAIRS = tu.selected_cases(
    [
        (in_layout, tgt_layout)
        for in_layout in VIEW_LAYOUTS
        for tgt_layout in VIEW_LAYOUTS
    ],
    quick=[],
)
# Numeric out cases: a dense buffer and an offset buffer for every reduction. Every
# retained case compares the returned object, the post-call geometry and the complete
# logical values against the native out overload.
OUT_CASES = tu.selected_cases(
    [(r, layout) for r in (0, 1, 2) for layout in ("dense", "offset")], quick=[]
)
# Unresolved coverage gap, deliberately not exercised and not claimed as covered: a
# stride-2 out buffer. Measured natively at reduction 0 in float32 on a (8, 11) input,
# a (8,) stride-2 buffer received the eight per-row values contiguously in raw storage
# from the buffer base while its logical view read [r0, r2, r4, r6, fill, fill, fill,
# fill], which differs from the allocating native result; the independent repeatability
# probe reported the same disagreement for every stride-2 combination over four dtypes
# and three seeds. The native overload does not honour that layout, so the strided out
# value contract stays unresolved instead of being required, masked, or downgraded to
# an identity-only pass.
SPECIAL_LAYOUTS = ["target", "non_target", "batched"]
SPECIAL_CASES = tu.selected_cases(
    list(tu.special_value_cases(SUPPORTED_DTYPES)), quick=[]
)
# An empty batch carries no values, so the value-range dimension does not apply here.
EMPTY_BATCH_CASES = tu.selected_cases(SUPPORTED_DTYPES, quick=[])
CLASS_COUNT_CASES = tu.selected_cases(
    [
        (n, dtype, r)
        for n in (255, 256, 257)
        for dtype in SUPPORTED_DTYPES
        for r in RANGES
    ],
    quick=[],
)
EXTRA_REDUCTIONS = tu.selected_cases(
    [(r, dtype) for r in (-1, 3) for dtype in SUPPORTED_DTYPES], quick=[]
)
DEFAULT_REDUCTION_SHAPES = tu.selected_cases([(32, 47), (256, 8)], quick=[])
OUT_FILL = -3.0


def _class_rows(shape):
    # (rows, classes) of the logical (N, C) form; rank <= 1 is one class row.
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    rows = 1
    for extent in shape[:-1]:
        rows *= extent
    return rows, shape[-1]


def _fill_rows(target, pattern):
    rows, classes = target.shape
    ids = torch.arange(classes, device=target.device)
    if pattern == "empty":
        return
    if pattern == "single":
        target[:, 0] = 0
        return
    if pattern == "full":
        target[:] = ids
        return
    k = max(1, classes - 1) if pattern == "near_full" else max(1, classes // 2)
    if pattern == "permuted":
        # Active ids are the last k classes in descending order: a unique non-prefix
        # set that reaches class C-1, so an implementation assuming the active ids are
        # the prefix 0..k-1 computes a different loss.
        target[:, :k] = torch.arange(classes - k, classes, device=target.device).flip(0)
    elif pattern == "ragged":
        # Row r keeps 1 + r % k active ids, so row lengths differ inside one target.
        counts = 1 + torch.arange(rows, device=target.device) % k
        mask = torch.arange(k, device=target.device)[None, :] < counts[:, None]
        target[:, :k] = torch.where(mask, ids[:k], torch.full_like(ids[:k], -1))
    else:
        target[:, :k] = ids[:k]
        if pattern == "payload" and k + 1 < classes:
            # Ids behind the first -1 terminator must be ignored by the kernel.
            target[:, k + 1] = 0


def _make_target(shape, pattern):
    rows, classes = _class_rows(shape)
    target = torch.full((rows, classes), -1, dtype=torch.int64, device=flag_gems.device)
    if rows and classes:
        _fill_rows(target, pattern)
    return target.reshape(shape)


def _base_and_view(shape, layout):
    # Parent shape plus the callable that materialises the logical shape in it.
    rows, classes = _class_rows(shape)
    if layout == "contiguous":
        return (rows, classes), lambda tensor: tensor
    if layout == "offset":
        return (rows + 1, classes), lambda tensor: tensor[1:]
    if layout == "row_strided":
        return (2 * rows, classes), lambda tensor: tensor[1::2]
    if layout == "col_strided":
        return (rows, 2 * classes), lambda tensor: tensor[:, 1::2]
    return (classes, rows), lambda tensor: tensor.t()


def _view(buffer, shape, layout):
    return _base_and_view(shape, layout)[1](buffer)


def _laid_out_input(shape, layout, values):
    # Parent buffer whose view of the logical shape carries the values. The buffer
    # follows the value tensor's dtype and device so a reference-side oracle stays on
    # the configured reference device.
    base_shape, _ = _base_and_view(shape, layout)
    buffer = torch.zeros(base_shape, dtype=values.dtype, device=values.device)
    _view(buffer, shape, layout).copy_(values)
    return buffer


def _laid_out_target(shape, layout, pattern=MAIN_PATTERN):
    # Labels are built for the logical shape first and copied into the laid-out view,
    # so every layout keeps the unique active ids plus the -1 terminator; slicing or
    # transposing a parent-shaped target would duplicate class ids.
    base_shape, _ = _base_and_view(shape, layout)
    buffer = torch.full(base_shape, -1, dtype=torch.int64, device=flag_gems.device)
    _view(buffer, shape, layout).copy_(_make_target(shape, pattern))
    return _view(buffer, shape, layout)


def _out_buffer(out_shape, span, layout, dtype, device):
    # Dense buffer, or a slice of a longer parent so the out argument carries a
    # nonzero storage offset.
    if layout == "dense":
        return torch.full(out_shape, OUT_FILL, dtype=dtype, device=device)
    if layout == "offset":
        return torch.full((span + 1,), OUT_FILL, dtype=dtype, device=device)[1:]
    raise ValueError(f"no numeric out case for layout {layout}")


def _geometry(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())


def _special_input(dtype, shape, special, layout):
    # tu.make_special_input(dtype, scenario) returns the shared payload for that
    # scenario (nan; both infinities; nan with both infinities plus signed zeros and
    # finite controls). It is laid into the columns the target references ('target'),
    # the ignored ones ('non_target') or all of them ('batched'), cycling the payload
    # across those columns.
    inp = tu.make_input(dtype, shape, MAIN_RANGE)
    payload = tu.make_special_input(dtype, special)
    _, classes = _class_rows(shape)
    active = max(1, classes // 2)
    if layout == "target":
        cols = list(range(active))
    elif layout == "non_target":
        cols = list(range(active, classes))
    else:
        cols = list(range(classes))
    if not cols:
        return inp
    index = torch.arange(len(cols), device=inp.device) % payload.numel()
    inp[..., cols] = payload[index]
    return inp


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value_range", RANGES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_multilabel_margin_loss(shape, value_range, reduction, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, MAIN_PATTERN)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("pattern", PATTERN_CASES)
@pytest.mark.parametrize("value_range", RANGES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_multilabel_margin_loss_target_patterns(pattern, value_range, reduction, dtype):
    shape = (8, 11)
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, pattern)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("shape", DEFAULT_REDUCTION_SHAPES)
@pytest.mark.parametrize("value_range", RANGES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_multilabel_margin_loss_default_reduction(shape, value_range, dtype):
    # Omitting reduction exercises the schema default in the public call signature.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, MAIN_PATTERN)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target)
    res_out = flag_gems.multilabel_margin_loss(inp, target)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction,dtype", EXTRA_REDUCTIONS)
def test_multilabel_margin_loss_extra_reduction_values(reduction, dtype):
    shape = (32, 47)
    inp = tu.make_input(dtype, shape, MAIN_RANGE)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, MAIN_PATTERN)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction,out_layout", OUT_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_multilabel_margin_loss_out(reduction, out_layout, dtype):
    shape = (8, 11)
    inp = tu.make_input(dtype, shape, MAIN_RANGE)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, MAIN_PATTERN)
    ref_target = tu.to_reference(target)

    span = shape[0] if reduction == 0 else 1
    out_shape = (shape[0],) if reduction == 0 else ()
    # Independent sentinel-filled buffers per side, so the returned object, the
    # post-call geometry and the complete logical values can all be compared.
    res_buf = _out_buffer(out_shape, span, out_layout, dtype, inp.device)
    ref_buf = _out_buffer(out_shape, span, out_layout, dtype, ref_inp.device)

    ref_out = torch.ops.aten.multilabel_margin_loss.out(
        ref_inp, ref_target, reduction, out=ref_buf
    )
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction, out=res_buf)

    assert res_out is res_buf
    # The overload may resize the buffer it is handed (a (1,) buffer becomes 0-dim for
    # the scalar reductions), so geometry is compared after the call rather than
    # assumed.
    assert _geometry(res_out) == _geometry(ref_out)
    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("in_layout,tgt_layout", LAYOUT_PAIRS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_multilabel_margin_loss_noncontiguous(in_layout, tgt_layout, reduction, dtype):
    shape = (8, 11)
    values = tu.make_input(dtype, shape, MAIN_RANGE)
    inp = _view(_laid_out_input(shape, in_layout, values), shape, in_layout)
    ref_inp = tu.to_reference(inp)
    target = _laid_out_target(shape, tgt_layout)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("in_layout,tgt_layout", LAYOUT_PAIRS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_multilabel_margin_loss_backward(in_layout, tgt_layout, dtype):
    shape = (8, 11)
    values = tu.make_input(dtype, shape, MAIN_RANGE)
    # Independent laid-out leaves, one per side; the reference side keeps the
    # configured reference device and dtype.
    res_buf = _laid_out_input(shape, in_layout, values)
    ref_buf = _laid_out_input(shape, in_layout, tu.to_reference(values))
    res_buf.requires_grad_(True)
    ref_buf.requires_grad_(True)
    res_inp = _view(res_buf, shape, in_layout)
    ref_inp = _view(ref_buf, shape, in_layout)
    res_target = _laid_out_target(shape, tgt_layout)
    ref_target = tu.to_reference(res_target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, 1)
    res_out = flag_gems.multilabel_margin_loss(res_inp, res_target, 1)

    tu.assert_result_close(res_out, ref_out)

    ref_grad = torch.autograd.grad(ref_out, ref_buf)[0]
    res_grad = torch.autograd.grad(res_out, res_buf)[0]

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("dtype", EMPTY_BATCH_CASES)
def test_multilabel_margin_loss_empty_batch(dtype):
    # A batch without frames; reduction 0 compares the empty per-row result.
    shape = (0, 11)
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, MAIN_PATTERN)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, 0)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 0)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("n_classes,dtype,value_range", CLASS_COUNT_CASES)
def test_multilabel_margin_loss_class_count(n_classes, dtype, value_range):
    shape = (4, n_classes)
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, MAIN_PATTERN)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, 1)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 1)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("dtype,special", SPECIAL_CASES)
@pytest.mark.parametrize("layout", SPECIAL_LAYOUTS)
def test_multilabel_margin_loss_special_values(dtype, special, layout):
    shape = (8, 11)
    inp = _special_input(dtype, shape, special, layout)
    ref_inp = tu.to_reference(inp)
    target = _make_target(shape, MAIN_PATTERN)
    ref_target = tu.to_reference(target)

    ref_out = torch.ops.aten.multilabel_margin_loss(ref_inp, ref_target, 1)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 1)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_multilabel_margin_loss_invalid_input_dtype(dtype):
    inp = torch.zeros((8, 11), dtype=dtype, device=flag_gems.device)
    target = torch.full((8, 11), -1, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.int32, torch.int16])
def test_multilabel_margin_loss_invalid_target_dtype(target_dtype):
    inp = torch.zeros((8, 11), dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros((8, 11), dtype=target_dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("target_shape", [(4,), (3, 8), (4, 9)])
def test_multilabel_margin_loss_invalid_target_shape(target_shape):
    inp = torch.zeros((4, 8), dtype=torch.float32, device=flag_gems.device)
    target = torch.full(target_shape, -1, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", [1.5, "mean", None])
def test_multilabel_margin_loss_invalid_reduction(reduction):
    inp = torch.zeros((8, 11), dtype=torch.float32, device=flag_gems.device)
    target = torch.full((8, 11), -1, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, reduction)
