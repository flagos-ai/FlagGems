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
from . import conftest as cfg

# aten::multilabel_margin_loss is the dispatch target the wrapper replaces, so
# every reference goes through it rather than through the nn.functional alias.
_ATEN_OP = torch.ops.aten.multilabel_margin_loss


def _cases(cases, quick):
    """Pick the quick subset under --quick, the full list otherwise."""
    return list(quick) if cfg.QUICK_MODE else list(cases)


# The op scores an (N, C) matrix, or a single C-class row, against an int64
# target of the same shape whose rows list active class ids terminated by -1.
# reduction follows at::Reduction: 0 is per-row, 1 the mean, 2 the sum. The
# native kernel takes rank <= 2 only, so the high-rank shapes are folded to
# rank 2 with the element count preserved: (20, 320, 15) -> (6400, 15),
# (16, 128, 64, 60) -> (131072, 60), (16, 7, 57, 32, 29) -> (204288, 29).
# Rank 0 and rank 1 are the single class-row form. There is no broadcast
# dimension: input and target must have exactly the same shape, and the native
# kernel rejects a mismatch outright.
MML_SPEC_SHAPES = [
    (),
    (1,),
    (256,),
    (1024, 1024),
    (6400, 15),
    (131072, 60),
    (204288, 29),
]
# Quick mode keeps one small batched shape as a smoke case.
QUICK_SHAPES = [(2, 19)]
MML_SHAPES = _cases(MML_SPEC_SHAPES, QUICK_SHAPES)

# The scores feed a hinge, so the value ranges are swept per dtype: [0, max]
# and [min, 0] drive the margin terms to the dtype bounds where the difference
# overflows, while the signed sub-ranges separate the clamped-to-zero branch
# from the active one.
MML_VALUE_RANGES = [
    ("-1", "1"),
    ("0", "1"),
    ("-1", "0"),
    ("0", "max"),
    ("min", "0"),
]
QUICK_VALUE_RANGES = [("-1", "1")]
MML_RANGES = _cases(MML_VALUE_RANGES, QUICK_VALUE_RANGES)
# Tests that do not sweep ranges reuse the first shared range instead of
# spelling out literal bounds.
MAIN_RANGE = MML_RANGES[0]

MML_REDUCTIONS = _cases([0, 1, 2], [1])

# Probed on the NVIDIA CUDA backend: int8, uint8, int32, int64, float8_e4m3fn
# and float8_e5m2 raise "multilabel_margin_loss_forward_kernel not implemented
# for Char / Byte / Int / Long / Float8_e4m3fn / Float8_e5m2", so only these
# floating types have a native forward kernel to compare against.
MML_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    MML_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    MML_DTYPES.append(torch.float64)

# Kernel availability was probed on one vendor, so the input-dtype negatives
# are scoped statically to it alongside the probed capability flags.
ON_MEASURED_VENDOR = (
    getattr(flag_gems.runtime.device, "vendor_name", None)
    # The dtype rejections above were measured on the NVIDIA CUDA backend only.
    == "nvidia"
)
MML_UNSUPPORTED_DTYPES = []
if ON_MEASURED_VENDOR:
    MML_UNSUPPORTED_DTYPES += [torch.int8, torch.uint8, torch.int32]
    if utils.int64_is_supported:
        MML_UNSUPPORTED_DTYPES.append(torch.int64)
    if utils.fp8_is_supported:
        MML_UNSUPPORTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]

# Each pattern shapes the active target prefix of a row, which is what drives
# both the vector path (a single active id) and the outer-product path;
# "full" collapses the non-target set so no hinge pair survives at all, and
# "payload" puts live ids behind the first -1 terminator to prove they are
# ignored.
MML_TARGET_PATTERNS = [
    "empty",
    "single",
    "half",
    "full",
    "near_full",
    "permuted",
    "ragged",
    "payload",
    # The duplicate patterns are what distinguish a gradient that counts a
    # repeated class id once per entry from one that counts it once per id:
    # measured natively at C=3 with target [0, 0, -1], the gradient is
    # [-4/3, 2/3, 2/3], exactly twice the single-entry [-2/3, 1/3, 1/3]. The
    # forward loss scales the same way, so both sides of the op are covered.
    "dup_single",
    "dup_repeat",
    "dup_mixed",
]
MAIN_PATTERN = "half"
MML_PATTERN_CASES = _cases(MML_TARGET_PATTERNS, [])
# Patterns whose active prefix repeats a class id.
MML_DUP_PATTERNS = ["dup_single", "dup_repeat", "dup_mixed"]

# Both operands are addressed through explicit row and class strides, so every
# layout pairing is swept: a view whose storage order differs from its logical
# order must still be read in logical order.
MML_VIEW_LAYOUTS = ["contiguous", "offset", "row_strided", "col_strided", "transposed"]
MML_LAYOUT_PAIRS = _cases(
    [
        (in_layout, tgt_layout)
        for in_layout in MML_VIEW_LAYOUTS
        for tgt_layout in MML_VIEW_LAYOUTS
    ],
    [("transposed", "col_strided")],
)

# Numeric out cases: a dense buffer and an offset buffer for every reduction.
# A stride-2 out buffer is deliberately left out rather than claimed as
# covered: measured natively at reduction 0 in float32 on an (8, 11) input, a
# (8,) stride-2 buffer received the eight per-row values contiguously in raw
# storage from the buffer base while its logical view read
# [r0, r2, r4, r6, fill, fill, fill, fill], which disagrees with the allocating
# native result. The native overload does not honour that layout, so the
# strided out value contract stays unresolved instead of being masked.
MML_OUT_CASES = _cases(
    [(r, layout) for r in (0, 1, 2) for layout in ("dense", "offset")],
    [(1, "dense")],
)
# Sentinel fill, so a buffer element the overload never writes is visible.
OUT_FILL = -3.0

# A non-finite value reached through a target id and one reached only as a
# non-target class take different routes into the hinge; "batched" poisons
# every column and checks that one row does not leak into the others.
MML_SPECIAL_LAYOUTS = ["target", "non_target", "batched"]
MML_SPECIAL_SCENARIOS = ["nan", "inf", "mixed"]
MML_SPECIAL_CASES = _cases(
    [(dtype, scenario) for dtype in MML_DTYPES for scenario in MML_SPECIAL_SCENARIOS],
    [],
)

# An empty batch carries no values, so the value-range dimension does not apply.
# Measured natively on a (0, 11) input: every reduction returns shape (0,), so
# the mean does not produce a 0-dim nan and the sum does not produce a 0-dim
# zero. All three are swept rather than reduction 0 alone.
MML_EMPTY_BATCH_CASES = _cases(
    [(dtype, r) for dtype in MML_DTYPES for r in (0, 1, 2)],
    [],
)

# The class axis is tiled in blocks of 64, so the counts sit either side of a
# whole number of tiles and exercise the partial-tile mask. The 1023/1024/1025
# and 2047/2048/2049 triples straddle the reduction-0 single-tile cap (1024 for
# fp64, 2048 otherwise), which is where a row stops being computed by one
# launch and moves to the split grid plus the finish kernel.
MML_CLASS_COUNTS = (255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049)
MML_CLASS_COUNT_CASES = _cases(
    [
        (n, dtype, value_range)
        for n in MML_CLASS_COUNTS
        for dtype in MML_DTYPES
        for value_range in MML_RANGES
    ],
    [],
)
# Class counts large enough that no dtype takes the single-tile path, swept over
# every reduction so the split grid's partial buffer and finish kernel are
# covered on the scalar reductions too. Kept to one dtype per count and run on a
# 2-row input, since the point is the class-axis split rather than the dtype.
MML_SPLIT_PATH_CASES = _cases(
    [(n, r) for n in (4095, 4096, 4097) for r in (0, 1, 2)],
    [],
)

# Every integer other than 0 and 1 is treated as "sum" by the native kernel.
MML_EXTRA_REDUCTIONS = _cases([(r, dtype) for r in (-1, 3) for dtype in MML_DTYPES], [])
MML_DEFAULT_REDUCTION_SHAPES = _cases([(32, 47), (256, 8)], [])
# Non-divisible rows and classes, so both axes leave a partial tile.
MML_MAIN_SHAPE = (8, 11)


def _dtype_bounds(dtype):
    """Return the (min, max) representable values of ``dtype``."""
    finfo = torch.finfo(dtype)
    return finfo.min, finfo.max


def _resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol to a concrete value for ``dtype``."""
    low, high = _dtype_bounds(dtype)
    # The symbol table is the value-range vocabulary used by the case lists
    # above; "max"/"min" mean the dtype's own bounds rather than fixed numbers.
    return {"-1": -1.0, "0": 0.0, "1": 1.0, "max": high, "min": low}[symbol]


def _make_input(dtype, shape, value_range=MAIN_RANGE):
    """Build a score tensor of ``shape`` / ``dtype`` with values in range."""
    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)
    if low == high:
        return torch.full(shape, low, dtype=dtype, device=flag_gems.device)
    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def _special_payload(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return torch.tensor(
        payloads[scenario], dtype=torch.float32, device=flag_gems.device
    ).to(dtype)


def _class_rows(shape):
    """(rows, classes) of the logical (N, C) form; rank <= 1 is one class row."""
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
    if pattern == "dup_single":
        # One class id repeated across the whole active prefix, so the per-entry
        # multiplicity is the prefix length itself.
        target[:, : max(1, classes // 2)] = 0
        return
    if pattern == "dup_repeat":
        # Each active id appears exactly twice, by halving the id stride.
        k = max(2, (classes // 2) & ~1)
        target[:, :k] = (torch.arange(k, device=target.device) // 2) % classes
        return
    if pattern == "dup_mixed":
        # A mix of multiplicities in one row: id 0 three times, id 1 twice and
        # the remaining ids once, so no single multiplicity can be hoisted.
        k = max(1, classes // 2)
        head = torch.tensor([0, 0, 0, 1, 1], device=target.device)[:k]
        rest = torch.arange(2, 2 + max(0, k - head.numel()), device=target.device)
        target[:, :k] = torch.cat([head, rest % classes])[:k]
        return
    # "near_full" keeps every class but one active; the rest keep half.
    k = max(1, classes - 1) if pattern == "near_full" else max(1, classes // 2)
    if pattern == "permuted":
        # Active ids are the last k classes in descending order: a unique
        # non-prefix set that reaches class C-1, so an implementation assuming
        # the active ids are the prefix 0..k-1 computes a different loss.
        target[:, :k] = torch.arange(classes - k, classes, device=target.device).flip(0)
    elif pattern == "ragged":
        # Row r keeps 1 + r % k active ids, so row lengths differ inside one
        # target and the per-row terminator search cannot be hoisted.
        counts = 1 + torch.arange(rows, device=target.device) % k
        mask = torch.arange(k, device=target.device)[None, :] < counts[:, None]
        target[:, :k] = torch.where(mask, ids[:k], torch.full_like(ids[:k], -1))
    else:
        target[:, :k] = ids[:k]
        if pattern == "payload" and k + 1 < classes:
            # An id behind the first -1 terminator, which must be ignored.
            target[:, k + 1] = 0


def _make_target(shape, pattern=MAIN_PATTERN):
    """Build an int64 target whose rows hold class ids ended by the first -1."""
    rows, classes = _class_rows(shape)
    target = torch.full((rows, classes), -1, dtype=torch.int64, device=flag_gems.device)
    if rows and classes:
        _fill_rows(target, pattern)
    return target.reshape(shape)


def _base_and_view(shape, layout):
    """Parent shape plus the callable materialising the logical shape in it."""
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
    """Parent buffer whose view of the logical shape carries ``values``."""
    base_shape, _ = _base_and_view(shape, layout)
    # The buffer follows the value tensor's dtype and device, so a
    # reference-side oracle stays on the configured reference device.
    buffer = torch.zeros(base_shape, dtype=values.dtype, device=values.device)
    _view(buffer, shape, layout).copy_(values)
    return buffer


def _laid_out_target(shape, layout, pattern=MAIN_PATTERN):
    """Target laid out in ``layout`` while keeping its logical class ids."""
    base_shape, _ = _base_and_view(shape, layout)
    buffer = torch.full(base_shape, -1, dtype=torch.int64, device=flag_gems.device)
    # Labels are built for the logical shape first and copied into the laid-out
    # view, so every layout keeps unique active ids plus the -1 terminator;
    # slicing or transposing a parent-shaped target would duplicate class ids.
    _view(buffer, shape, layout).copy_(_make_target(shape, pattern))
    return _view(buffer, shape, layout)


def _out_buffer(out_shape, span, layout, dtype, device):
    """Dense buffer, or a slice of a longer parent for a nonzero offset."""
    if layout == "dense":
        return torch.full(out_shape, OUT_FILL, dtype=dtype, device=device)
    return torch.full((span + 1,), OUT_FILL, dtype=dtype, device=device)[1:]


def _geometry(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())


def _special_input(dtype, shape, scenario, layout):
    """Score tensor with the scenario payload laid into the chosen columns."""
    inp = _make_input(dtype, shape)
    payload = _special_payload(dtype, scenario)
    _, classes = _class_rows(shape)
    # MAIN_PATTERN activates the first half of the classes, so these are the
    # columns the target references.
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


def _ref_input(inp):
    """Reference-side copy of a score tensor.

    The reference is upcast only when it runs on CPU, where this op has no fp16
    or bf16 forward kernel. On device it keeps the input dtype, so the
    reference saturates exactly where the kernel does and the extreme value
    ranges stay meaningful.
    """
    return utils.to_reference(inp, utils.TO_CPU)


def _reduce_dim(shape, reduction):
    rows, n_classes = _class_rows(shape)
    # Each output element is a sum over up to rows * n_classes hinge terms, so
    # the comparison budget has to grow with the number of accumulated pairs.
    return n_classes if reduction == 0 else max(1, rows * n_classes)


def _assert_close(res, ref, shape, reduction):
    utils.gems_assert_close(
        res,
        ref,
        res.dtype,
        equal_nan=True,
        reduce_dim=_reduce_dim(shape, reduction),
    )


def _grad_oracle(inp, target, reduction, seed=None):
    """Vectorised torch reference for d(loss)/d(input).

    The gradient reference is computed here instead of via
    ``torch.autograd.grad`` on the native op, because the native *CUDA*
    backward drops the target-side term outright for any target class id in
    ``[32, 128)``. Measured on an H20-3e with torch 2.9.0+cu128, a single-entry
    target at ``C=64``: ids 0..31 give ``grad[id] = -0.984375`` and agree with
    the CPU backward, while ids 32..63 give ``grad[id] = 0``; the pattern starts
    at 32 and stops at 128 for every ``C`` (``C=33`` loses only id 32, ``C=200``
    loses 32..127), is independent of the row count and the reduction, and does
    not affect the forward loss, which agrees to 7e-15. The CPU backward and
    this oracle agree with each other and with an entry-by-entry transcription
    of the native algorithm, so the device backward is the outlier and is not a
    usable contract.

    The formula is the native one, with ``m[d]`` the number of times class ``d``
    occurs in the active prefix:
      ``d`` in S  : ``-m[d] * #{k in NT : 1 - x[d] + x[k] > 0}``
      ``d`` in NT : ``+sum_{k in S} m[k] * [1 - x[k] + x[d] > 0]``
    scaled by ``1/C`` and, for the mean, by ``1/N``.
    """
    rows, classes = _class_rows(tuple(inp.shape))
    # The oracle is the reference side, so it has to live wherever the harness
    # puts references: under --ref cpu, gems_assert_close requires a CPU
    # reference, and this oracle does not go through to_reference on the way in.
    x = utils.to_reference(inp.detach()).reshape(rows, classes)
    t = utils.to_reference(target.detach()).reshape(rows, classes)
    # The hinge is evaluated in the *input* dtype, as the kernel and the native
    # op both do, rather than upcast. The counts it produces are integers, so a
    # term sitting exactly on 1 - x[d] + x[k] == 0 that flips sign between two
    # precisions moves a count by a whole unit rather than by a rounding error,
    # and at the extreme value ranges the margin has to saturate where the
    # kernel's does. ``to_reference`` is called without upcast for that reason;
    # the subtraction and comparison used here are available for fp16 and
    # bfloat16 on the CPU reference path too.
    # Counts and the 1/C scaling are accumulated in the op's at::acc_type.
    acc = torch.float64 if inp.dtype == torch.float64 else torch.float32

    ids = torch.arange(classes, device=t.device)
    # The active prefix ends at the first negative entry.
    neg = t < 0
    first_neg = torch.where(
        neg.any(dim=1),
        neg.to(torch.uint8).argmax(dim=1),
        torch.full((rows,), classes, dtype=torch.long, device=t.device),
    )
    valid = (ids[None, :] < first_neg[:, None]) & (t >= 0) & (t < classes)

    # m[r, d]: multiplicity of class d in row r's active prefix.
    mult = torch.zeros((rows, classes), dtype=acc, device=x.device)
    mult.scatter_add_(1, torch.where(valid, t, torch.zeros_like(t)), valid.to(acc))
    is_t = mult > 0
    nt = ~is_t

    cnt_t = torch.zeros((rows, classes), dtype=acc, device=x.device)
    cnt_n = torch.zeros((rows, classes), dtype=acc, device=x.device)
    mult_t = mult * is_t
    # Both counts reduce over the class axis, so that axis is the one chunked:
    # a large class count never needs the whole classes x classes pair tile
    # resident at once, while the result axis stays full width.
    chunk = max(1, 4096 // max(1, rows))
    for lo in range(0, classes, chunk):
        hi = min(lo + chunk, classes)
        # d plays the target, the chunk the non-target class.
        hit_t = (1.0 - x[:, :, None] + x[:, None, lo:hi]) > 0.0
        cnt_t += (hit_t & nt[:, None, lo:hi]).to(acc).sum(dim=2)
        # The chunk plays the target, d the non-target class.
        hit_n = (1.0 - x[:, lo:hi, None] + x[:, None, :]) > 0.0
        cnt_n += (hit_n.to(acc) * mult_t[:, lo:hi, None]).sum(dim=1)

    g = torch.where(is_t, -mult * cnt_t, cnt_n) / classes
    if reduction == 1:
        g = g / rows
    if seed is not None:
        s = utils.to_reference(seed.detach()).to(acc).reshape(-1)
        g = g * (s[:, None] if s.numel() == rows else s.reshape(()))
    return g.reshape(inp.shape).to(inp.dtype)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("shape", MML_SHAPES)
@pytest.mark.parametrize("value_range", MML_RANGES)
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss(shape, value_range, reduction, dtype):
    inp = _make_input(dtype, shape, value_range)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    assert res_out.shape == ref_out.shape
    _assert_close(res_out, ref_out, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("pattern", MML_PATTERN_CASES)
@pytest.mark.parametrize("value_range", MML_RANGES)
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_target_patterns(pattern, value_range, reduction, dtype):
    shape = MML_MAIN_SHAPE
    inp = _make_input(dtype, shape, value_range)
    target = _make_target(shape, pattern)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    _assert_close(res_out, ref_out, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("shape", MML_DEFAULT_REDUCTION_SHAPES)
@pytest.mark.parametrize("value_range", MML_RANGES)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_default_reduction(shape, value_range, dtype):
    # Omitting reduction exercises the schema default ("mean"), which has to
    # produce a 0-dim result.
    inp = _make_input(dtype, shape, value_range)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target)
    res_out = flag_gems.multilabel_margin_loss(inp, target)

    assert res_out.shape == ()
    _assert_close(res_out, ref_out, shape, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction, dtype", MML_EXTRA_REDUCTIONS)
def test_multilabel_margin_loss_extra_reduction_values(reduction, dtype):
    # Enough rows that a mean and a sum differ by more than the tolerance, with
    # a class count that leaves a partial tile.
    shape = (32, 47)
    inp = _make_input(dtype, shape)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    _assert_close(res_out, ref_out, shape, 2)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction, out_layout", MML_OUT_CASES)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_out(reduction, out_layout, dtype):
    shape = MML_MAIN_SHAPE
    inp = _make_input(dtype, shape)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    span = shape[0] if reduction == 0 else 1
    out_shape = (shape[0],) if reduction == 0 else ()
    # Independent sentinel-filled buffers per side, so the returned object, the
    # post-call geometry and the complete logical values can all be compared.
    res_buf = _out_buffer(out_shape, span, out_layout, dtype, inp.device)
    # The reference buffer follows the reference input's dtype, which differs
    # from the result dtype when the reference is upcast on CPU.
    ref_buf = _out_buffer(out_shape, span, out_layout, ref_inp.dtype, ref_inp.device)

    ref_out = torch.ops.aten.multilabel_margin_loss.out(
        ref_inp, ref_target, reduction, out=ref_buf
    )
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction, out=res_buf)

    assert res_out is res_buf
    # The overload may resize the buffer it is handed (a (1,) buffer becomes
    # 0-dim for the scalar reductions), so geometry is compared after the call
    # rather than assumed.
    assert _geometry(res_out) == _geometry(ref_out)
    _assert_close(res_out, ref_out, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("in_layout, tgt_layout", MML_LAYOUT_PAIRS)
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_noncontiguous(in_layout, tgt_layout, reduction, dtype):
    shape = MML_MAIN_SHAPE
    values = _make_input(dtype, shape)
    inp = _view(_laid_out_input(shape, in_layout, values), shape, in_layout)
    target = _laid_out_target(shape, tgt_layout)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    _assert_close(res_out, ref_out, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("in_layout, tgt_layout", MML_LAYOUT_PAIRS)
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_backward(in_layout, tgt_layout, reduction, dtype):
    shape = MML_MAIN_SHAPE
    values = _make_input(dtype, shape)
    # A laid-out leaf, so the gradient is compared on the parent buffer rather
    # than on a view of it.
    res_buf = _laid_out_input(shape, in_layout, values).requires_grad_(True)
    res_inp = _view(res_buf, shape, in_layout)
    target = _laid_out_target(shape, tgt_layout)
    ref_inp = _ref_input(values)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(res_inp, target, reduction)

    _assert_close(res_out, ref_out, shape, reduction)

    # Reduction 0 returns a vector, so the backward needs a seed gradient; the
    # scalar reductions differentiate directly.
    res_seed = torch.ones_like(res_out) if reduction == 0 else None
    res_grad = torch.autograd.grad(res_out, res_buf, grad_outputs=res_seed)[0]
    # The gradient lands on the parent buffer, so only its logical view carries
    # the result; the rest of the parent stays zero.
    ref_grad = _grad_oracle(res_inp, target, reduction, res_seed)

    _assert_close(_view(res_grad, shape, in_layout), ref_grad, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("pattern", MML_DUP_PATTERNS)
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("value_range", MML_RANGES)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_backward_duplicate_targets(
    pattern, reduction, value_range, dtype
):
    # A repeated class id must contribute once per target *entry*, not once per
    # distinct id: measured natively at C=3 with target [0, 0, -1] the gradient
    # is [-4/3, 2/3, 2/3], twice the single-entry value. An implementation that
    # drops the multiplicity factor on the target side still matches the forward
    # loss, so only a backward comparison over a duplicated prefix catches it.
    shape = MML_MAIN_SHAPE
    values = _make_input(dtype, shape, value_range)
    res_inp = values.clone().requires_grad_(True)
    target = _make_target(shape, pattern)
    ref_inp = _ref_input(values)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(res_inp, target, reduction)

    _assert_close(res_out, ref_out, shape, reduction)

    res_seed = torch.ones_like(res_out) if reduction == 0 else None
    res_grad = torch.autograd.grad(res_out, res_inp, grad_outputs=res_seed)[0]
    ref_grad = _grad_oracle(values, target, reduction, res_seed)

    _assert_close(res_grad, ref_grad, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("pattern", _cases([MAIN_PATTERN] + MML_DUP_PATTERNS, []))
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_backward_strided_grad_out(pattern, dtype):
    # Reduction 0's upstream gradient is a vector, so it can arrive as a
    # non-contiguous view. The kernel reads it through its own stride rather
    # than copying it, which only a strided seed distinguishes from a
    # contiguous one, and a non-uniform seed is needed for the stride to
    # matter at all.
    shape = MML_MAIN_SHAPE
    rows = shape[0]
    values = _make_input(dtype, shape)
    res_inp = values.clone().requires_grad_(True)
    target = _make_target(shape, pattern)

    # A stride-2 view of a 2N-long parent, so consecutive logical elements are
    # two storage slots apart.
    res_seed = torch.zeros(2 * rows, dtype=dtype, device=flag_gems.device)[::2]
    res_seed.copy_(torch.arange(1, rows + 1, dtype=torch.float32).to(dtype))
    assert res_seed.stride() == (2,)

    res_out = flag_gems.multilabel_margin_loss(res_inp, target, 0)
    res_grad = torch.autograd.grad(res_out, res_inp, grad_outputs=res_seed)[0]
    ref_grad = _grad_oracle(values, target, 0, res_seed)

    _assert_close(res_grad, ref_grad, shape, 0)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("n_classes", _cases([255, 1024, 4097], []))
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
def test_multilabel_margin_loss_backward_large_class_count(n_classes, reduction):
    # The backward tiles the class axis too, so a class count past one tile
    # covers the grid's second dimension. The oracle reference is what makes
    # these counts testable at all: the native CUDA backward zeroes the
    # target-side term for ids in [32, 128), which every one of these shapes
    # reaches.
    shape = (2, n_classes)
    values = _make_input(torch.float32, shape)
    res_inp = values.clone().requires_grad_(True)
    target = _make_target(shape)

    res_out = flag_gems.multilabel_margin_loss(res_inp, target, reduction)
    res_seed = torch.ones_like(res_out) if reduction == 0 else None
    res_grad = torch.autograd.grad(res_out, res_inp, grad_outputs=res_seed)[0]
    ref_grad = _grad_oracle(values, target, reduction, res_seed)

    _assert_close(res_grad, ref_grad, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("dtype, reduction", MML_EMPTY_BATCH_CASES)
def test_multilabel_margin_loss_empty_batch(dtype, reduction):
    # A batch without frames. On the device path every reduction returns an
    # empty (0,) result, not the 0-dim scalar a populated batch would give.
    #
    # The two aten backends disagree here, so only one of them can be the
    # contract: measured on a (0, 11) fp32 input, CUDA returns shape (0,) for
    # all three reductions, while CPU returns (0,) for reduction 0 and a 0-dim
    # tensor holding *uninitialised* memory for reductions 1 and 2 (observed
    # values 2.4e-44 and -1.5e+21 on consecutive calls). An uninitialised
    # scalar cannot be a reference, so the device shape is the contract and the
    # scalar reductions are only compared when the reference runs on device.
    shape = (0, 11)
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    assert res_out.shape == (0,)

    if reduction != 0 and utils.TO_CPU:
        pytest.skip("aten CPU returns uninitialised memory for this case")

    ref_out = _ATEN_OP(_ref_input(inp), utils.to_reference(target), reduction)
    assert res_out.shape == ref_out.shape
    _assert_close(res_out, ref_out, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("n_classes, dtype, value_range", MML_CLASS_COUNT_CASES)
def test_multilabel_margin_loss_class_count(n_classes, dtype, value_range):
    shape = (4, n_classes)
    inp = _make_input(dtype, shape, value_range)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, 1)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 1)

    _assert_close(res_out, ref_out, shape, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("n_classes, reduction", MML_SPLIT_PATH_CASES)
def test_multilabel_margin_loss_class_split_path(n_classes, reduction):
    # Past the reduction-0 single-tile cap the class axis is split across
    # programs and folded by a second launch, so these counts exercise the
    # partial buffer and the finish kernel that the smaller counts never reach.
    shape = (2, n_classes)
    inp = _make_input(torch.float32, shape)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    assert res_out.shape == ref_out.shape
    _assert_close(res_out, ref_out, shape, reduction)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("dtype, special", MML_SPECIAL_CASES)
@pytest.mark.parametrize("layout", MML_SPECIAL_LAYOUTS)
def test_multilabel_margin_loss_special_values(dtype, special, layout):
    shape = MML_MAIN_SHAPE
    inp = _special_input(dtype, shape, special, layout)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, 1)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 1)

    _assert_close(res_out, ref_out, shape, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("bad_id", [11, 12, 1 << 40, (1 << 63) - 1, -2, -(1 << 40)])
def test_multilabel_margin_loss_out_of_range_target(bad_id, reduction):
    # An id outside [0, C) is garbage input, and the native kernel indexes with
    # it unchecked: measured on CUDA, C=3 with id 5 returns a value assembled
    # from memory past the row rather than raising, so there is no reference
    # value to compare against and none is asserted. What is asserted is that
    # the op stays memory-safe and produces a finite result of the right shape,
    # which is what the in-kernel address clamp guarantees. Ids are swept on
    # both sides of the range, including the int64 extremes whose unclamped
    # address arithmetic would wrap.
    shape = MML_MAIN_SHAPE
    inp = _make_input(torch.float32, shape)
    target = _make_target(shape)
    target[:, 0] = bad_id

    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)
    torch.cuda.synchronize() if inp.is_cuda else None

    assert res_out.shape == ((shape[0],) if reduction == 0 else ())
    assert torch.isfinite(res_out).all()


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32, torch.float64])
def test_multilabel_margin_loss_out_dtype(out_dtype):
    # The native overload casts into whatever floating buffer it is handed
    # rather than requiring the input dtype (measured: fp16 and fp64 buffers
    # both accept an fp32 input), so the wrapper accepts the same set.
    if out_dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("fp64 is not supported on this device")
    shape = MML_MAIN_SHAPE
    inp = _make_input(torch.float32, shape)
    target = _make_target(shape)
    ref_inp = _ref_input(inp)
    ref_target = utils.to_reference(target)

    res_buf = torch.full((), OUT_FILL, dtype=out_dtype, device=inp.device)
    # The buffer under test carries out_dtype; the reference buffer follows the
    # reference input's dtype, which is upcast on the CPU reference path.
    ref_buf = torch.full((), OUT_FILL, dtype=ref_inp.dtype, device=ref_inp.device)

    ref_out = torch.ops.aten.multilabel_margin_loss.out(
        ref_inp, ref_target, 1, out=ref_buf
    )
    res_out = flag_gems.multilabel_margin_loss(inp, target, 1, out=res_buf)

    assert res_out is res_buf
    assert res_out.dtype == out_dtype
    _assert_close(res_out, ref_out, shape, 1)


@pytest.mark.multilabel_margin_loss
def test_multilabel_margin_loss_out_invalid():
    # A non-floating buffer and a buffer on another device are both rejected,
    # the latter matching the native overload's cross-device error.
    shape = MML_MAIN_SHAPE
    inp = _make_input(torch.float32, shape)
    target = _make_target(shape)

    int_buf = torch.zeros((), dtype=torch.int32, device=inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1, out=int_buf)

    if inp.device.type != "cpu":
        cpu_buf = torch.zeros((), dtype=torch.float32, device="cpu")
        with pytest.raises((RuntimeError, TypeError)):
            flag_gems.multilabel_margin_loss(inp, target, 1, out=cpu_buf)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("dtype", MML_UNSUPPORTED_DTYPES)
def test_multilabel_margin_loss_invalid_input_dtype(dtype):
    inp = torch.zeros(MML_MAIN_SHAPE, dtype=dtype, device=flag_gems.device)
    target = _make_target(MML_MAIN_SHAPE)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.int32, torch.int16])
def test_multilabel_margin_loss_invalid_target_dtype(target_dtype):
    inp = torch.zeros(MML_MAIN_SHAPE, dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros(MML_MAIN_SHAPE, dtype=target_dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("target_shape", [(4,), (3, 8), (4, 9)])
def test_multilabel_margin_loss_invalid_target_shape(target_shape):
    # A (4, 8) input against a target that differs in rank, in rows and in
    # classes; the native kernel requires exactly the same shape.
    inp = torch.zeros((4, 8), dtype=torch.float32, device=flag_gems.device)
    target = torch.full(target_shape, -1, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", [1.5, "mean", None])
def test_multilabel_margin_loss_invalid_reduction(reduction):
    inp = torch.zeros(MML_MAIN_SHAPE, dtype=torch.float32, device=flag_gems.device)
    target = _make_target(MML_MAIN_SHAPE)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, reduction)


@pytest.mark.multilabel_margin_loss
def test_multilabel_margin_loss_invalid_rank():
    # The native kernel takes rank <= 2 only, so a rank-3 input is rejected
    # even when the target matches it exactly.
    inp = torch.zeros((2, 3, 4), dtype=torch.float32, device=flag_gems.device)
    target = torch.full((2, 3, 4), -1, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.multilabel_margin_loss(inp, target, 1)
