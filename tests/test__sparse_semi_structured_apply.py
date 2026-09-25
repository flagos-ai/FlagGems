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

"""Correctness tests for ``torch.ops.aten._sparse_semi_structured_apply``.

The operator takes a rank-2 activation tile ``input`` of shape (M, K) and a
``uint8`` ``thread_masks`` tensor, and returns two tensors: ``out0`` of shape
(M, K / 2) and ``out1`` of shape (K, M / 2), both contiguous and in the input
dtype. Neither output is computed arithmetically: with a uniform 0xFF mask on a
(32, 64) tile every element of both outputs is a value of the input tile, and the
two results are independently packed orientations of the mask-selected values,
not transposes of one another (``out0.T`` would be (K / 2, M), not (K, M / 2)).
Both outputs are therefore compared with the shared zero-tolerance equality
assertion.

The constraints the workloads below are built around, all measured through
``torch.ops.aten`` on the active (NVIDIA) backend -- the probe transcript is kept
with the run's review evidence rather than here:
  * the tile is rank 2 with positive extents, ``M % 32 == 0`` and ``K % 64 == 0``.
    Rank 0/1 fail while the native wrapper reads ``size(1)`` (IndexError) and rank
    >= 3 fails its ``input.dim() == 2`` check (RuntimeError), so the 0-D and 1-D
    spec rungs can only appear as negative workloads;
  * ``input`` is read through its strides: a row-major tile (``stride(1) == 1``)
    requires ``stride(0) % 8 == 0``, while a column-major tile (``stride(0) == 1``)
    or a tile whose second stride is 8-aligned is accepted as it stands;
  * ``thread_masks`` is ``uint8`` with the structural shape (M / 8, K / 8, 8) and
    contiguous 8-byte groups (``stride(1) == 8``, ``stride(2) == 1``); a non-zero
    storage offset is accepted and the operand is only ever read;
  * float16 and bfloat16 are the only accepted input dtypes.

Broadcast does not apply: ``thread_masks`` is a structural mask whose extents are
pinned to (M / 8, K / 8, 8), any other extent being rejected, so no pair of
extents could broadcast. There is no tensor/scalar form either, because both
operands are tensors, and ``overloads()`` returns ['default'], so no ``.out``
overload exists to exercise.

Backward does not apply: ``torch.autograd.grad`` against the native operator with
``grad_outputs`` raises 'derivative for aten::_sparse_semi_structured_apply is not
implemented' for either output and either supported dtype.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ``pytest.mark`` refuses attribute access for underscore-prefixed names, so the
# marker is registered on the MarkGenerator directly.
setattr(
    pytest.mark,
    "_sparse_semi_structured_apply",
    MarkDecorator(
        Mark("_sparse_semi_structured_apply", (), {}, _ispytest=True), _ispytest=True
    ),
)

# Only float16 and bfloat16 reach the native kernel. bfloat16 is gated on the
# static device capability so collection works where the backend lacks it.
SUPPORTED_DTYPES = [torch.float16] + (
    [torch.bfloat16] if utils.bf16_is_supported else []
)

# The spec shape ladder is projected onto legal 2:4 tiles: the native kernel takes
# rank 2 with M % 32 == 0 and K % 64 == 0, so the 0-D/1-D rungs can only be
# negative ranks (below), the 3-D/4-D/5-D rungs all reach the same rank-2 kernel,
# and the quick specification (2, 19, 7) becomes the smallest legal tile. The first
# eleven rows are the original ladder; the last three adapt the higher-rank rungs,
# whose element counts cannot all be factored into aligned extents:
# (16, 128, 64, 60) = 7864320 = 2048 * 3840 exactly, while (20, 320, 15) = 96000
# and (16, 7, 57, 32, 29) = 5924352 are not divisible by 32 * 64, so (320, 320)
# (102400 elements) and (2048, 2880) (5898240 elements) keep the element magnitude
# of the rungs they adapt.
TILE_SHAPES = tu.selected_cases(
    [
        (32, 64),
        (32, 128),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (256, 256),
        (320, 640),
        (512, 1024),
        (1024, 1024),
        (2048, 1024),
        (320, 320),
        (2048, 2880),
        (2048, 3840),
    ],
    quick=[(32, 64)],
)

# Every byte value is a legal mask: the 8 single-bit patterns, their 8 complements,
# 0x00, 0xFF, 0x0F, 0x55 and 0xAA all ran without error, and 200 uniform-random
# uint8 masks were accepted. A uniform byte selects the same lane in every group and
# every row, so it cannot detect a candidate that permutes lanes within a group;
# "lane_bits" (a different bit per lane) and "row_ramp" (a different byte per row)
# vary the selection so such a permutation changes the result.
SINGLE_BIT_PATTERNS = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
COMPLEMENT_PATTERNS = [0xFE, 0xFD, 0xFB, 0xF7, 0xEF, 0xDF, 0xBF, 0x7F]
MASK_PATTERNS = tu.selected_cases(
    SINGLE_BIT_PATTERNS
    + COMPLEMENT_PATTERNS
    + [0x00, 0xFF, 0x0F, 0x55, 0xAA, "lane_bits", "row_ramp", "random"],
    quick=[],
)
MASK_SHAPES = tu.selected_cases([(32, 64), (256, 256), (512, 1024)], quick=[])
MASK_LAYOUT_SHAPE = (256, 256)

# The kernel reads both operands through their strides. Measured valid tiles:
# (128, 512)[::2] -> (64, 512) stride (1024, 1); (64, 512)[:, ::2] -> (64, 256)
# stride (512, 2); (72, 256)[8:] -> (64, 256) with storage offset 2048; and the
# column-major (256, 64).transpose(0, 1) -> (64, 256) stride (1, 64), which is
# legal because a leading stride of 1 needs no further alignment. Each returned
# results identical to those of its contiguous copy.
STRIDED_LAYOUTS = tu.selected_cases(
    [
        ("row_step2", (128, 512)),
        ("col_step2", (64, 512)),
        ("row_offset", (72, 256)),
        ("transpose", (256, 64)),
    ],
    quick=[],
)

SPECIAL_VALUES = tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[])

# The rejection matrices below were measured on the NVIDIA backend, where the native
# kernel raises 'Unsupported dtype - only `float16` and `bfloat16` are supported
# currently' for every other input dtype and requires ``uint8`` masks. They are
# implementation properties, so they are asserted only on that backend; on any other
# backend these parameter sets are empty and pytest collects the tests as skipped
# rather than claiming an unprobed restriction. float64/int64/fp8 operands can only
# be constructed where the device supports those dtypes, so those rows are
# additionally gated on the static capability flags.
DTYPE_REJECTION_MEASURED = getattr(flag_gems, "vendor_name", "") == "nvidia"

UNSUPPORTED_DTYPES = (
    (
        [
            torch.float32,
            torch.int32,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]
        + ([torch.float64] if utils.fp64_is_supported else [])
        + ([torch.int64] if utils.int64_is_supported else [])
        + ([torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else [])
    )
    if DTYPE_REJECTION_MEASURED
    else []
)

UNSUPPORTED_MASK_DTYPES = (
    (
        [torch.float32, torch.int32, torch.int8, torch.bool]
        + ([torch.int64] if utils.int64_is_supported else [])
    )
    if DTYPE_REJECTION_MEASURED
    else []
)

BAD_MASK_SHAPES = [(4, 8, 4), (4, 4, 8), (8, 8, 8), (4, 16, 8)]

# The native wrapper reads size(1) before reaching its own rank assertion, so rank
# 0 and rank 1 fail with IndexError while rank >= 3 fails the dim() check.
BAD_RANKS = [
    ((), IndexError),
    ((64,), IndexError),
    ((1, 32, 64), RuntimeError),
    ((1, 1, 32, 64), RuntimeError),
    ((1, 1, 1, 32, 64), RuntimeError),
]

# M % 32 != 0 (the derived mask row count no longer matches the launch grid), K % 64
# != 0 ('Wrong alignment shape[1]') and zero extents.
BAD_TILES = [
    (8, 64),
    (16, 64),
    (24, 64),
    (48, 64),
    (32, 32),
    (32, 65),
    (64, 8),
    (64, 16),
    (64, 24),
    (64, 40),
    (64, 48),
    (64, 56),
    (64, 72),
    (0, 64),
    (32, 0),
]

MASK_LAYOUT_KINDS = ["last_dim_step2", "transpose_view", "col_slice"]


def _thread_masks(shape, pattern):
    """Build the per-thread byte mask the kernel expects for an ``(M, K)`` tile."""
    rows, cols = shape
    mask_shape = (rows // 8, cols // 8, 8)
    if pattern == "random":
        return torch.randint(
            0, 256, mask_shape, dtype=torch.uint8, device=flag_gems.device
        )
    if pattern == "lane_bits":
        lanes = 1 << torch.arange(8, dtype=torch.uint8, device=flag_gems.device)
        return lanes.repeat(mask_shape[0], mask_shape[1], 1)
    if pattern == "row_ramp":
        ramp = 17 * torch.arange(
            mask_shape[0], dtype=torch.uint8, device=flag_gems.device
        )
        return ramp.view(mask_shape[0], 1, 1).expand(mask_shape).contiguous()
    return torch.full(mask_shape, pattern, dtype=torch.uint8, device=flag_gems.device)


def _strided_tile(base, layout):
    """Return the measured non-contiguous tile described in ``STRIDED_LAYOUTS``."""
    if layout == "row_step2":
        return base[::2]
    if layout == "col_step2":
        return base[:, ::2]
    if layout == "row_offset":
        return base[8:]
    if layout == "transpose":
        return base.transpose(0, 1)
    raise ValueError(f"unknown layout {layout}")


def _strided_masks(shape, layout):
    """Mask layouts the native stride contract accepts for an ``(M, K)`` tile."""
    rows, cols = shape
    if layout == "row_step2":
        wide = torch.randint(
            0,
            256,
            (2 * rows // 8, cols // 8, 8),
            dtype=torch.uint8,
            device=flag_gems.device,
        )
        return wide[::2]
    if layout == "offset":
        padded = torch.randint(
            0,
            256,
            (rows // 8 + 4, cols // 8, 8),
            dtype=torch.uint8,
            device=flag_gems.device,
        )
        return padded[2 : 2 + rows // 8]
    raise ValueError(f"unknown mask layout {layout}")


def _bad_thread_masks(kind):
    """Mask layouts the native stride contract rejects for a (32, 64) tile."""
    if kind == "last_dim_step2":
        wide = torch.randint(
            0, 256, (4, 8, 16), dtype=torch.uint8, device=flag_gems.device
        )
        return wide[..., ::2]
    if kind == "transpose_view":
        return (
            torch.zeros((4, 8, 8), dtype=torch.uint8, device=flag_gems.device)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
        )
    if kind == "col_slice":
        return torch.zeros((4, 8, 8), dtype=torch.uint8, device=flag_gems.device)[:, :4]
    raise ValueError(f"unknown mask layout {kind}")


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("shape", TILE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_semi_structured_apply_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    masks = _thread_masks(shape, 0x0F)
    ref_masks = tu.to_reference(masks)

    ref_out0, ref_out1 = torch.ops.aten._sparse_semi_structured_apply(
        ref_inp, ref_masks
    )
    res_out0, res_out1 = flag_gems._sparse_semi_structured_apply(inp, masks)

    tu.assert_result_equal(res_out0, ref_out0)
    tu.assert_result_equal(res_out1, ref_out1)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("shape", MASK_SHAPES)
@pytest.mark.parametrize("pattern", MASK_PATTERNS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_semi_structured_apply_mask_patterns(shape, pattern, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    masks = _thread_masks(shape, pattern)
    ref_masks = tu.to_reference(masks)
    masks_before = masks.clone()

    ref_out0, ref_out1 = torch.ops.aten._sparse_semi_structured_apply(
        ref_inp, ref_masks
    )
    res_out0, res_out1 = flag_gems._sparse_semi_structured_apply(inp, masks)

    tu.assert_result_equal(res_out0, ref_out0)
    tu.assert_result_equal(res_out1, ref_out1)
    # thread_masks is a read-only operand: its bytes must survive the call.
    tu.assert_result_equal(masks, masks_before)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("layout", tu.selected_cases(["row_step2", "offset"], quick=[]))
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_semi_structured_apply_mask_layout(layout, dtype):
    shape = MASK_LAYOUT_SHAPE
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    masks = _strided_masks(shape, layout)
    ref_masks = tu.to_reference(masks)

    ref_out0, ref_out1 = torch.ops.aten._sparse_semi_structured_apply(
        ref_inp, ref_masks
    )
    res_out0, res_out1 = flag_gems._sparse_semi_structured_apply(inp, masks)

    tu.assert_result_equal(res_out0, ref_out0)
    tu.assert_result_equal(res_out1, ref_out1)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("layout,base_shape", STRIDED_LAYOUTS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_semi_structured_apply_strided_input(layout, base_shape, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    inp = _strided_tile(base, layout)
    ref_inp = tu.to_reference(inp)
    masks = _thread_masks(inp.shape, 0x0F)
    ref_masks = tu.to_reference(masks)

    ref_out0, ref_out1 = torch.ops.aten._sparse_semi_structured_apply(
        ref_inp, ref_masks
    )
    res_out0, res_out1 = flag_gems._sparse_semi_structured_apply(inp, masks)

    tu.assert_result_equal(res_out0, ref_out0)
    tu.assert_result_equal(res_out1, ref_out1)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("dtype,scenario", SPECIAL_VALUES)
def test__sparse_semi_structured_apply_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    count = 32 * 64
    inp = payload.repeat(count // payload.numel() + 1)[:count].reshape(32, 64)
    ref_inp = tu.to_reference(inp)
    masks = _thread_masks(inp.shape, 0x55)
    ref_masks = tu.to_reference(masks)

    ref_out0, ref_out1 = torch.ops.aten._sparse_semi_structured_apply(
        ref_inp, ref_masks
    )
    res_out0, res_out1 = flag_gems._sparse_semi_structured_apply(inp, masks)

    tu.assert_result_equal(res_out0, ref_out0)
    tu.assert_result_equal(res_out1, ref_out1)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test__sparse_semi_structured_apply_rejects_unsupported_dtype(dtype):
    inp = torch.zeros((32, 64), dtype=dtype, device=flag_gems.device)
    masks = _thread_masks((32, 64), 0x0F)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply(inp, masks)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("dtype", UNSUPPORTED_MASK_DTYPES)
def test__sparse_semi_structured_apply_rejects_mask_dtype(dtype):
    inp = torch.zeros((32, 64), dtype=torch.float16, device=flag_gems.device)
    masks = torch.zeros((4, 8, 8), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply(inp, masks)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("mask_shape", BAD_MASK_SHAPES)
def test__sparse_semi_structured_apply_rejects_mask_shape(mask_shape):
    inp = torch.zeros((32, 64), dtype=torch.float16, device=flag_gems.device)
    masks = torch.zeros(mask_shape, dtype=torch.uint8, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply(inp, masks)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("shape,error", BAD_RANKS)
def test__sparse_semi_structured_apply_rejects_rank(shape, error):
    inp = torch.zeros(shape, dtype=torch.float16, device=flag_gems.device)
    masks = _thread_masks((32, 64), 0x0F)
    with pytest.raises(error):
        flag_gems._sparse_semi_structured_apply(inp, masks)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("shape", BAD_TILES)
def test__sparse_semi_structured_apply_rejects_bad_tile(shape):
    inp = torch.zeros(shape, dtype=torch.float16, device=flag_gems.device)
    masks = _thread_masks(shape, 0x0F)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply(inp, masks)


@pytest.mark._sparse_semi_structured_apply
def test__sparse_semi_structured_apply_rejects_misstrided_input():
    # A column slice of a 324-wide tensor is row-major with stride(0) == 324, and a
    # row-major tile requires input.stride(0) % 8 == 0.
    inp = tu.make_input(torch.float16, (64, 324), ["-1", "1"])[:, :64]
    masks = _thread_masks((64, 64), 0x0F)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply(inp, masks)


@pytest.mark._sparse_semi_structured_apply
@pytest.mark.parametrize("kind", MASK_LAYOUT_KINDS)
def test__sparse_semi_structured_apply_rejects_misstrided_masks(kind):
    inp = torch.zeros((32, 64), dtype=torch.float16, device=flag_gems.device)
    masks = _bad_thread_masks(kind)
    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_apply(inp, masks)
