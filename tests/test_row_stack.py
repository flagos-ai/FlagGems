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

# Core shapes exercised by ROW_STACK_SHAPES (mirrors worktree CI branch).
ROW_STACK_SHAPES = [
    [(3,), (3,)],
    [(3, 33), (7, 33)],
    [(13, 3, 333), (17, 3, 333), (7, 3, 333)],
    [
        (13, 3, 64, 5, 2),
        (16, 3, 64, 5, 2),
        (7, 3, 64, 5, 2),
        (4, 3, 64, 5, 2),
        (1, 3, 64, 5, 2),
    ],
]

# Shapes whose trailing dimensions are zero-sized. numel() == 0 while stride(0)
# can still be non-zero (e.g. (3, 0) has stride (1, 1)), so a kernel launch
# sized from ``rows * stride(0)`` would read and write out of bounds.
ZERO_ROW_SHAPES = [
    [(3, 0), (2, 0)],
    [(3, 0, 4), (2, 0, 4)],
    [(2, 3, 0, 5), (1, 3, 0, 5)],
    [(0, 4), (2, 4)],
    [(0, 3, 4, 5), (0, 3, 4, 5)],
]

# (shape, memory format) pairs for the memory-format parity tests. Row_stack
# derives its output layout from its inputs through ATen's
# cat_compute_output_memory_format, so channels_last / channels_last_3d inputs
# must yield a channels-last output.
MEMORY_FORMAT_CASES = [
    ((2, 3, 4, 5), torch.channels_last),
    ((2, 3, 4, 5), torch.contiguous_format),
    ((2, 3, 4, 5, 6), torch.channels_last_3d),
    ((2, 3, 4, 5, 6), torch.contiguous_format),
]


def _make_inputs(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        return [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        return [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]


@pytest.mark.row_stack
@pytest.mark.parametrize("shape", ROW_STACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_row_stack(shape, dtype):
    inp = _make_inputs(shape, dtype)
    ref_inp = [utils.to_reference(e) for e in inp]
    # Reference via the public aten op on the reference (CPU) inputs.
    ref_out = torch.row_stack(ref_inp)
    # GEMS direct call: the kernel vertically stacks on the accelerator.
    res_out = flag_gems.row_stack(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_stack
@pytest.mark.parametrize("shape", ZERO_ROW_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_row_stack_zero_sized_trailing_dims(shape, dtype):
    """Zero-sized trailing dims must not launch an out-of-bounds kernel.

    Beyond correctness, we assert full metadata parity with ATen because
    ``stride(0)`` of inputs like ``(3, 0)`` is non-zero despite ``numel() == 0``,
    so a launch sized from ``rows * stride(0)`` would run out of bounds.
    """
    inp = _make_inputs(shape, dtype)
    ref_inp = [utils.to_reference(e) for e in inp]

    ref_out = torch.row_stack(ref_inp)
    res_out = flag_gems.row_stack(inp)

    assert res_out.numel() == ref_out.numel()
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_stack
@pytest.mark.parametrize("shape,memory_format", MEMORY_FORMAT_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_row_stack_memory_format(shape, memory_format, dtype):
    """The output must keep the inputs' channels-last memory format."""
    inp = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device).to(
            memory_format=memory_format
        )
        for _ in range(2)
    ]
    ref_inp = [utils.to_reference(e) for e in inp]

    ref_out = torch.row_stack(ref_inp)
    res_out = flag_gems.row_stack(inp)

    # ATen derives the output memory format from its inputs; the strides must
    # match exactly, which is how channels-last is observed from the outside.
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    if memory_format != torch.contiguous_format:
        assert res_out.is_contiguous(memory_format=memory_format)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_stack
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_row_stack_mixed_dtype_promotion(dtype):
    """Mixed dtypes follow ATen's type promotion instead of being rejected."""
    other = torch.float64 if dtype == torch.float32 else torch.int64
    if dtype in utils.FLOAT_DTYPES:
        a = torch.randn(3, 4, dtype=dtype, device=flag_gems.device)
        b = torch.randn(2, 4, dtype=other, device=flag_gems.device)
    else:
        a = torch.randint(0, 100, (3, 4), dtype=dtype, device=flag_gems.device)
        b = torch.randint(0, 100, (2, 4), dtype=other, device=flag_gems.device)
    ref_a = utils.to_reference(a)
    ref_b = utils.to_reference(b)

    ref_out = torch.row_stack([ref_a, ref_b])
    res_out = flag_gems.row_stack([a, b])

    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_stack_out
@pytest.mark.parametrize("shape", ROW_STACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_row_stack_out(shape, dtype):
    inp = _make_inputs(shape, dtype)
    ref_inp = [utils.to_reference(e) for e in inp]

    # The functional variant gives the expected output metadata (1-D inputs are
    # promoted to rows by atleast_2d, so the shape is not just the row sum).
    expected = torch.row_stack(ref_inp)

    # Use the native out= path as the reference so the out= contract itself is
    # checked, not only the values of the functional variant.
    ref_out = torch.empty_like(expected)
    ref_ret = torch.row_stack(ref_inp, out=ref_out)

    res_out = torch.empty(expected.shape, dtype=dtype, device=flag_gems.device)
    res_ret = flag_gems.row_stack_out(inp, out=res_out)

    # Both calls must return the tensor they were handed.
    assert ref_ret is ref_out
    assert res_ret is res_out
    assert res_ret.data_ptr() == res_out.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_stack_out
@pytest.mark.parametrize("shape,memory_format", MEMORY_FORMAT_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_row_stack_out_memory_format(shape, memory_format, dtype):
    """The out= path must reproduce ATen's memory-format handling.

    ATen keeps the out tensor's layout when it is already the derived format,
    and re-imposes the derived format when the shape had to be resized.
    """
    inp = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device).to(
            memory_format=memory_format
        )
        for _ in range(2)
    ]
    ref_inp = [utils.to_reference(e) for e in inp]
    expected = torch.row_stack(ref_inp)

    for out_shape in (expected.shape, tuple([7] + list(expected.shape[1:]))):
        ref_out = torch.empty(out_shape, dtype=dtype, device=ref_inp[0].device).to(
            memory_format=memory_format
        )
        if tuple(out_shape) != tuple(expected.shape):
            # ATen only resizes a wrong-shaped out without warning once it has
            # zero elements, so shrink the reference first; the GEMS call below
            # still receives the wrong-shaped tensor and must resize it itself.
            ref_out.resize_(0)
        torch.row_stack(ref_inp, out=ref_out)

        res_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device).to(
            memory_format=memory_format
        )
        ret = flag_gems.row_stack_out(inp, out=res_out)

        assert ret is res_out
        assert tuple(res_out.shape) == tuple(ref_out.shape)
        assert tuple(res_out.stride()) == tuple(ref_out.stride())
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_stack_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_row_stack_out_non_contiguous(dtype):
    """A correctly shaped but non-contiguous out must receive correct values.

    ``row_stack_out`` used to write ``out`` as flat storage, ignoring its
    strides, so a sliced destination silently kept stale values.
    """
    inp = [torch.randn((2, 3), dtype=dtype, device=flag_gems.device) for _ in range(2)]
    ref_inp = [utils.to_reference(e) for e in inp]
    ref_out = torch.row_stack(ref_inp)

    # Shape (4, 3) with stride (6, 1): a column slice of a (4, 6) buffer.
    buffer = torch.zeros((4, 6), dtype=dtype, device=flag_gems.device)
    out = buffer[:, 1:4]
    assert not out.is_contiguous()

    ret = flag_gems.row_stack_out(inp, out=out)

    assert ret is out
    assert tuple(out.stride()) == (6, 1)
    utils.gems_assert_equal(out, ref_out)
    # The destination's strides must be respected: the untouched columns stay 0.
    assert torch.count_nonzero(buffer[:, 0]) == 0
    assert torch.count_nonzero(buffer[:, 4:]) == 0


@pytest.mark.row_stack_out
@pytest.mark.parametrize("shape", [(10, 3), (1, 3), (0,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_row_stack_out_resize(shape, dtype):
    """A wrongly shaped out is resized in place and returned."""
    inp = [torch.randn((2, 3), dtype=dtype, device=flag_gems.device) for _ in range(2)]
    ref_inp = [utils.to_reference(e) for e in inp]
    ref_out = torch.row_stack(ref_inp)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ret = flag_gems.row_stack_out(inp, out=out)

    assert ret is out
    assert tuple(out.shape) == tuple(ref_out.shape)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.row_stack_out
def test_row_stack_out_dtype_cast():
    """ATen casts the result into a different (castable) out dtype."""
    inp = [torch.randn((2, 3), device=flag_gems.device) for _ in range(2)]
    ref_inp = [utils.to_reference(e) for e in inp]
    ref_out = torch.empty((4, 3), dtype=torch.float64, device=ref_inp[0].device)
    torch.row_stack(ref_inp, out=ref_out)

    out = torch.empty((4, 3), dtype=torch.float64, device=flag_gems.device)
    ret = flag_gems.row_stack_out(inp, out=out)

    assert ret is out
    assert out.dtype == torch.float64
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.row_stack_out
def test_row_stack_out_non_castable_dtype_raises():
    """Matching ATen, an impossible cast raises TypeError."""
    inp = [torch.randn((2, 3), device=flag_gems.device) for _ in range(2)]
    out = torch.empty((4, 3), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(TypeError):
        flag_gems.row_stack_out(inp, out=out)


@pytest.mark.row_stack_out
def test_row_stack_out_device_mismatch_raises():
    inp = [
        torch.randn((2, 3), device=flag_gems.device),
        torch.randn((2, 3), device="cpu"),
    ]
    out = torch.empty((4, 3), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.row_stack_out(inp, out=out)


@pytest.mark.row_stack_out
def test_row_stack_out_overlapping_input_raises():
    """An out that aliases an input must be rejected, as ATen does."""
    inp = torch.randn((4, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.row_stack_out([inp[:2], inp[2:]], out=inp)


@pytest.mark.row_stack
def test_row_stack_shape_mismatch_raises():
    with pytest.raises(RuntimeError):
        flag_gems.row_stack(
            [
                torch.randn((2, 3), device=flag_gems.device),
                torch.randn((2, 4), device=flag_gems.device),
            ]
        )


@pytest.mark.row_stack
def test_row_stack_empty_list_raises():
    with pytest.raises(RuntimeError):
        flag_gems.row_stack([])
