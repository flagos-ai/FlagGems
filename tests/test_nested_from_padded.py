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
from . import conftest as cfg

# Padded shapes plus the per-component extents fed to the operator. Every case
# keeps the last extent equal to the padded trailing dim (except the explicit
# "trailing_trim" cases), because the nested components must agree on their
# trailing shapes for the result to be a valid nested tensor.
# The ragged dimension is dim 1 by definition of the operator, so the shapes
# differ only in the number of trailing dims, the padded length and the batch.
NON_FUSED_CASES = [
    # 2D padded: no trailing dims at all, lengths vary and one component empty.
    ((3, 5), [[0], [5], [3]]),
    ((4, 8), [[8], [1], [4], [6]]),
    # 3D padded: the canonical (B, T, D) transformer layout.
    ((2, 8, 4), [[3, 4], [8, 4]]),
    ((3, 5, 7), [[2, 7], [5, 7], [1, 7]]),
    # Trailing extents trimmed below the padded size: only the prefix box counts.
    ((2, 6, 5), [[3, 2], [6, 4]]),
    # 4D padded: two trailing dims, both trimmed independently.
    ((2, 4, 3, 5), [[2, 3, 5], [4, 1, 2]]),
    # 5D padded: checks the general N-dimensional index arithmetic.
    ((2, 3, 2, 2, 3), [[1, 2, 2, 3], [3, 1, 2, 2]]),
    # A component whose ragged length fills the padded dim (max-length path).
    ((2, 16, 3), [[16, 3], [16, 3]]),
]


# ``fuse_transform_0213`` permutes the padded tensor to ``[B, E, T * C]``, so the
# fused layout is only well defined when the component row length equals the
# padded ``T * C`` (the standard attention usage): the CUDA reference derives its
# channel step from an integer division by the padded dim-1 size and reads out of
# bounds for trimmed rows, while the CPU generic path applies a prefix box.
# Cases therefore keep row_len == T * C; E and C vary to cover the
# (0<->2)/(1<->3) swap.
FUSED_CASES = [
    ((2, 2, 2, 4), [[2, 8], [1, 8]]),
    ((1, 3, 3, 5), [[3, 15]]),
    ((2, 4, 4, 2), [[2, 8], [4, 8]]),
    ((3, 2, 2, 3), [[2, 6], [1, 6], [2, 6]]),
]


def _assert_nested_equal(res_out, ref_out, dtype):
    """Compare every observable piece of the legacy nested layout."""
    assert res_out.is_nested and ref_out.is_nested
    assert (
        torch.ops.aten._nested_tensor_size(res_out).tolist()
        == torch.ops.aten._nested_tensor_size(ref_out).tolist()
    ), "component sizes differ"
    assert (
        torch.ops.aten._nested_tensor_strides(res_out).tolist()
        == torch.ops.aten._nested_tensor_strides(ref_out).tolist()
    ), "component strides differ"
    assert (
        torch.ops.aten._nested_tensor_storage_offsets(res_out).tolist()
        == torch.ops.aten._nested_tensor_storage_offsets(ref_out).tolist()
    ), "component storage offsets differ"
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    for res_comp, ref_comp in zip(res_out.unbind(), ref_out.unbind()):
        utils.gems_assert_close(res_comp, ref_comp, dtype)


def _make_sizes(size_rows):
    # ``cpu_nested_shape_example`` is an int64 CPU tensor by definition.
    return torch.tensor(size_rows, dtype=torch.int64)


@pytest.mark.nested_from_padded
@pytest.mark.parametrize("shape,size_rows", NON_FUSED_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded(shape, size_rows, dtype):
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    sizes = _make_sizes(size_rows)

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded
@pytest.mark.skipif(
    cfg.TO_CPU,
    reason=(
        "fuse_transform_0213 only runs on CUDA; the CPU reference falls back to "
        "the generic masked_select path, whose component layout differs"
    ),
)
@pytest.mark.parametrize("shape,size_rows", FUSED_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_fuse_transform_0213(shape, size_rows, dtype):
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    sizes = _make_sizes(size_rows)

    ref_out = torch.ops.aten._nested_from_padded(padded, sizes.clone(), True)
    res_out = flag_gems._nested_from_padded(padded, sizes, True)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_from_padded
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_nested_from_padded_int_dtypes(dtype):
    # Integer inputs follow the same index path; the CUDA generic fallback in the
    # reference is only exercised for floats, so cover ints explicitly here.
    padded = torch.randint(-100, 100, (2, 5, 3), dtype=dtype, device=flag_gems.device)
    sizes = _make_sizes([[2, 3], [5, 1]])

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    assert (
        torch.ops.aten._nested_tensor_size(res_out).tolist()
        == torch.ops.aten._nested_tensor_size(ref_out).tolist()
    )
    utils.gems_assert_equal(res_out.values(), ref_out.values())


@pytest.mark.nested_from_padded
def test_nested_from_padded_non_contiguous_input():
    # The operator makes its input contiguous internally; verify that a strided
    # view of the same values produces the same result as the reference.
    base = torch.randn((2, 5, 6), dtype=torch.float32, device=flag_gems.device)
    padded = base.transpose(1, 2)
    sizes = _make_sizes([[2, 5], [6, 3]])

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    _assert_nested_equal(res_out, ref_out, torch.float32)


@pytest.mark.nested_from_padded
def test_nested_from_padded_all_padding():
    # Every component can be empty (all padding); the values buffer is then empty.
    padded = torch.randn((2, 4, 3), dtype=torch.float32, device=flag_gems.device)
    sizes = _make_sizes([[0, 3], [0, 3]])

    ref_out = torch.ops.aten._nested_from_padded(
        utils.to_reference(padded), sizes.clone()
    )
    res_out = flag_gems._nested_from_padded(padded, sizes)

    assert res_out.values().numel() == ref_out.values().numel() == 0
    _assert_nested_equal(res_out, ref_out, torch.float32)


# Backward coverage. The operator is a pure copy, so every backward case only
# needs the flat values-layout gradient of the nested result plus the shape
# example; the padded tensor itself is rebuilt from its recorded shape.
# The legacy nested layout has no CPU autograd: on CPU `_nested_from_padded`
# produces a tensor without a grad_fn, so the reference side of every backward
# case cannot build a graph. The device paths that do support it are covered by
# the CUDA run; the CPU reference run skips these instead of comparing against
# an impossible baseline.
requires_nested_autograd = pytest.mark.skipif(
    cfg.TO_CPU,
    reason=(
        "legacy nested autograd is CUDA-only: the CPU reference produces no "
        "grad_fn, so a backward comparison is not defined"
    ),
)

BACKWARD_CASES = [
    # 2D padded: no trailing dims at all, lengths vary and one component empty.
    ((3, 5), [[0], [5], [3]], False),
    ((4, 8), [[8], [1], [4], [6]], False),
    # 3D padded: the canonical (B, T, D) transformer layout.
    ((2, 8, 4), [[3, 4], [8, 4]], False),
    # Trailing extents trimmed below the padded size: the general scatter path,
    # where a component's prefix box is not a contiguous run of the padded row.
    ((2, 6, 5), [[3, 2], [6, 4]], False),
    ((2, 4, 3, 5), [[2, 3, 5], [4, 1, 2]], False),
    ((2, 3, 2, 2, 3), [[1, 2, 2, 3], [3, 1, 2, 2]], False),
    # A component whose ragged length fills the padded dim (max-length path).
    ((2, 16, 3), [[16, 3], [16, 3]], False),
    # fuse_transform_0213: the forward reads the (0, 2, 1, 3) permuted view.
    ((2, 2, 2, 4), [[2, 8], [1, 8]], True),
    ((2, 4, 4, 2), [[2, 8], [4, 8]], True),
    ((3, 2, 2, 3), [[2, 6], [1, 6], [2, 6]], True),
]


def _grad_values_for(out, seed):
    generator = torch.Generator(device=out.values().device).manual_seed(seed)
    return torch.randn(
        out.values().shape,
        generator=generator,
        dtype=out.dtype,
        device=out.values().device,
    )


def _nested_grad(out, grad_values):
    # A nested-typed upstream gradient reaching the nested output directly
    # rather than through .values(); same values, strided-layout wrapper.
    return torch.ops.aten._nested_view_from_buffer(
        grad_values,
        torch.ops.aten._nested_tensor_size(out),
        torch.ops.aten._nested_tensor_strides(out),
        torch.ops.aten._nested_tensor_storage_offsets(out),
    )


@pytest.mark.nested_from_padded_backward
@pytest.mark.parametrize("shape,size_rows,fuse", BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@requires_nested_autograd
def test_nested_from_padded_backward(shape, size_rows, fuse, dtype):
    sizes = _make_sizes(size_rows)
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_padded = utils.to_reference(padded.detach(), True).requires_grad_()
    gem_padded = padded.detach().clone().requires_grad_()

    ref_out = torch.ops.aten._nested_from_padded(ref_padded, sizes.clone(), fuse)
    gem_out = flag_gems._nested_from_padded(gem_padded, sizes.clone(), fuse)
    _assert_nested_equal(gem_out, ref_out, dtype)

    grad_values = _grad_values_for(ref_out, 11)
    (ref_grad,) = torch.autograd.grad(ref_out.values(), ref_padded, grad_values)
    (gem_grad,) = torch.autograd.grad(
        gem_out.values(), gem_padded, utils.to_reference(grad_values, cfg.TO_CPU)
    )
    utils.gems_assert_close(gem_grad, ref_grad, dtype)

    # Padding positions must be exactly zero, and every prefix box must hold
    # exactly its slice of the upstream gradient: this pins the un-padding
    # semantics rather than only comparing two paths that could share a bug.
    # The expected dense gradient is rebuilt directly from the recorded extents,
    # in the layout the forward actually read: with fuse_transform_0213 the
    # forward copies from the (0, 2, 1, 3)-permuted [B, E, T*C] view, so the
    # reference is built there and permuted back.
    if fuse:
        batch, length, inner, channels = shape
        read_shape = (batch, inner, length * channels)
    else:
        read_shape = shape
    read_grad = torch.zeros(read_shape, dtype=gem_grad.dtype, device=gem_grad.device)
    offset = 0
    for index, row in enumerate(size_rows):
        count = 1
        for extent in row:
            count *= extent
        read_grad[(index,) + tuple(slice(0, e) for e in row)] = grad_values[
            offset : offset + count
        ].reshape(row)
        offset += count
    assert offset == grad_values.numel()
    if fuse:
        batch, length, inner, channels = shape
        expected = read_grad.view(batch, inner, length, channels).permute(0, 2, 1, 3)
    else:
        expected = read_grad
    utils.gems_assert_close(gem_grad, expected, dtype)


@pytest.mark.nested_from_padded_backward
@pytest.mark.parametrize("shape,size_rows,fuse", BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@requires_nested_autograd
def test_nested_from_padded_backward_nested_grad(shape, size_rows, fuse, dtype):
    """The gradient also flows when the nested output itself receives the grad."""
    sizes = _make_sizes(size_rows)
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_padded = utils.to_reference(padded.detach(), True).requires_grad_()
    gem_padded = padded.detach().clone().requires_grad_()

    ref_out = torch.ops.aten._nested_from_padded(ref_padded, sizes.clone(), fuse)
    gem_out = flag_gems._nested_from_padded(gem_padded, sizes.clone(), fuse)
    grad_values = _grad_values_for(ref_out, 13)

    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_padded, _nested_grad(ref_out, grad_values.clone())
    )
    (gem_grad,) = torch.autograd.grad(
        gem_out,
        gem_padded,
        _nested_grad(ref_out, utils.to_reference(grad_values, cfg.TO_CPU)),
    )
    utils.gems_assert_close(gem_grad, ref_grad, dtype)


@pytest.mark.nested_from_padded_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@requires_nested_autograd
def test_nested_from_padded_backward_empty_components(dtype):
    """Every component empty: the whole padded gradient is zero."""
    shape, size_rows = (2, 4, 3), [[0, 3], [0, 3]]
    sizes = _make_sizes(size_rows)
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    gem_padded = padded.detach().clone().requires_grad_()

    gem_out = flag_gems._nested_from_padded(gem_padded, sizes.clone())
    assert gem_out.values().numel() == 0
    (gem_grad,) = torch.autograd.grad(gem_out.values(), gem_padded, gem_out.values())
    assert tuple(gem_grad.shape) == shape
    assert torch.equal(
        gem_grad, torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    )


@pytest.mark.nested_from_padded_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@requires_nested_autograd
def test_nested_from_padded_backward_single_kernel(dtype):
    """The backward launches one kernel, not a zero-fill plus one per component."""
    from torch.profiler import ProfilerActivity, profile

    shape, size_rows = (16, 32, 8), [[8, 8], [32, 8]] * 8
    sizes = _make_sizes(size_rows)
    padded = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    out = flag_gems._nested_from_padded(padded, sizes.clone())
    grad_values = _grad_values_for(out, 17)

    def run():
        torch.autograd.grad(out.values(), padded, grad_values, retain_graph=True)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()

    kernels = [
        event
        for event in prof.key_averages()
        if event.self_device_time_total > 0 and "nested_from_padded" in event.key
    ]
    assert len(kernels) == 1, [event.key for event in prof.key_averages()]
    # And nothing else on the device: no separate zero-fill or per-component copy.
    device_ops = [
        event for event in prof.key_averages() if event.self_device_time_total > 0
    ]
    assert len(device_ops) == 1, [event.key for event in device_ops]


@pytest.mark.nested_from_padded_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@requires_nested_autograd
def test_nested_from_padded_backward_matches_aten_bitwise(dtype):
    """The scatter is exact, so both sides agree bit for bit."""
    shape, size_rows = (8, 24, 16), [[11, 16], [24, 16], [3, 16], [7, 16]] * 2
    sizes = _make_sizes(size_rows)
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_padded = padded.detach().clone().requires_grad_()
    gem_padded = padded.detach().clone().requires_grad_()

    ref_out = torch.ops.aten._nested_from_padded(ref_padded, sizes.clone())
    gem_out = flag_gems._nested_from_padded(gem_padded, sizes.clone())
    grad_values = _grad_values_for(ref_out, 19)

    (ref_grad,) = torch.autograd.grad(ref_out.values(), ref_padded, grad_values)
    (gem_grad,) = torch.autograd.grad(gem_out.values(), gem_padded, grad_values)
    torch.testing.assert_close(gem_grad, ref_grad, atol=0, rtol=0)


@pytest.mark.nested_from_padded_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@requires_nested_autograd
def test_nested_from_padded_backward_grad_fn(dtype):
    """The custom node is in the graph, and inference keeps the forward path."""
    sizes = _make_sizes([[3, 4], [2, 4]])
    padded = torch.randn((2, 5, 4), dtype=dtype, device=flag_gems.device)
    grad_padded = padded.detach().clone().requires_grad_()
    out = flag_gems._nested_from_padded(grad_padded, sizes.clone())

    # _nested_view_from_buffer is a view op with its own generated node; our
    # wrapper sits one step below it and owns the un-padding.
    node = out.values().grad_fn
    seen = []
    while node is not None:
        seen.append(type(node).__name__)
        node = next((n[0] for n in node.next_functions if n[0] is not None), None)
    assert "_NestedFromPaddedBackwardBackward" in seen, seen

    # No-grad inference must not build the extra node at all.
    with torch.no_grad():
        plain = flag_gems._nested_from_padded(padded, sizes.clone())
    assert plain.grad_fn is None
    assert plain.values().grad_fn is None
