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
"""Tests for the gradient of ``torch.chunk``.

``chunk`` has no ``aten::chunk_backward`` schema, so there is nothing to call
directly: every gradient test drives the full autograd chain through
``flag_gems.chunk`` and compares against ``torch.chunk``'s own autograd. The
Triton kernel is reachable only that way (see ``ChunkFunction`` in
``flag_gems.ops.chunk``), which is what these tests pin down.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes covering 1-D through 4-D and a size that does not divide evenly by the
# chunk count.
CHUNK_SHAPES = [
    (64,),
    (128, 64),
    (1024, 1024),
    (64, 512, 512),
    (7, 33, 5),
    (5, 6, 7, 8),
]


@pytest.mark.chunk_backward
@pytest.mark.parametrize("shape", CHUNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("chunks", [1, 2, 3, 7])
@pytest.mark.parametrize("dim", [0, -1])
def test_chunk_backward(shape, dtype, chunks, dim):
    """chunk's backward is a concatenation of the per-chunk gradients."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    res_inp = inp.clone().requires_grad_(True)

    ref_chunks = torch.chunk(ref_inp, chunks, dim)
    res_chunks = flag_gems.chunk(res_inp, chunks, dim)
    assert len(ref_chunks) == len(res_chunks)

    res_grads = [
        torch.randn(c.shape, dtype=dtype, device=flag_gems.device) for c in res_chunks
    ]
    ref_grads = [utils.to_reference(g, True) for g in res_grads]

    (ref_in_grad,) = torch.autograd.grad(ref_chunks, ref_inp, ref_grads)
    (res_in_grad,) = torch.autograd.grad(res_chunks, res_inp, res_grads)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
def test_chunk_backward_expanded_grad(dtype, dim):
    """An expanded upstream gradient must not be read as a dense buffer.

    ``expand`` produces a stride-0 view, so reading it linearly as the chunk's
    ``d_i * inner`` run would silently pick the wrong elements.
    """
    shape = (6, 9)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_(True)

    res_chunks = flag_gems.chunk(inp, 3, dim)
    ref_chunks = torch.chunk(ref_inp, 3, dim)

    res_grads = []
    for chunk in res_chunks:
        # A size-1 axis along ``dim`` expanded to the chunk's shape: stride 0
        # there, so the buffer is not dense.
        base_shape = tuple(1 if i == dim else s for i, s in enumerate(chunk.shape))
        res_grads.append(
            torch.randn(base_shape, dtype=dtype, device=flag_gems.device).expand(
                chunk.shape
            )
        )
    ref_grads = [utils.to_reference(g) for g in res_grads]
    assert all(not g.is_contiguous() for g in res_grads)

    (dx,) = torch.autograd.grad(res_chunks, inp, res_grads)
    (ref_dx,) = torch.autograd.grad(ref_chunks, ref_inp, ref_grads)
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("slice_step", [1, 2])
def test_chunk_backward_strided_grad(dtype, slice_step):
    """A non-contiguous (transposed / stepped) grad must be read element-wise."""
    shape = (12, 6)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_(True)

    res_chunks = flag_gems.chunk(inp, 3, 0)
    ref_chunks = torch.chunk(ref_inp, 3, 0)

    # Build each grad from a contiguous buffer whose *inner* axis is the one
    # sliced/transposed, so the leading (split) axis is untouched: the result
    # has the chunk's shape but a non-dense layout. A linear read of it would
    # therefore pick the wrong elements.
    res_grads = []
    for chunk in res_chunks:
        rows, cols = chunk.shape
        source = torch.randn((cols, rows), dtype=dtype, device=flag_gems.device)
        if slice_step != 1:
            source = torch.randn(
                (cols, rows * slice_step), dtype=dtype, device=flag_gems.device
            )
            source = source[:, ::slice_step]
        res_grads.append(source.t())
    assert all(not g.is_contiguous() for g in res_grads)
    assert all(g.shape == c.shape for g, c in zip(res_grads, res_chunks))

    (dx,) = torch.autograd.grad(res_chunks, inp, res_grads)
    (ref_dx,) = torch.autograd.grad(
        ref_chunks, ref_inp, [utils.to_reference(g) for g in res_grads]
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_backward_unused_chunk(dtype):
    """An unused chunk contributes a ``None`` gradient, not a missing slab.

    The offsets of the remaining chunks depend on the full size list, so the
    ``None`` has to become an explicit zero block of the right shape.
    """
    shape = (9, 4)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()

    # Only the last chunk participates, so the first two arrive as None.
    (dx,) = torch.autograd.grad(flag_gems.chunk(inp, 3, 0)[2].sum(), inp)
    (ref_dx,) = torch.autograd.grad(torch.chunk(ref_inp, 3, 0)[2].sum(), ref_inp)
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("chunks", [2, 3, 5])
def test_chunk_backward_empty_dim(dtype, chunks):
    """A zero-length split axis yields empty chunks; the gradient is empty too."""
    inp = torch.zeros((0, 4), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_(True)

    res_chunks = flag_gems.chunk(inp, chunks, 0)
    ref_chunks = torch.chunk(ref_inp, chunks, 0)
    assert all(c.shape[0] == 0 for c in res_chunks)
    assert len(res_chunks) == len(ref_chunks)

    (dx,) = torch.autograd.grad(
        res_chunks, inp, [torch.zeros_like(c) for c in res_chunks]
    )
    (ref_dx,) = torch.autograd.grad(
        ref_chunks, ref_inp, [torch.zeros_like(c) for c in ref_chunks]
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_backward_more_chunks_than_elems(dtype):
    """``chunks > dim_size`` produces one-element pieces, matching ATen."""
    inp = torch.randn((4, 3), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()

    res_chunks = flag_gems.chunk(inp, 9, 0)
    ref_chunks = torch.chunk(ref_inp, 9, 0)
    assert len(res_chunks) == len(ref_chunks) == 4

    grads = [torch.randn_like(c) for c in res_chunks]
    (dx,) = torch.autograd.grad(res_chunks, inp, grads)
    (ref_dx,) = torch.autograd.grad(
        ref_chunks, ref_inp, [utils.to_reference(g) for g in grads]
    )
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_backward_accumulates(dtype):
    """Two backward passes through one chunk accumulate, as native autograd does."""
    inp = torch.randn((8, 5), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_(True)

    flag_gems.chunk(inp, 2, 0)[0].sum().backward()
    torch.chunk(ref_inp, 2, 0)[0].sum().backward()
    flag_gems.chunk(inp, 2, 0)[0].sum().backward()
    torch.chunk(ref_inp, 2, 0)[0].sum().backward()

    utils.gems_assert_close(inp.grad, ref_inp.grad, dtype)


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_forward_is_a_view(dtype):
    """The forward must stay a zero-copy view, aliasing the input's storage."""
    inp = torch.randn((6, 7), dtype=dtype, device=flag_gems.device)
    chunks = flag_gems.chunk(inp, 3, 0)
    ref_chunks = torch.chunk(inp, 3, 0)

    assert all(c._base is not None for c in chunks)
    # Every piece is a view of the *same* storage as the input, at the offset
    # ATen's own split would use.
    assert all(
        c.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
        for c in chunks
    )
    assert [c.storage_offset() for c in chunks] == [
        c.storage_offset() for c in ref_chunks
    ]
    assert [c.stride() for c in chunks] == [c.stride() for c in ref_chunks]


@pytest.mark.chunk_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape, dim",
    [((12,), 0), ((6, 4), 0), ((6, 9), 1), ((3, 4, 5), -1)],
)
def test_chunk_through_dispatch(dtype, shape, dim):
    """``torch.chunk`` itself must keep working under ``use_gems``.

    The Autograd registration is what makes this pass: a fallthrough there
    hands the call back to ATen's own chunk formula, which rebuilds the pieces
    as views of the *differentiable* input and trips PyTorch's "Attempted to
    make a tensor into a differentiable view" check.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = [torch.randn_like(c) for c in torch.chunk(inp, 3, dim)]
    res_inp = inp.clone().requires_grad_(True)

    with flag_gems.use_gems():
        res_chunks = torch.chunk(res_inp, 3, dim)
    (dx,) = torch.autograd.grad(res_chunks, res_inp, grad)

    ref_inp = utils.to_reference(inp.detach()).requires_grad_(True)
    (ref_dx,) = torch.autograd.grad(
        torch.chunk(ref_inp, 3, dim), ref_inp, [utils.to_reference(g) for g in grad]
    )
    utils.gems_assert_close(dx, ref_dx, dtype)
