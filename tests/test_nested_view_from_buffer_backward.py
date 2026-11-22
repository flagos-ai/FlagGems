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
from flag_gems.ops._nested_view_from_buffer_backward import (
    NestedViewFromBufferBackwardFunction,
)

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="CUDA-only op; no CPU reference implementation"
)

if cfg.QUICK_MODE:
    # --quick keeps the smoke run on a single dtype to stay fast; the full run
    # covers every float dtype through the shared constant.
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# Component layouts: each case is (buffer_size, sizes, strides, offsets).
# Case 1/2 are exactly packed buffers (the common strided NestedTensor), case 3
# carries a zero-size component, case 4 has inter-component gaps (uncovered
# storage must restore as zero gradient) and case 5 covers non-contiguous
# component strides. Small shapes keep the per-case cost comparable to the
# sibling nested backward tests while still exercising every kernel path
# (identity fast path, contiguous decode, general stride decode, fused and
# two-kernel variants across the C <= 64 threshold).
NESTED_VIEW_BACKWARD_CASES = [
    (13, [[3, 3], [1, 4]], [[3, 1], [4, 1]], [0, 9]),
    (8000, [[1000, 4], [1000, 4]], [[4, 1], [4, 1]], [0, 4000]),
    (16000, [[1000, 4], [0, 4], [3000, 4]], [[4, 1], [4, 1], [4, 1]], [0, 4000, 4000]),
    # buffer = last_offset + 8*4 = 80 + 32, with two gaps of length 8 in between
    # (components cover [0,32), [40,72), [80,112); gaps at 32..39 and 72..79)
    (112, [[8, 4], [8, 4], [8, 4]], [[4, 1], [4, 1], [4, 1]], [0, 40, 80]),
    # component 0 strided: rows at buffer positions {0,1,2, 8,9,10} (stride 8)
    (12, [[2, 3]], [[8, 1]], [0]),
]


def _eager_reference(grad_output, buffer, sizes, strides, offsets):
    """Restore the buffer gradient with a loop of as_strided copies.

    Component ``c`` occupies packed gradient elements
    ``[cum_numel[c], cum_numel[c] + numel_c)`` and is scattered into the
    buffer view ``as_strided(sizes[c], strides[c], offsets[c])``; positions
    no component covers stay zero.
    """
    out = torch.zeros_like(buffer)
    start = 0
    for c in range(sizes.shape[0]):
        comp_sizes = sizes[c].tolist()
        numel = 1
        for s in comp_sizes:
            numel *= s
        comp = grad_output[start : start + numel].view(comp_sizes)
        out.as_strided(comp_sizes, strides[c].tolist(), int(offsets[c])).copy_(comp)
        start += numel
    return out


def _make_case(case, dtype):
    buffer_size, sizes_data, strides_data, offsets_data = case
    total = sum(torch.tensor(s).prod().item() for s in sizes_data)
    grad_output = torch.randn(int(total), dtype=dtype, device=flag_gems.device)
    buffer = torch.zeros(buffer_size, dtype=dtype, device=flag_gems.device)
    sizes = torch.tensor(sizes_data, dtype=torch.int64)
    strides = torch.tensor(strides_data, dtype=torch.int64)
    offsets = torch.tensor(offsets_data, dtype=torch.int64)
    return grad_output, buffer, sizes, strides, offsets


@pytest.mark.nested_view_from_buffer_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("case", NESTED_VIEW_BACKWARD_CASES)
def test_nested_view_from_buffer_backward(case, dtype):
    grad_output, buffer, sizes, strides, offsets = _make_case(case, dtype)

    ref_out = _eager_reference(grad_output, buffer, sizes, strides, offsets)
    res_out = flag_gems._nested_view_from_buffer_backward(
        grad_output, buffer, sizes, strides, offsets
    )

    assert res_out.shape == buffer.shape
    assert res_out.dtype == dtype
    assert res_out.device == buffer.device
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.nested_view_from_buffer_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nested_view_from_buffer_backward_empty(dtype):
    # A zero-size packed gradient restores an all-zero buffer gradient.
    buffer = torch.zeros(64, dtype=dtype, device=flag_gems.device)
    sizes = torch.tensor([], dtype=torch.int64).reshape(0, 2)
    strides = torch.tensor([], dtype=torch.int64).reshape(0, 2)
    offsets = torch.tensor([], dtype=torch.int64)
    grad_output = torch.zeros(0, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems._nested_view_from_buffer_backward(
        grad_output, buffer, sizes, strides, offsets
    )
    assert res_out.shape == buffer.shape
    assert res_out.abs().sum().item() == 0.0


@pytest.mark.nested_view_from_buffer_backward
# float32 only: the autograd chain exercises metadata/autograd mechanics, not dtype
# coverage, which the parametrized direct tests above already provide.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_view_from_buffer_backward_autograd_chain(dtype):
    """End-to-end autograd: buffer -> nested view -> components -> backward.

    The reference restores the buffer gradient through native autograd view
    rebasing; the candidate packs the per-component gradients produced by
    ``torch.autograd.grad`` and restores the buffer with the Triton kernel.
    Both sides run with ``requires_grad=True`` inputs and ``autograd.grad``.
    """
    # Exactly packed layout: the native autograd engine requires the packed
    # component gradients to cover the buffer.
    buffer_size, sizes_data, strides_data, offsets_data = NESTED_VIEW_BACKWARD_CASES[1]
    sizes = torch.tensor(sizes_data, dtype=torch.int64)
    strides = torch.tensor(strides_data, dtype=torch.int64)
    offsets = torch.tensor(offsets_data, dtype=torch.int64)

    res_buffer = torch.randn(
        buffer_size, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_buffer = utils.to_reference(res_buffer).detach().requires_grad_(True)

    # Forward chains: the native view op needs host-resident metadata.
    res_nt = torch.ops.aten._nested_view_from_buffer.default(
        res_buffer, sizes, strides, offsets
    )
    ref_nt = torch.ops.aten._nested_view_from_buffer.default(
        ref_buffer, sizes, strides, offsets
    )

    res_c0, res_c1 = res_nt.unbind()
    ref_c0, ref_c1 = ref_nt.unbind()
    res_out = res_c0.pow(2).sum() + res_c1.sum()
    ref_out = ref_c0.pow(2).sum() + ref_c1.sum()
    # Same upstream gradient on both sides (scalar, broadcast by autograd).
    out_grad = torch.randn((), dtype=dtype, device=flag_gems.device)

    res_c0_grad, res_c1_grad = torch.autograd.grad(
        res_out, (res_c0, res_c1), out_grad, retain_graph=True
    )
    (res_buffer_grad,) = torch.autograd.grad(res_out, res_buffer, out_grad)

    ref_c0_grad, ref_c1_grad = torch.autograd.grad(
        ref_out, (ref_c0, ref_c1), utils.to_reference(out_grad), retain_graph=True
    )
    (ref_buffer_grad,) = torch.autograd.grad(
        ref_out, ref_buffer, utils.to_reference(out_grad)
    )

    utils.gems_assert_close(res_c0_grad, ref_c0_grad, dtype)
    utils.gems_assert_close(res_c1_grad, ref_c1_grad, dtype)
    utils.gems_assert_close(res_buffer_grad, ref_buffer_grad, dtype)

    # Restore the buffer gradient from the packed component gradients with the
    # FlagGems kernel and compare against the native autograd buffer gradient.
    packed = torch.cat(
        [res_c0_grad.contiguous().reshape(-1), res_c1_grad.contiguous().reshape(-1)]
    )
    restored = flag_gems._nested_view_from_buffer_backward(
        packed, res_buffer.detach(), sizes, strides, offsets
    )
    utils.gems_assert_close(restored, res_buffer_grad, dtype)


@pytest.mark.nested_view_from_buffer_backward
# float32 only: the bridge smoke covers autograd Function plumbing, not dtype
# coverage, which the parametrized direct tests above already provide.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_view_from_buffer_backward_bridge(dtype):
    """The autograd.Function bridge drives the Triton restore under autograd."""
    grad_output, buffer, sizes, strides, offsets = _make_case(
        NESTED_VIEW_BACKWARD_CASES[0], dtype
    )
    # requires_grad=True so autograd records the Function node (the restore
    # output itself is a gradient tensor flowing through autograd).
    grad_output = grad_output.detach().requires_grad_(True)

    res_out = NestedViewFromBufferBackwardFunction.apply(
        grad_output, buffer, sizes, strides, offsets
    )
    assert res_out.grad_fn is not None
    assert "NestedViewFromBufferBackward" in type(res_out.grad_fn).__name__

    ref_out = _eager_reference(grad_output, buffer, sizes, strides, offsets)
    utils.gems_assert_close(res_out, ref_out, dtype)
