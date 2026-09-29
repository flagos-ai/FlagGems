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

# aten::ravel exposes only the `default` overload on the configured backend
# (ravel.overloads() == ["default"], and there is no `.out` form), so there is
# no out= workload to write. The native op aliases the input storage only for an
# input that is already contiguous after flattening (contiguous tensors, empty
# tensors and flattened singletons); every other rank/layout materializes a
# fresh contiguous buffer. A `default` + torch.ops.aten.copy_ simulation would
# reproduce neither path, so the overload is documented here instead of faked.

# Every dtype list below is filtered by these static capability flags; the file
# never probes native support at import or collection time.
_DTYPE_CAPABILITIES = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
    torch.complex128: utils.fp64_is_supported,
}


def _supported_dtypes(dtypes):
    """Drop the dtypes whose static backend capability flag is unavailable."""
    return [dtype for dtype in dtypes if _DTYPE_CAPABILITIES.get(dtype, True)]


# The spec's nine required dtypes plus the 64-bit, small-int and complex types
# the configured backend reports support for.
SUPPORTED_DTYPES = _supported_dtypes(
    tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.int16, torch.complex64, torch.complex128]
)


@pytest.mark.ravel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_ravel(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    source_snapshot = tu.to_reference(inp)

    ref_out = torch.ops.aten.ravel(ref_inp)
    res_out = flag_gems.ravel(inp)

    # Each grid input is contiguous, so this flatten only rewrites shape
    # metadata: no element is copied and the values must compare exactly.
    tu.assert_result_equal(res_out, ref_out)
    assert tuple(res_out.shape) == (inp.numel(),)
    assert res_out.stride() == (1,)
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out.storage_offset() == inp.storage_offset()
    # The result shares the input storage, so the source must stay untouched.
    tu.assert_result_equal(inp, source_snapshot)


def _layout_input(kind, base):
    """Build one natively valid ravel input geometry from ``base``.

    Returns ``(input, parent)`` where ``parent`` is the tensor whose whole
    storage the flatten must leave untouched. The lazy neg/conj rows apply the
    flag directly to ``base``, so the returned parent really is the backing
    tensor of the input; a clone inside this helper would hide the true parent.
    """
    if kind in ("contiguous", "zero_dim"):
        return base, base
    if kind == "row_slice":
        # Contiguous window with a nonzero storage offset.
        return base[1:-1], base
    if kind == "transposed":
        # transpose(-1, -2) needs rank >= 2, which every row using it satisfies;
        # the result is non-contiguous, so ravel materializes a copy.
        return base.transpose(-1, -2), base
    if kind == "step_slice":
        return base[:, ::2], base
    if kind == "expanded":
        # Zero stride: one row of values spread over every output slot.
        return base[:1].expand(base.shape[0], base.shape[1]), base
    if kind == "offset_step":
        # Nonzero storage offset combined with a stride gap.
        return base[1:-1, 1::2], base
    if kind == "singleton_transposed":
        # (9, 1) -> (1, 9): collapsing the size-1 dim leaves this contiguous,
        # so ravel still returns a view rather than a materialized copy.
        return base.transpose(0, 1), base
    if kind == "empty":
        return base[3:3], base
    if kind == "negative_view":
        return torch._neg_view(base), base
    if kind == "conj_view":
        return base.conj(), base
    if kind == "transposed_negative_view":
        # A lazy neg over a non-contiguous transpose: ravel must materialize,
        # so the flag has to be folded into the values instead of carried on.
        return torch._neg_view(base.transpose(-1, -2)), base
    if kind == "step_slice_conj_view":
        # Same for a lazy conj over a strided slice.
        return base[:, ::2].conj(), base
    raise AssertionError(f"unknown layout kind: {kind}")


# (layout kind, base shape, dtype). ravel keeps the input storage when the
# tensor can be flattened in place and otherwise materializes a fresh
# contiguous buffer; either way the lazy neg/conj bits follow the native result.
LAYOUT_CASES = [
    ("contiguous", (6, 8), torch.float32),
    ("row_slice", (6, 8), torch.float32),
    ("transposed", (6, 8), torch.float32),
    ("step_slice", (6, 8), torch.float32),
    ("expanded", (6, 8), torch.float32),
    ("offset_step", (6, 8), torch.float32),
    ("empty", (6, 8), torch.float32),
    ("zero_dim", (), torch.float32),
    ("singleton_transposed", (9, 1), torch.float32),
    ("negative_view", (6, 8), torch.float32),
    ("conj_view", (6, 8), torch.complex64),
    ("transposed_negative_view", (6, 8), torch.float32),
    ("step_slice_conj_view", (6, 8), torch.complex64),
]


@pytest.mark.ravel
@pytest.mark.parametrize(
    "kind,base_shape,dtype", tu.selected_cases(LAYOUT_CASES, quick=[])
)
def test_ravel_layout(kind, base_shape, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    inp, parent = _layout_input(kind, base)
    ref_inp = tu.to_reference(inp)
    parent_snapshot = tu.to_reference(parent)

    ref_out = torch.ops.aten.ravel(ref_inp)
    res_out = flag_gems.ravel(inp)

    tu.assert_result_equal(res_out, ref_out)
    assert tuple(res_out.shape) == (inp.numel(),)
    assert res_out.stride() == (1,)

    if inp.numel() > 0:
        # A data pointer only establishes aliasing for a non-empty tensor.
        aliases = res_out.data_ptr() == inp.data_ptr()
        ref_aliases = ref_out.data_ptr() == ref_inp.data_ptr()
        assert aliases == ref_aliases
        if aliases:
            assert res_out.storage_offset() == inp.storage_offset()
    # Native-guided lazy bits: the copy path materializes them away, the view
    # path preserves them.
    assert res_out.is_neg() == ref_out.is_neg()
    assert res_out.is_conj() == ref_out.is_conj()
    # Neither path may rewrite the backing storage: the aliasing view path keeps
    # every element in place, and the materializing copy path only reads them.
    # That covers the elements outside a sliced window too, which the output
    # comparison alone cannot observe.
    tu.assert_result_equal(parent, parent_snapshot)


# Copy-path geometries: ravel materializes fresh contiguous elements and the
# source storage must stay untouched in both directions.
_COPY_CASES = [
    ("transposed", (4, 6)),
    ("expanded", (4, 6)),
    ("offset_step", (6, 8)),
]


@pytest.mark.ravel
@pytest.mark.parametrize("layout,base_shape", tu.selected_cases(_COPY_CASES, quick=[]))
def test_ravel_copy_isolation(layout, base_shape):
    base = tu.make_input(torch.float32, base_shape, ["-1", "1"])
    inp, parent = _layout_input(layout, base)
    ref_inp = tu.to_reference(inp)
    source_snapshot = tu.to_reference(parent)

    res_out = flag_gems.ravel(inp)
    ref_out = torch.ops.aten.ravel(ref_inp)

    assert tuple(res_out.shape) == tuple(ref_out.shape) == (inp.numel(),)
    assert res_out.stride() == (1,)
    # This path materializes, so the result must not share the input storage.
    assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(parent, source_snapshot)

    res_out.fill_(-3)
    # Writing the materialized copy must not reach the source storage.
    tu.assert_result_equal(parent, source_snapshot)

    out_snapshot = tu.to_reference(res_out)
    base.fill_(7)
    # Writing the source storage must not reach the earlier copy.
    tu.assert_result_equal(res_out, out_snapshot)


_INPLACE_DTYPES = _supported_dtypes(
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.float8_e4m3fn,
        torch.bool,
    ]
)


@pytest.mark.ravel
@pytest.mark.parametrize("dtype", tu.selected_cases(_INPLACE_DTYPES, quick=[]))
def test_ravel_view_mutation(dtype):
    """Writes through a view result must land in the source window only."""
    base = tu.make_input(dtype, (6, 5), ["-1", "1"])
    inp = base[1:5]
    ref_base = tu.to_reference(base)
    ref_inp = tu.to_reference(inp)

    res_out = flag_gems.ravel(inp)
    ref_out = torch.ops.aten.ravel(ref_inp)

    # Compare the returned tensor first: a candidate handing back the rank-2
    # input is caught here, before the write-through below can hide it.
    assert tuple(res_out.shape) == tuple(ref_out.shape) == (20,)
    assert res_out.data_ptr() == inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)

    res_out.fill_(1)
    ref_out.fill_(1)

    # The write travels through the flattened view into the source window ...
    tu.assert_result_equal(inp, ref_inp)
    # ... and must not touch storage outside that window.
    tu.assert_result_equal(base[0], ref_base[0])
    tu.assert_result_equal(base[5], ref_base[5])


_BACKWARD_CASES = [
    ((4, 6), "transposed", torch.float32),
    ((4, 6), "transposed", torch.float64),
    ((4, 6), "transposed", torch.float16),
    ((4, 6), "transposed", torch.bfloat16),
    ((20, 320, 15), "transposed", torch.float32),
    ((4, 6), "row_slice", torch.float32),
    ((20, 320, 15), "row_slice", torch.float32),
]
BACKWARD_CASES = [
    case for case in _BACKWARD_CASES if _DTYPE_CAPABILITIES.get(case[2], True)
]


@pytest.mark.ravel
@pytest.mark.parametrize(
    "shape,layout,dtype", tu.selected_cases(BACKWARD_CASES, quick=[])
)
def test_ravel_backward(shape, layout, dtype):
    # The upstream gradient stays on the candidate device while the reference
    # builds its own copy on the reference device: a shared tensor would hand
    # the candidate a CPU grad_output under --ref cpu.
    base = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    inp, parent = _layout_input(layout, base)
    ref_base = tu.to_reference(base)
    ref_inp = _layout_input(layout, ref_base)[0]
    parent_snapshot = tu.to_reference(parent)
    upstream = tu.make_input(dtype, (inp.numel(),), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.ravel(ref_inp)
    res_out = flag_gems.ravel(inp)
    tu.assert_result_equal(res_out, ref_out)

    ref_grad = torch.autograd.grad(ref_out, ref_base, grad_outputs=ref_upstream)[0]
    res_grad = torch.autograd.grad(res_out, base, grad_outputs=upstream)[0]

    assert tuple(res_grad.shape) == tuple(ref_grad.shape) == tuple(shape)
    # Flattening only reorders gradients, so the comparison is exact.
    tu.assert_result_equal(res_grad, ref_grad)
    # Neither the forward nor the backward pass may rewrite the backing storage.
    tu.assert_result_equal(parent, parent_snapshot)


_SPECIAL_DTYPES = _supported_dtypes(
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
)


@pytest.mark.ravel
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test_ravel_special_values(dtype, scenario):
    # The shared generator only emits representable scenarios: float8_e4m3fn
    # contributes nan, float8_e5m2 adds inf and mixed.
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    source_snapshot = tu.to_reference(inp)

    ref_out = torch.ops.aten.ravel(ref_inp)
    res_out = flag_gems.ravel(inp)

    tu.assert_result_equal(res_out, ref_out)
    # Special payloads are rearranged, never rewritten.
    tu.assert_result_equal(inp, source_snapshot)


@pytest.mark.ravel
@pytest.mark.parametrize("bad_input", [3.14, [1.0, 2.0]])
def test_ravel_rejects_non_tensor(bad_input):
    # `self` is a Tensor in the schema; a Python scalar or list is not.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ravel(bad_input)


def _sparse_input(layout):
    values = torch.ones(2, dtype=torch.float32, device=flag_gems.device)
    if layout == "sparse_coo":
        indices = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.int64, device=flag_gems.device
        )
        return torch.sparse_coo_tensor(indices, values, (3, 3))
    # CSR stores crow/col as pointers, so this fixture can use the narrowest
    # index width the backend actually has.
    index_dtype = torch.int64 if utils.int64_is_supported else torch.int32
    row_indices = torch.tensor([0, 1, 2, 2], dtype=index_dtype, device=flag_gems.device)
    col_indices = torch.tensor([1, 2], dtype=index_dtype, device=flag_gems.device)
    return torch.sparse_csr_tensor(row_indices, col_indices, values, (3, 3))


# COO stores its coordinates as int64 whatever the constructor receives, so that
# fixture needs the backend's int64 storage and is gated on the static
# capability flag; CSR keeps its negative with narrower pointers where int64 is
# unavailable. Gating is per fixture, never on the shared value dtype, and the
# negatives are never probed or caught at runtime.
SPARSE_CASES = (["sparse_coo"] if utils.int64_is_supported else []) + ["sparse_csr"]


@pytest.mark.ravel
@pytest.mark.parametrize("layout", SPARSE_CASES)
def test_ravel_rejects_sparse_layout(layout):
    # Native ravel refuses non-strided layouts: sparse COO raises
    # "unsupported memory format option Contiguous" and sparse CSR raises
    # "Sparse CSR tensors do not have is_contiguous".
    inp = _sparse_input(layout)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ravel(inp)
