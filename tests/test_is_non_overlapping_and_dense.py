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

from . import test_utils as tu

# aten::is_non_overlapping_and_dense(Tensor self) -> bool is decided entirely by
# sizes / strides / storage_offset, so it has no value semantics, never
# broadcasts an operand and is not differentiable (autograd.grad on its bool
# result raises TypeError: 'bool' object is not iterable).  Native probes on the
# active backend returned a Python bool for every dtype below, False for a sparse
# COO self and RuntimeError for a non-Tensor self.

# Static runtime capability flags, read once at import time (no probe call).
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    return flag_name is None or bool(getattr(flag_gems.runtime.device, flag_name))


DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.float64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    )
    if _dtype_supported(dtype)
]


def _assert_bool_result(result, reference):
    # The schema returns a Python bool: compare type and value directly.
    assert isinstance(result, bool)
    assert result == reference


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DTYPES)
def test_is_non_overlapping_and_dense(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_inp)
    res_out = flag_gems.is_non_overlapping_and_dense(inp)

    _assert_bool_result(res_out, ref_out)


_BASE_SHAPE = (4, 6)

# Layouts of one 4x6 storage: contiguity, dense permutations (via view and via
# as_strided strides), strided and offset views, singleton / empty / scalar
# boundaries, stride-0 expansion and as_strided gaps / overlaps.  These rows are
# the whole semantics of the predicate and each allocates one small storage, so
# every row runs in quick mode too.
_LAYOUTS = {
    "contiguous": lambda t: t,
    "transposed": lambda t: t.t(),
    "permuted": lambda t: t.reshape(2, 2, 6).permute(2, 0, 1),
    "strided_last_dim": lambda t: t[:, ::2],
    "strided_3d_last_dim": lambda t: t.reshape(2, 2, 6)[..., ::2],
    "sliced_transposed": lambda t: t[:, 1:5].t(),
    "expanded_rows": lambda t: t[:1].expand(4, 6),
    "expanded_3d": lambda t: t[:1].reshape(1, 2, 3).expand(5, 2, 3),
    "offset_view": lambda t: t.reshape(-1)[6:].reshape(3, 6),
    "narrow": lambda t: t[1:3],
    "singleton_slice": lambda t: t[:1],
    "select_index": lambda t: t[0],
    "scalar_view": lambda t: t[0, 0],
    "empty_view": lambda t: t[:0],
    "as_strided_gap": lambda t: t.as_strided((2, 3), (7, 1)),
    "as_strided_overlap": lambda t: t.as_strided((2, 4), (1, 1)),
    "as_strided_dense_perm": lambda t: t.as_strided((3, 2, 4), (1, 12, 3)),
}

_LAYOUT_DTYPES = [
    torch.float32,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.bool,
    torch.complex64,
]

_LAYOUT_ROWS = [
    (layout, dtype)
    for layout in _LAYOUTS
    for dtype in _LAYOUT_DTYPES
    if _dtype_supported(dtype)
]


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("layout,dtype", _LAYOUT_ROWS)
def test_is_non_overlapping_and_dense_layout(layout, dtype):
    base = tu.make_input(dtype, _BASE_SHAPE, ["-1", "1"])
    inp = _LAYOUTS[layout](base)
    meta = (tuple(inp.shape), inp.stride(), inp.storage_offset())
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_inp)
    res_out = flag_gems.is_non_overlapping_and_dense(inp)

    # The predicate must not disturb the input's own layout metadata.
    assert (tuple(inp.shape), inp.stride(), inp.storage_offset()) == meta
    _assert_bool_result(res_out, ref_out)


@pytest.mark.is_non_overlapping_and_dense
def test_is_non_overlapping_and_dense_sparse():
    # A COO tensor is not a dense layout: the native predicate returns False
    # instead of raising, so the candidate must agree on that input form.
    def _sparse():
        indices = torch.tensor([[0, 1], [1, 2]], device=flag_gems.device)
        values = torch.ones(2, dtype=torch.float32, device=flag_gems.device)
        return torch.sparse_coo_tensor(indices, values, (4, 6))

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(_sparse())
    res_out = flag_gems.is_non_overlapping_and_dense(_sparse())

    _assert_bool_result(res_out, ref_out)


_SPECIAL_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

# nan / inf / mixed payloads for every supported floating dtype.  The predicate
# reads metadata only, so each payload is also observed through a stride-0
# expanded view: a candidate that inspected element values instead of strides
# would diverge there.  Default-only, like the other positive special cases.
_SPECIAL_ROWS = tu.selected_cases(
    [
        (dtype, scenario, arrangement)
        for dtype, scenario in tu.special_value_cases(
            [dtype for dtype in _SPECIAL_DTYPES if _dtype_supported(dtype)]
        )
        for arrangement in ("dense", "expanded")
    ],
    quick=[],
)


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("dtype,scenario,arrangement", _SPECIAL_ROWS)
def test_is_non_overlapping_and_dense_special_values(dtype, scenario, arrangement):
    base = tu.make_special_input(dtype, scenario)
    inp = base if arrangement == "dense" else base[:1].expand(base.shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_inp)
    res_out = flag_gems.is_non_overlapping_and_dense(inp)

    _assert_bool_result(res_out, ref_out)


# The native schema rejects a non-Tensor 'self' with RuntimeError and a call
# without arguments with TypeError; both levels keep every row.
_NEGATIVE_ARGS = [3.14, [1.0, 2.0], "tensor"]


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("arg", _NEGATIVE_ARGS)
def test_is_non_overlapping_and_dense_invalid_argument(arg):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_non_overlapping_and_dense(arg)


@pytest.mark.is_non_overlapping_and_dense
def test_is_non_overlapping_and_dense_missing_argument():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_non_overlapping_and_dense()
