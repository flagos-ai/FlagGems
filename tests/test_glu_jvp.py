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

# aten::glu_jvp(glu, x, dx, dim) is the JVP of aten::glu(x, dim): glu is the
# primal result, so the output takes glu's shape and x / dx are narrowed to
# glu.size(dim) along dim. The operator only requires glu.size(dim) to fit into
# x.size(dim) // 2, so extents smaller than the half width and odd halving
# sizes (15 -> 7) are valid and the primal need not come from aten::glu.
#
# Native backward is not available ("derivative for aten::glu_jvp is not
# implemented"), so this file carries no gradient workload.
_SUPPORT_BF16 = flag_gems.runtime.device.support_bf16
_SUPPORT_FP64 = flag_gems.runtime.device.support_fp64

# The kernel is float-only, so the int8 / uint8 / fp8 / int32 / int64 slots of
# the spec dtype grid are covered by the rejection workloads further down.
_DTYPES = (
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if _SUPPORT_BF16 else [])
    + ([torch.float64] if _SUPPORT_FP64 else [])
)
# Rejection expectation measured on nvidia (glu_cuda). On any other backend the
# list is empty and pytest reports no case instead of asserting unmeasured
# dispatch.
_UNSUPPORTED_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.complex64,
    ]
    if flag_gems.vendor_name == "nvidia"
    else []
)


def _primal_shape(shape, dim, extent=None):
    """Shape of the primal operand: its extent along dim."""
    primal = list(shape)
    primal[dim] = shape[dim] // 2 if extent is None else extent
    return tuple(primal)


def _operand(kind, dtype, shape, dim, value_range):
    """Build one operand of shape in the input state named by kind."""
    if kind == "contiguous":
        return tu.make_input(dtype, shape, value_range)
    if kind == "strided":
        wide = list(shape)
        wide[dim] *= 2
        base = tu.make_input(dtype, tuple(wide), value_range)
        index = [slice(None)] * len(shape)
        index[dim] = slice(1, None, 2)
        return base[tuple(index)]
    if kind == "offset":
        base = tu.make_input(dtype, (shape[0] + 2, shape[1] + 2), value_range)
        return base[1 : shape[0] + 1, : shape[1]]
    if kind == "transposed":
        return tu.make_input(dtype, (shape[1], shape[0]), value_range).t()
    if kind == "permuted":
        base = tu.make_input(dtype, (shape[2], shape[0], shape[1]), value_range)
        return base.permute(1, 2, 0)
    raise ValueError(f"unknown input state: {kind}")


# Value grid: every supported dtype over all five value ranges, on the shape
# levels that have a dim to halve. A 0-dim input has no dim at all and only
# takes part in the rejection workloads.
_GRID_CASES = [
    (shape, -1, dtype, value_range, "contiguous", None)
    for shape in tu.selected_shapes()
    if len(shape) >= 1
    for value_range in tu.selected_ranges()
    for dtype in _DTYPES
]

# dim=0 coverage uses the prescribed parameter dtype: fp16 is the only dtype
# from that list the float-only kernel supports, and bf16 joins it when the
# backend advertises support. fp32 is kept as an extra row.
_PARAM_DTYPES = [torch.float16, torch.float32] + (
    [torch.bfloat16] if _SUPPORT_BF16 else []
)
_DIM0_CASES = [
    ((1024, 1024), 0, dtype, ["-1", "1"], "contiguous", None) for dtype in _PARAM_DTYPES
]

# Remaining dims, the empty-input and rank-1 states, an odd halving dim, primal
# extents below the half width, and the non-contiguous / offset states. Each row
# is one workload: the fifth element names the input state and the sixth the
# primal extent along dim (None = the half width). Default only.
_EXTRA_CASES = _DIM0_CASES + [
    ((20, 320, 15), 1, torch.float16, ["-1", "1"], "contiguous", None),
    ((16, 128, 64, 60), 2, torch.float32, ["-1", "1"], "contiguous", None),
    ((0, 8), -1, torch.float32, ["-1", "1"], "contiguous", None),
    ((4, 0), -1, torch.float32, ["-1", "1"], "contiguous", None),
    ((0,), 0, torch.float32, ["-1", "1"], "contiguous", None),
    ((4, 9), 1, torch.float32, ["-1", "1"], "contiguous", None),
    ((1024, 1024), 1, torch.float16, ["-1", "1"], "contiguous", 100),
    ((1024, 1024), 1, torch.float32, ["-1", "1"], "contiguous", 1),
    ((20, 320, 15), -1, torch.float32, ["-1", "1"], "contiguous", 3),
    ((20, 320, 15), -1, torch.float32, ["-1", "1"], "strided", None),
    ((1024, 1024), 1, torch.float16, ["-1", "1"], "strided", None),
    ((8, 16), -1, torch.float32, ["-1", "1"], "transposed", None),
    ((10, 10), -1, torch.float32, ["-1", "1"], "offset", None),
    ((3, 8, 10), 1, torch.float16, ["-1", "1"], "permuted", None),
]

_POSITIVE_CASES = tu.selected_cases(
    _GRID_CASES + _EXTRA_CASES,
    quick=[
        ((2, 19, 7), -1, dtype, ["-1", "1"], "contiguous", None) for dtype in _DTYPES
    ],
)


@pytest.mark.glu_jvp
@pytest.mark.parametrize("shape,dim,dtype,value_range,kind,extent", _POSITIVE_CASES)
def test_glu_jvp(shape, dim, dtype, value_range, kind, extent):
    x = _operand(kind, dtype, shape, dim, value_range)
    dx = _operand(kind, dtype, shape, dim, value_range)
    glu = tu.make_input(dtype, _primal_shape(shape, dim, extent), value_range)

    ref_out = torch.ops.aten.glu_jvp(
        tu.to_reference(glu), tu.to_reference(x), tu.to_reference(dx), dim
    )
    res_out = flag_gems.glu_jvp(glu, x, dx, dim)

    tu.assert_result_close(res_out, ref_out)


# Broadcast against the axes glu_jvp does not narrow. The narrowed axis keeps
# its full extent on every operand, since the native narrow(dim, m, m) needs the
# same extent there.
_BROADCAST_ROWS = [
    ((16, 128, 64, 60), -1, (16, 1, 64, 30), (16, 128, 64, 60)),
    ((20, 320, 15), 1, (20, 160, 15), (1, 320, 1)),
]
_BROADCAST_CASES = tu.selected_cases(
    [
        (shape, dim, glu_shape, dx_shape, dtype)
        for shape, dim, glu_shape, dx_shape in _BROADCAST_ROWS
        for dtype in _DTYPES
    ],
    quick=[],
)


@pytest.mark.glu_jvp
@pytest.mark.parametrize("shape,dim,glu_shape,dx_shape,dtype", _BROADCAST_CASES)
def test_glu_jvp_broadcast(shape, dim, glu_shape, dx_shape, dtype):
    x = tu.make_input(dtype, shape, ["-1", "1"])
    dx = tu.make_input(dtype, dx_shape, ["-1", "1"])
    glu = tu.make_input(dtype, glu_shape, ["-1", "1"])

    ref_out = torch.ops.aten.glu_jvp(
        tu.to_reference(glu), tu.to_reference(x), tu.to_reference(dx), dim
    )
    res_out = flag_gems.glu_jvp(glu, x, dx, dim)

    tu.assert_result_close(res_out, ref_out)


# aten::glu_jvp.out is a real native kernel on this backend: it fills the caller
# buffer and returns that same object, so the reference keeps its own buffer and
# the candidate is reached through the same public entry point with out=.
_OUT_ROWS = [((256,), -1), ((20, 320, 15), -2), ((0, 8), -1), ((1024, 1024), 1)]
_OUT_CASES = tu.selected_cases(
    [(shape, dim, dtype) for shape, dim in _OUT_ROWS for dtype in _DTYPES],
    quick=[],
)


@pytest.mark.glu_jvp
@pytest.mark.parametrize("shape,dim,dtype", _OUT_CASES)
def test_glu_jvp_out(shape, dim, dtype):
    x = tu.make_input(dtype, shape, ["-1", "1"])
    dx = tu.make_input(dtype, shape, ["-1", "1"])
    glu = tu.make_input(dtype, _primal_shape(shape, dim), ["-1", "1"])

    ref_out = torch.empty_like(tu.to_reference(glu))
    torch.ops.aten.glu_jvp.out(
        tu.to_reference(glu), tu.to_reference(x), tu.to_reference(dx), dim, out=ref_out
    )

    out = torch.full_like(glu, 0.5)
    res_out = flag_gems.glu_jvp(glu, x, dx, dim, out=out)

    # The returned object itself must be the caller buffer, checked on the
    # object because an empty tensor's data_ptr does not identify it.
    assert res_out is out
    tu.assert_result_close(res_out, ref_out)


# nan / inf / mixed payloads, default only. The shared payload is one 5-element
# row; it is doubled so x has twice the primal extent along dim.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_DTYPES),
    quick=[],
)


@pytest.mark.glu_jvp
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_glu_jvp_special_values(dtype, scenario):
    x = tu.make_special_input(dtype, scenario).repeat(2)
    dx = tu.make_special_input(dtype, scenario).repeat(2)
    glu = tu.make_special_input(dtype, scenario)

    ref_out = torch.ops.aten.glu_jvp(
        tu.to_reference(glu), tu.to_reference(x), tu.to_reference(dx), -1
    )
    res_out = flag_gems.glu_jvp(glu, x, dx, -1)

    tu.assert_result_close(res_out, ref_out)


# Rejection workloads, kept in both execution levels.
@pytest.mark.glu_jvp
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_glu_jvp_rejects_non_float_dtype(dtype):
    x = torch.ones((4, 8), dtype=dtype, device=flag_gems.device)
    glu = torch.ones((4, 4), dtype=dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.glu_jvp(glu, x, x, 1)


@pytest.mark.glu_jvp
@pytest.mark.parametrize("bad_dim", [2, -3])
def test_glu_jvp_rejects_out_of_range_dim(bad_dim):
    x = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
    glu = tu.make_input(torch.float32, (4, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.glu_jvp(glu, x, x, bad_dim)


# The narrowed dim must supply 2 * glu.size(dim) values, so a dx that is too
# short there and a primal extent above x.size(dim) // 2 are both rejected with
# an in-range dim.
_NARROW_CASES = [
    ((4, 8), (4, 6), (4, 4), 1),
    ((4, 8), (4, 8), (4, 5), 1),
]


@pytest.mark.glu_jvp
@pytest.mark.parametrize("x_shape,dx_shape,glu_shape,dim", _NARROW_CASES)
def test_glu_jvp_rejects_invalid_narrow_extent(x_shape, dx_shape, glu_shape, dim):
    x = tu.make_input(torch.float32, x_shape, ["-1", "1"])
    dx = tu.make_input(torch.float32, dx_shape, ["-1", "1"])
    glu = tu.make_input(torch.float32, glu_shape, ["-1", "1"])

    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.glu_jvp(glu, x, dx, dim)


@pytest.mark.glu_jvp
def test_glu_jvp_rejects_zero_dim_input():
    x = torch.tensor(0.5, device=flag_gems.device)

    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.glu_jvp(x, x, x, 0)
