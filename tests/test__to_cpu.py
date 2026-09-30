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

"""Correctness tests for ``aten::_to_cpu``.

``_to_cpu(tensors: Tensor[]) -> Tensor[]`` returns one CPU tensor per input,
copying device tensors to the host and returning host tensors unchanged. Every
test calls the native operator on the *actual* input and then the candidate on
that same tensor: the oracle must see the real source placement, because under
``--ref cpu`` a relocated input would turn the native call into the host
identity path and hide the layout and lazy-flag behaviour that a device-to-host
transfer defines. ``tu.to_reference`` is used here only for independent
snapshots proving the source is unchanged.

Not applicable: broadcast and scalar operands (the schema takes only a tensor
sequence) and arithmetic tolerance (a copy is exact).
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Optional dtypes that only exist on a backend whose capability flag is on.
_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# The nine required dtypes plus float64/bool/complex, which the native copy also
# accepts and reproduces exactly.
SUPPORTED_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.complex64, torch.complex128, torch.bool]
    if _dtype_supported(dtype)
]

_FLOAT_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _dtype_supported(dtype)
]

# Empty tensors and the 0-dim scalar sit outside the spec's seven-shape grid;
# both are cheap, so quick mode keeps them (its grid shape is (2, 19, 7)).
_BOUNDARY_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.int32, torch.float8_e4m3fn)
    if _dtype_supported(dtype)
]
_BOUNDARY_CASES = [
    (dtype, shape, value_range)
    for dtype in _BOUNDARY_DTYPES
    for shape, value_range in (
        ((), ["-1", "1"]),
        ((0,), ["-1", "1"]),
        ((0,), ["min", "max"]),
    )
]

# One row per workload: the five value ranges x the seven spec shapes for every
# supported dtype, plus the empty/0-dim boundary rows.
GRID_ROWS = [
    (dtype, shape, value_range)
    for dtype in SUPPORTED_DTYPES
    for shape in tu.selected_shapes()
    for value_range in tu.selected_ranges()
] + tu.selected_cases(
    _BOUNDARY_CASES,
    quick=_BOUNDARY_CASES,
)


@pytest.mark.to_cpu
@pytest.mark.parametrize("dtype,shape,value_range", GRID_ROWS)
def test__to_cpu(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    source = tu.to_reference(inp)

    # The oracle runs on the actual tensor, not on a relocated copy, and the
    # candidate then sees that same tensor.
    ref_out = torch.ops.aten._to_cpu([inp])
    res_out = flag_gems._to_cpu([inp])

    tu.assert_result_equal(inp, source)
    assert len(res_out) == 1
    assert res_out[0].device.type == "cpu"
    tu.assert_result_equal(res_out[0], ref_out[0])


_CONTAINERS = {"list": list, "tuple": tuple, "empty": lambda _tensors: []}
# Mixed entries for the sequence forms; int64 and fp8 drop out only when their
# backend capability flag is off.
_MIXED_ROWS = [
    (dtype, shape, value_range)
    for dtype, shape, value_range in (
        (torch.float32, (4,), ["-1", "1"]),
        (torch.int64, (2, 3), ["min", "max"]),
        (torch.float8_e4m3fn, (5,), ["-1", "1"]),
    )
    if _dtype_supported(dtype)
]


@pytest.mark.to_cpu
@pytest.mark.parametrize("form", ["list", "tuple", "empty"])
def test__to_cpu_tensor_list_forms(form):
    # Tensor[] accepts a list, a plain tuple and an empty sequence; the result
    # keeps the input length and order.
    tensors = [
        tu.make_input(dtype, shape, value_range)
        for dtype, shape, value_range in _MIXED_ROWS
    ]
    sources = [tu.to_reference(t) for t in tensors]
    entries = _CONTAINERS[form](tensors)

    ref_out = torch.ops.aten._to_cpu(entries)
    res_out = flag_gems._to_cpu(entries)

    for inp, source in zip(tensors, sources):
        tu.assert_result_equal(inp, source)
    assert isinstance(res_out, list)
    assert len(res_out) == (0 if form == "empty" else len(tensors)) == len(ref_out)
    for res, ref in zip(res_out, ref_out):
        assert res.device.type == "cpu"
        tu.assert_result_equal(res, ref)


@pytest.mark.to_cpu
def test__to_cpu_host_input_identity():
    # Host placement is part of the contract: host inputs come back as the same
    # objects, which `is` asserts and a value comparison cannot.
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"]).cpu()
    other = tu.make_input(torch.int32, (3, 4), ["min", "max"]).cpu()
    entries = [inp, other, inp]
    before = [tu.to_reference(t) for t in entries]

    ref_out = torch.ops.aten._to_cpu(entries)
    res_out = flag_gems._to_cpu(entries)

    assert res_out[0] is inp
    assert res_out[1] is other
    assert res_out[2] is inp
    assert len(res_out) == len(entries)
    for res, ref, snapshot in zip(res_out, ref_out, before):
        assert res.device.type == "cpu"
        tu.assert_result_equal(res, ref)
        tu.assert_result_equal(res, snapshot)


@pytest.mark.to_cpu
def test__to_cpu_duplicate_entries():
    inp = tu.make_input(torch.float32, (16,), ["-1", "1"])
    source = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_cpu([inp, inp])
    res_out = flag_gems._to_cpu([inp, inp])

    tu.assert_result_equal(inp, source)
    assert len(res_out) == 2
    for res, ref in zip(res_out, ref_out):
        tu.assert_result_equal(res, ref)
    if inp.device.type != "cpu":
        # A device-to-host copy is per entry: equal inputs stay distinct.
        assert res_out[0].data_ptr() != res_out[1].data_ptr()


def _layout_input(kind):
    """Device tensor whose stride/offset/contiguity profile is ``kind``."""
    dev = flag_gems.device
    if kind == "contiguous":
        return torch.arange(4 * 8, device=dev, dtype=torch.float32).reshape(4, 8)
    if kind == "permute":
        return (
            torch.arange(4 * 8, device=dev, dtype=torch.float32)
            .reshape(4, 8)
            .permute(1, 0)
        )
    if kind == "strided_slice":
        base = torch.arange(8 * 10, device=dev, dtype=torch.float32).reshape(8, 10)
        return base[1:7:2, 2:8]
    if kind == "offset_slice":
        base = torch.arange(8 * 10, device=dev, dtype=torch.float32).reshape(8, 10)
        return base[3:5]
    if kind == "expanded":
        return (
            torch.arange(1, 5, device=dev, dtype=torch.float32)
            .reshape(4, 1)
            .expand(4, 6)
        )
    if kind == "channels_last":
        base = torch.arange(2 * 3 * 4 * 5, device=dev, dtype=torch.float32).reshape(
            2, 3, 4, 5
        )
        return base.contiguous(memory_format=torch.channels_last)
    raise AssertionError(f"unknown layout case: {kind}")


# All six profiles are cheap, so quick mode keeps every one of them.
LAYOUT_CASES = tu.selected_cases(
    [
        "contiguous",
        "permute",
        "strided_slice",
        "offset_slice",
        "expanded",
        "channels_last",
    ],
    quick=[
        "contiguous",
        "permute",
        "strided_slice",
        "offset_slice",
        "expanded",
        "channels_last",
    ],
)


@pytest.mark.to_cpu
@pytest.mark.parametrize("kind", LAYOUT_CASES)
def test__to_cpu_layout(kind):
    inp = _layout_input(kind)
    source = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_cpu([inp])[0]
    res_out = flag_gems._to_cpu([inp])[0]

    tu.assert_result_equal(inp, source)
    tu.assert_result_equal(res_out, ref_out)
    # The native result decides the transferred layout: a dense source keeps its
    # strides, a strided/offset/expanded source is materialized with a zero
    # storage offset, channels_last stays channels_last.
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_contiguous() == ref_out.is_contiguous()
    assert res_out.is_contiguous(
        memory_format=torch.channels_last
    ) == ref_out.is_contiguous(memory_format=torch.channels_last)


def _lazy_flag_input(kind):
    """Tensor carrying a lazy negative or conjugate bit."""
    if kind == "neg":
        return torch._neg_view(tu.make_input(torch.float32, (4, 8), ["-1", "1"]))
    return tu.make_input(torch.complex64, (4, 8), ["-1", "1"]).conj()


# Both flags are cheap, so quick mode keeps both.
LAZY_CASES = tu.selected_cases(["neg", "conj"], quick=["neg", "conj"])


@pytest.mark.to_cpu
@pytest.mark.parametrize("kind", LAZY_CASES)
def test__to_cpu_lazy_view_flags(kind):
    inp = _lazy_flag_input(kind)
    source = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_cpu([inp])[0]
    res_out = flag_gems._to_cpu([inp])[0]

    tu.assert_result_equal(inp, source)
    # The native result decides whether the lazy bit survives: a device-to-host
    # copy materializes it, a host input keeps it.
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    tu.assert_result_equal(res_out, ref_out)


# Differentiable types the native copy-backward supports. FP8 and complex are
# included because a copy performs no arithmetic. Default-only: backward is not
# a quick-mode dimension.
_BACKWARD_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.complex64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _dtype_supported(dtype)
]
BACKWARD_CASES = tu.selected_cases(
    [(dtype, shape) for dtype in _BACKWARD_DTYPES for shape in ((256,), (20, 320, 15))],
    quick=[],
)


@pytest.mark.to_cpu
@pytest.mark.parametrize("dtype,shape", BACKWARD_CASES)
def test__to_cpu_backward(dtype, shape):
    # The gradient flows through the original leaf. The reference leaf is an
    # independent tensor on the same actual device, so the oracle performs a
    # real transfer instead of differentiating a host identity, and the
    # upstream gradient is non-uniform.
    leaf = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    source = tu.to_reference(leaf.detach())
    upstream = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    ref_leaf = leaf.detach().clone().requires_grad_(True)

    ref_out = torch.ops.aten._to_cpu([ref_leaf])[0]
    res_out = flag_gems._to_cpu([leaf])[0]

    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(leaf, source)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_leaf, grad_outputs=upstream)
    (res_grad,) = torch.autograd.grad(res_out, leaf, grad_outputs=upstream)
    # Gradients live on the device; observing them through the host keeps the
    # shared --ref cpu device check satisfied in both reference modes.
    tu.assert_result_equal(res_grad.cpu(), ref_grad.cpu())


SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])


@pytest.mark.to_cpu
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test__to_cpu_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    source = tu.to_reference(inp)

    ref_out = torch.ops.aten._to_cpu([inp])[0]
    res_out = flag_gems._to_cpu([inp])[0]

    tu.assert_result_equal(inp, source)
    tu.assert_result_equal(res_out, ref_out)


# The schema accepts only a tensor sequence; each form fails while binding the
# arguments (probed with the native operator). Every row runs in both modes.
_INVALID_ARGUMENTS = {
    "missing": lambda inp: (),
    "none": lambda inp: (None,),
    "bare_tensor": lambda inp: (inp,),
    "list_with_non_tensor": lambda inp: ([inp, 1.0],),
    "string": lambda inp: ("not a tensor list",),
}


@pytest.mark.to_cpu
@pytest.mark.parametrize("kind", sorted(_INVALID_ARGUMENTS))
def test__to_cpu_rejects_invalid_arguments(kind):
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._to_cpu(*_INVALID_ARGUMENTS[kind](inp))
