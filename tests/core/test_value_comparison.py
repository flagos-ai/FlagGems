# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Value comparisons preserve tensor contracts without unsupported device kernels."""

import pytest
import torch

import flag_gems
from tests import accuracy_utils as utils
from tests import kernelgen_test_utils as tu

pytestmark = pytest.mark.value_comparison

_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.complex64]


def _dtype_param(dtype, *values):
    return pytest.param(
        dtype,
        *values,
        id="-".join([str(dtype).split(".")[-1], *values]),
        marks=pytest.mark.skipif(
            not dtype.is_complex and not utils.fp8_is_supported,
            reason="FP8 storage is required",
        ),
    )


@pytest.fixture(params=[tu.assert_result_equal, tu.assert_result_close])
def compare(request):
    return request.param


@pytest.fixture(params=[_dtype_param(dtype) for dtype in _DTYPES])
def dtype(request):
    return request.param


def _reference(cpu_values):
    return cpu_values.clone() if utils.TO_CPU else cpu_values.to(flag_gems.device)


def _metadata(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.is_neg(),
        tensor.is_conj(),
    )


@pytest.mark.parametrize("kind", ["scalar", "empty", "strided", "special"])
def test_comparison_preserves_values_and_metadata(compare, dtype, kind):
    if kind == "scalar":
        cpu_base = torch.tensor(0.5 + 0.25j if dtype.is_complex else 0.5).to(dtype)
    elif kind == "empty":
        cpu_base = torch.empty((0, 3), dtype=dtype)
    elif kind == "strided":
        payload = [[0.0, 1.0, -1.0], [0.5, -0.5, 2.0]]
        if dtype.is_complex:
            payload = [[1j, 1 - 2j, -1 + 3j], [0.5j, -0.5 - 1j, 2 + 1j]]
        cpu_base = torch.tensor(payload).to(dtype)
    else:
        payload = [float("nan"), 0.0, -0.0, 1.0, -1.0]
        if dtype == torch.float8_e5m2:
            payload += [float("inf"), float("-inf")]
        if dtype.is_complex:
            payload = [
                complex(float("nan"), 1),
                complex(1, float("nan")),
                complex(float("inf"), -2),
                -1 + 2j,
                complex(-0.0, 0.0),
            ]
        cpu_base = torch.tensor(payload).to(dtype)
    base = cpu_base.to(flag_gems.device)
    result = base.t() if kind == "strided" else base
    cpu_values = cpu_base.t() if kind == "strided" else cpu_base
    reference = _reference(cpu_values)
    metadata = _metadata(result)
    before = cpu_base.reshape(-1).view(torch.uint8).clone()

    compare(result, reference)

    assert _metadata(result) == metadata
    assert torch.equal(base.cpu().reshape(-1).view(torch.uint8), before)


_VIEW_CASES = [
    _dtype_param(dtype, kind)
    for dtype in _DTYPES
    for kind in ["sliced", "expanded", "negative"]
    + (["conjugate"] if dtype.is_complex else [])
]


@pytest.mark.parametrize("view_dtype,kind", _VIEW_CASES)
def test_comparison_observes_view_values_without_mutation(compare, view_dtype, kind):
    payload = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5]
    if view_dtype.is_complex:
        payload = [1j, 1 - 2j, -1 + 3j, -0.5j, 0.5 + 1j, -0.5 - 2j]
    cpu_base = torch.tensor(payload).to(view_dtype)
    base = cpu_base.to(flag_gems.device)
    if kind == "sliced":
        result, expected = base[1::2], cpu_base[1::2]
    elif kind == "expanded":
        result, expected = base[:1].expand(2, 3), cpu_base[:1].expand(2, 3)
    elif kind == "negative":
        result = torch._neg_view(base)
        expected = (-cpu_base if view_dtype.is_complex else -cpu_base.float()).to(
            view_dtype
        )
    else:
        result, expected = base.conj(), cpu_base.conj().resolve_conj()
    metadata = _metadata(result)
    assert torch._C._is_alias_of(result, base)

    compare(result, _reference(expected))

    assert _metadata(result) == metadata
    assert torch._C._is_alias_of(result, base)
    assert torch.equal(base.cpu().view(torch.uint8), cpu_base.view(torch.uint8))


def test_comparison_rejects_changed_values(compare, dtype):
    result_values, ref_values = [0.0, 1.0], [0.0, -1.0]
    if dtype.is_complex:
        result_values, ref_values = [0j, 1 + 1j], [0j, 1 - 1j]
    result = torch.tensor(result_values).to(dtype).to(flag_gems.device)
    reference = _reference(torch.tensor(ref_values).to(dtype))
    with pytest.raises(AssertionError):
        compare(result, reference)


def test_comparison_rejects_wrong_shape(compare, dtype):
    cpu_values = torch.tensor([0.0, 1.0]).to(dtype)
    result = cpu_values.to(flag_gems.device)
    reference = _reference(cpu_values.reshape(1, 2))
    with pytest.raises(AssertionError):
        compare(result, reference)


def test_comparison_rejects_wrong_dtype(compare, dtype):
    cpu_values = torch.tensor([0.0, 1.0]).to(dtype)
    result = cpu_values.to(flag_gems.device)
    reference = _reference(torch.zeros(cpu_values.shape, dtype=torch.float32))
    with pytest.raises(AssertionError):
        compare(result, reference)


def test_comparison_rejects_wrong_device(compare, dtype):
    cpu_values = torch.tensor([0.0, 1.0]).to(dtype)
    reference = cpu_values.to(flag_gems.device)
    # CPU reference mode requires a CPU oracle; default mode requires matching
    # original devices. Neither may accept this invalid reference/result pair.
    result = reference.clone() if utils.TO_CPU else cpu_values
    with pytest.raises(AssertionError):
        compare(result, reference)


def test_comparison_allows_host_only_contracts(compare, dtype):
    cpu_values = torch.tensor([0.0, 1.0]).to(dtype)
    compare(cpu_values, cpu_values.clone())
