# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reference copies must describe the same view on the selected device."""

import pytest
import torch

import flag_gems
from tests import accuracy_utils as utils
from tests import kernelgen_test_utils as tu

pytestmark = pytest.mark.reference_geometry

_DTYPES = [
    torch.float32,
    torch.complex64,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
_VIEWS = {
    "scalar": ((), (), 5),
    "empty": ((0, 3), (9, 2), 7),
    "gapped": ((2, 3), (8, 2), 1),
    "expanded": ((3, 4), (0, 1), 2),
    "offset": ((2, 4), (4, 1), 4),
    "transposed": ((3, 2), (1, 3), 1),
    "overlap": ((2, 3), (1, 1), 2),
    "negative": ((2, 3), (3, 1), 3),
}


def _inputs(dtype, kind):
    values = torch.arange(32, dtype=torch.float32)
    if dtype.is_complex:
        values = torch.complex(values, values / 2)
    cpu_base = values.to(dtype)
    base = cpu_base.to(flag_gems.device)
    shape, stride, offset = _VIEWS.get(kind, _VIEWS["offset"])
    inp = torch.empty(0, dtype=dtype, device=base.device).set_(
        base.untyped_storage(), offset, shape, stride
    )
    expected = torch.empty(0, dtype=dtype).set_(
        cpu_base.untyped_storage(), offset, shape, stride
    )
    # Widen CPU FP8 values before applying lazy negation to the expected values.
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        expected = expected.float()
    if kind == "negative":
        inp, expected = torch._neg_view(inp), -expected
    elif kind == "conjugate":
        inp, expected = inp.conj(), expected.conj()
    return base, inp, expected, cpu_base


def _metadata(tensor):
    return (
        tensor.shape,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.is_conj(),
        tensor.is_neg(),
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kind", list(_VIEWS))
def test_reference_keeps_view_geometry_and_values(dtype, kind):
    base, inp, expected, cpu_base = _inputs(dtype, kind)
    before = _metadata(inp)
    reference = tu.to_reference(inp)
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else inp.device)
    assert reference.dtype == inp.dtype
    assert _metadata(reference) == before
    assert not torch._C._is_alias_of(reference, inp)
    assert not reference.requires_grad
    assert reference.grad_fn is None
    observed = tu._comparison_to_cpu(reference)
    torch.testing.assert_close(observed, expected.to(observed.dtype), rtol=0, atol=0)
    assert _metadata(inp) == before
    assert torch.equal(base.cpu().view(torch.uint8), cpu_base.view(torch.uint8))


def test_reference_keeps_lazy_conjugation():
    _, inp, expected, _ = _inputs(torch.complex64, "conjugate")
    reference = tu.to_reference(inp)
    assert _metadata(reference) == _metadata(inp)
    assert reference.is_conj()
    torch.testing.assert_close(
        tu._comparison_to_cpu(reference), expected, rtol=0, atol=0
    )


def test_host_input_reference_is_independent_in_both_modes():
    base = torch.arange(24, dtype=torch.float32)
    inp = base[3::2]
    reference = tu.to_reference(inp)
    assert reference.device.type == "cpu"
    assert _metadata(reference) == _metadata(inp)
    assert not torch._C._is_alias_of(reference, inp)
    reference.untyped_storage().fill_(0)
    assert torch.equal(base, torch.arange(24, dtype=torch.float32))


def test_reference_accepts_zero_byte_storage():
    inp = torch.empty(0, device=flag_gems.device)
    reference = tu.to_reference(inp)
    assert _metadata(reference) == _metadata(inp)
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else inp.device)
    assert reference.numel() == 0
    assert not torch._C._is_alias_of(reference, inp)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_reference_preserves_grad_requirement_without_input_history(dtype):
    _, inp, _, _ = _inputs(dtype, "gapped")
    inp.requires_grad_(True)
    reference = tu.to_reference(inp)
    assert reference.requires_grad
    assert _metadata(reference) == _metadata(inp)
    assert not torch._C._is_alias_of(reference, inp)
    if utils.TO_CPU:
        objective = reference.real.sum() if dtype.is_complex else reference.sum()
        gradient = torch.autograd.grad(objective, reference)[0]
        torch.testing.assert_close(gradient, torch.ones_like(reference))
    assert inp.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.uint8])
def test_reference_retains_existing_upcast_value_policy(dtype):
    _, inp, expected, _ = _inputs(dtype, "gapped")
    reference = tu.to_reference(inp, upcast=True)
    full_precision = utils.TO_CPU or utils.fp64_is_supported
    if dtype.is_complex:
        target_dtype = torch.complex128 if full_precision else torch.complex64
    else:
        target_dtype = torch.float64 if full_precision else torch.float32
    assert reference.dtype == target_dtype
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else inp.device)
    torch.testing.assert_close(
        tu._comparison_to_cpu(reference), expected.to(target_dtype), rtol=0, atol=0
    )
    assert not torch._C._is_alias_of(reference, inp)


def test_reference_none_remains_none():
    assert tu.to_reference(None) is None


def test_reference_preserves_cpu_sparse_fallback():
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 2.0])
    inp = torch.sparse_coo_tensor(indices, values, (2, 2))
    reference = tu.to_reference(inp)
    assert reference.layout == torch.sparse_coo
    assert reference.device.type == "cpu"
    torch.testing.assert_close(reference._indices(), inp._indices())
    torch.testing.assert_close(reference._values(), inp._values())
    assert reference._values().data_ptr() != inp._values().data_ptr()
