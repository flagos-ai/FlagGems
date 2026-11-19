from contextlib import contextmanager

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import _SKIP_FP64


def _is_suiyuan_double(inp):
    return flag_gems.vendor_name == "enflame" and inp.dtype == torch.float64


def _suiyuan_double_cpu(inp):
    # GCU typed Double copies can silently narrow. Raw storage preserves all bits.
    storage = inp.untyped_storage().cpu()
    result = torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        storage, inp.storage_offset(), tuple(inp.shape), inp.stride()
    )
    if inp.is_neg():
        result = result.neg()
    return result.clone()


def _reference_input(inp, *, clone=False):
    if _is_suiyuan_double(inp) and utils.TO_CPU:
        return _suiyuan_double_cpu(inp)
    return utils.to_reference(inp.clone() if clone else inp, True)


def _readonly_snapshot(inp):
    # Readonly means unchanged physical bytes, including NaN payloads and signed zero.
    storage = inp.untyped_storage().cpu()
    return (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .clone()
    )


def _assert_readonly_value(inp, snapshot):
    assert torch.equal(_readonly_snapshot(inp), snapshot)


def _comparison_result(result):
    # Call only after target-device/dtype/shape and return-identity assertions.
    return (
        _suiyuan_double_cpu(result)
        if utils.TO_CPU and _is_suiyuan_double(result)
        else result
    )


def _input_metadata(inp):
    storage = inp.untyped_storage()
    return (
        inp.device,
        inp.dtype,
        inp.shape,
        inp.stride(),
        inp.storage_offset(),
        storage.data_ptr(),
        storage._cdata,
        storage.nbytes(),
        inp.requires_grad,
        inp.is_conj(),
        inp.is_neg(),
    )


@contextmanager
def _check_inputs(inp, other, *, self_readonly=False):
    original_metadata = (_input_metadata(inp), _input_metadata(other))
    other_before = _readonly_snapshot(other)
    inp_before = _readonly_snapshot(inp) if self_readonly else None
    try:
        yield
    finally:
        assert _input_metadata(inp) == original_metadata[0]
        assert _input_metadata(other) == original_metadata[1]
        _assert_readonly_value(other, other_before)
        if self_readonly:
            _assert_readonly_value(inp, inp_before)


@pytest.mark.igamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", [torch.float32, pytest.param(torch.float64, marks=_SKIP_FP64)]
)
def test_igamma_(shape, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("float64 is not supported by this device")
    if _is_suiyuan_double(torch.empty(0, dtype=dtype, device=flag_gems.device)):
        if not utils.TO_CPU and all(size != 0 for size in shape):
            pytest.skip("Enflame native igamma does not compute nonempty Double values")
        inp = (
            (torch.rand(shape, dtype=dtype) + 1.0)
            .reshape(-1)
            .view(torch.uint8)
            .to(flag_gems.device)
            .view(dtype)
            .reshape(shape)
        )
        other = (
            (torch.rand(shape, dtype=dtype) + 1.0)
            .reshape(-1)
            .view(torch.uint8)
            .to(flag_gems.device)
            .view(dtype)
            .reshape(shape)
        )
    else:
        inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
        other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = _reference_input(inp, clone=True)
    ref_other = _reference_input(other)

    ref_out = ref_inp.igamma_(ref_other)
    with _check_inputs(inp, other):
        res_out = flag_gems.igamma_(inp, other)
    assert res_out is inp
    assert res_out.device == inp.device and res_out.dtype == inp.dtype

    utils.gems_assert_close(_comparison_result(res_out), ref_out, dtype, atol=1e-3)
