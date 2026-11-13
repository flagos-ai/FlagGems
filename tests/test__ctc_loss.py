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
from .accuracy_utils import gems_assert_close
from .conftest import QUICK_MODE

# CPU has no Half kernel for CTC loss, so comparisons stay on the float types.
CTC_DTYPES = [torch.float32]
if flag_gems.vendor_name != "iluvatar" or utils.fp64_is_supported:
    CTC_DTYPES.append(torch.float64)


def _to_target_double(value):
    result = value.contiguous().reshape(-1).view(torch.uint8).to(flag_gems.device).view(torch.float64).reshape(value.shape)
    assert result.dtype == torch.float64
    assert result.shape == value.shape
    return result


def _log_probs(raw, dtype, *, out_variant=False):
    vendor = flag_gems.vendor_name
    limited_fp64 = vendor == "kunlunxin" or (out_variant and vendor == "ascend")
    if limited_fp64 and dtype == torch.float64:
        if not utils.TO_CPU:
            pytest.skip(f"{vendor} native CTC does not preserve float64 outputs")
        return _to_target_double(raw.log_softmax(-1).cpu().double())
    return raw.log_softmax(-1).to(dtype)


def _reference_probs(value, *, out_variant=False):
    # Clone on CPU before Double device clone/cast can narrow the input on KU.
    vendor = flag_gems.vendor_name
    limited_fp64 = vendor == "kunlunxin" or (out_variant and vendor == "ascend")
    if limited_fp64 and value.dtype == torch.float64:
        return value.cpu().clone().detach()
    return utils.to_reference(value.clone().detach(), True)


def _assert_ctc_result(result, reference, dtype, *, reduce_dim):
    assert result.device.type == torch.device(flag_gems.device).type
    assert result.device.index == flag_gems.runtime.torch_device_fn.current_device()
    assert result.dtype == dtype
    assert result.shape == reference.shape
    # Observe values only after checking the target contract.
    gems_assert_close(result.cpu(), reference.cpu(), dtype, reduce_dim=reduce_dim)


def _make_inputs(T, N, S, C, dtype, *, out_variant=False):
    """Build one CTC batch, returning int-list and tensor length forms.

    _ctc_loss and _ctc_loss.out take int[] lengths, while _ctc_loss.Tensor and
    _ctc_loss.Tensor_out take them as tensors, so both forms are handed back.
    """
    raw = torch.randn(T, N, C, dtype=torch.float32, device=flag_gems.device)
    log_probs = _log_probs(raw, dtype, out_variant=out_variant)
    assert log_probs.dtype == dtype
    assert log_probs.device == raw.device
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = [T] * N
    target_lengths = [S] * N

    ref_log_probs = _reference_probs(log_probs, out_variant=out_variant)
    ref_targets = utils.to_reference(targets)

    gems_args = (log_probs, targets, input_lengths, target_lengths)
    ref_args = (ref_log_probs, ref_targets, input_lengths, target_lengths)
    return gems_args, ref_args


@pytest.mark.underscore_ctc_loss
@pytest.mark.parametrize("T", [50] if QUICK_MODE else [50, 100])
@pytest.mark.parametrize("N", [16] if QUICK_MODE else [16, 32])
@pytest.mark.parametrize("S", [30] if QUICK_MODE else [20, 30])
@pytest.mark.parametrize("C", [20] if QUICK_MODE else [20, 40])
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_accuracy(T, N, S, C, dtype):
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype)

    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(*ref_args, 0, False)
    res_nll, res_log_alpha = flag_gems._ctc_loss(*gems_args, 0, False)

    # _ctc_loss returns (neg_log_likelihood, log_alpha); both must match.
    # log_alpha accumulates over T timesteps, so its absolute error grows with
    # the recursion depth: measured at most ~2e-4 on the largest shape here.
    # A small fixed factor covers that while still catching real regressions;
    # scaling by T * (S + 1) as tests/test_ctc_loss.py does would push the
    # tolerance past 0.1 and stop discriminating.
    reduce_dim = 8
    _assert_ctc_result(res_nll, ref_nll, dtype, reduce_dim=reduce_dim)
    _assert_ctc_result(res_log_alpha, ref_log_alpha, dtype, reduce_dim=reduce_dim)


@pytest.mark.underscore_ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_blank_and_zero_infinity(dtype):
    """A non-zero blank index and an unreachable alignment must both hold up."""
    T, N, S, C = 8, 3, 4, 6
    raw = torch.randn(T, N, C, dtype=torch.float32, device=flag_gems.device)
    log_probs = _log_probs(raw, dtype)
    # keep targets clear of the blank symbol at index C - 1
    targets = torch.randint(0, C - 1, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = [T] * N
    target_lengths = [S] * N

    ref_log_probs = _reference_probs(log_probs)
    ref_targets = utils.to_reference(targets)

    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths, C - 1, True
    )
    res_nll, res_log_alpha = flag_gems._ctc_loss(
        log_probs, targets, input_lengths, target_lengths, C - 1, True
    )

    reduce_dim = 8
    _assert_ctc_result(res_nll, ref_nll, dtype, reduce_dim=reduce_dim)
    _assert_ctc_result(res_log_alpha, ref_log_alpha, dtype, reduce_dim=reduce_dim)


@pytest.mark.underscore_ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_tensor_overload(dtype):
    """The .Tensor overload takes the lengths as tensors instead of int lists."""
    T, N, S, C = 20, 4, 8, 10
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype)

    gems_lengths = (
        torch.tensor(gems_args[2], dtype=torch.long, device=flag_gems.device),
        torch.tensor(gems_args[3], dtype=torch.long, device=flag_gems.device),
    )
    ref_lengths = tuple(utils.to_reference(x) for x in gems_lengths)

    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss.Tensor(
        ref_args[0], ref_args[1], *ref_lengths, 0, False
    )
    res_nll, res_log_alpha = flag_gems._ctc_loss(
        gems_args[0], gems_args[1], *gems_lengths, 0, False
    )

    reduce_dim = 8
    _assert_ctc_result(res_nll, ref_nll, dtype, reduce_dim=reduce_dim)
    _assert_ctc_result(res_log_alpha, ref_log_alpha, dtype, reduce_dim=reduce_dim)


def _empty_output(shape, dtype):
    if flag_gems.vendor_name in ("kunlunxin", "ascend") and dtype == torch.float64:
        return _to_target_double(torch.empty(shape, dtype=dtype, device="cpu"))
    return torch.empty(shape, dtype=dtype, device=flag_gems.device)


def _ctc_reference(ref_args):
    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(*ref_args, 0, False)
    if flag_gems.vendor_name == "ascend" and not utils.TO_CPU:
        # Ascend pads this native workspace to a multiple of eight. Compare
        # the logical CTC states used by the CPU/CUDA API, not vendor padding.
        logical_width = 2 * max(ref_args[3]) + 1
        assert ref_log_alpha.ndim == 3
        assert ref_log_alpha.shape[-1] >= logical_width
        ref_log_alpha = ref_log_alpha[..., :logical_width]
    return ref_nll, ref_log_alpha


@pytest.mark.underscore_ctc_loss_out
@pytest.mark.parametrize('dtype', CTC_DTYPES)
def test__ctc_loss_out(dtype):
    """The out variant writes both results in place and returns those tensors."""
    T, N, S, C = (20, 4, 8, 10)
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype, out_variant=True)
    ref_nll, ref_log_alpha = _ctc_reference(ref_args)
    # The oracle may be upcast; candidate output buffers keep the input dtype.
    res_out0 = _empty_output(ref_nll.shape, dtype)
    res_out1 = _empty_output(ref_log_alpha.shape, dtype)
    pointer0, pointer1 = res_out0.data_ptr(), res_out1.data_ptr()
    got0, got1 = flag_gems._ctc_loss_out(*gems_args, 0, False, out0=res_out0, out1=res_out1)
    assert got0 is res_out0
    assert got1 is res_out1
    assert got0.data_ptr() == pointer0
    assert got1.data_ptr() == pointer1
    reduce_dim = 8
    _assert_ctc_result(res_out0, ref_nll, dtype, reduce_dim=reduce_dim)
    _assert_ctc_result(res_out1, ref_log_alpha, dtype, reduce_dim=reduce_dim)

@pytest.mark.underscore_ctc_loss_out
@pytest.mark.parametrize('dtype', CTC_DTYPES)
def test__ctc_loss_out_resize(dtype):
    """Empty out tensors must be resized to the produced shapes."""
    T, N, S, C = (20, 4, 8, 10)
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype, out_variant=True)
    ref_nll, ref_log_alpha = _ctc_reference(ref_args)
    res_out0 = _empty_output((0,), dtype)
    res_out1 = _empty_output((0,), dtype)
    flag_gems._ctc_loss_out(*gems_args, 0, False, out0=res_out0, out1=res_out1)
    assert res_out0.shape == ref_nll.shape
    assert res_out1.shape == ref_log_alpha.shape
    reduce_dim = 8
    _assert_ctc_result(res_out0, ref_nll, dtype, reduce_dim=reduce_dim)
    _assert_ctc_result(res_out1, ref_log_alpha, dtype, reduce_dim=reduce_dim)
