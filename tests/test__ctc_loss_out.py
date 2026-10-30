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

CTC_DTYPES = [torch.float32, torch.float64]


def _to_target_double(value):
    result = (
        value.contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .to(flag_gems.device)
        .view(torch.float64)
        .reshape(value.shape)
    )
    assert result.dtype == torch.float64
    assert result.shape == value.shape
    if flag_gems.vendor_name == "ascend":
        assert result.device.type == torch.device(flag_gems.device).type
        assert result.device.index == flag_gems.runtime.torch_device_fn.current_device()
        assert result.untyped_storage().nbytes() == value.numel() * value.element_size()
    return result


def _log_probs(raw, dtype):
    if flag_gems.vendor_name == "ascend" and dtype == torch.float64:
        if not utils.TO_CPU:
            pytest.skip("Ascend native CTC Float64 input conversion narrows to Float32")
        return _to_target_double(raw.log_softmax(-1).cpu().double())
    if flag_gems.vendor_name == "iluvatar" and dtype == torch.float64:
        if not utils.TO_CPU:
            pytest.skip(
                "Iluvatar native CTC Float64 output numerics are incorrect; CPU reference remains covered"
            )
        return _to_target_double(raw.log_softmax(-1).cpu().double())
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        if not utils.TO_CPU:
            pytest.skip("Kunlunxin native CTC does not preserve float64 outputs")
        return _to_target_double(raw.log_softmax(-1).cpu().double())
    return raw.log_softmax(-1).to(dtype)


def _double_to_cpu(value):
    raw = (
        torch.empty(0, dtype=torch.uint8, device=value.device)
        .set_(value.untyped_storage(), 0, (value.untyped_storage().nbytes(),), (1,))
        .cpu()
    )
    return raw.view(value.dtype).as_strided(
        value.shape, value.stride(), value.storage_offset()
    )


def _reference_probs(value):
    if flag_gems.vendor_name == "ascend" and value.dtype == torch.float64:
        return _double_to_cpu(value).clone().detach()
    if flag_gems.vendor_name == "iluvatar" and value.dtype == torch.float64:
        return value.cpu().clone().detach()
    # Clone on CPU before Double device clone/cast can narrow the input on KU.
    if flag_gems.vendor_name == "kunlunxin" and value.dtype == torch.float64:
        return value.cpu().clone().detach()
    return utils.to_reference(value.clone().detach(), True)


def _reference_alpha(value, log_probs, target_lengths):
    if flag_gems.vendor_name != "ascend":
        return value
    # NPU native alpha pads the state axis to eight. The canonical candidate
    # keeps the logical 2*S+1 states, so normalize only this reference format.
    assert value.device == log_probs.device
    assert value.dtype == log_probs.dtype
    logical_shape = (
        log_probs.shape[1],
        log_probs.shape[0],
        2 * max(target_lengths) + 1,
    )
    allowed_shapes = {logical_shape}
    if value.dtype == torch.float32:
        n, t, states = logical_shape
        allowed_shapes.add((n, t, ((states + 7) // 8) * 8))
    assert tuple(value.shape) in allowed_shapes
    return value[..., : logical_shape[-1]]


def _assert_ctc_result(result, reference, dtype, *, reduce_dim):
    assert result.device.type == torch.device(flag_gems.device).type
    assert result.device.index == flag_gems.runtime.torch_device_fn.current_device()
    assert result.dtype == dtype
    assert result.shape == reference.shape
    # Observe values only after checking the target contract.
    if flag_gems.vendor_name == "ascend" and dtype == torch.float64:
        gems_assert_close(
            _double_to_cpu(result),
            _double_to_cpu(reference),
            dtype,
            reduce_dim=reduce_dim,
        )
    else:
        gems_assert_close(result.cpu(), reference.cpu(), dtype, reduce_dim=reduce_dim)


def _empty_output(shape, dtype):
    if flag_gems.vendor_name == "ascend" and dtype == torch.float64:
        return _to_target_double(torch.empty(shape, dtype=dtype, device="cpu"))
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        return _to_target_double(torch.empty(shape, dtype=dtype, device="cpu"))
    return torch.empty(shape, dtype=dtype, device=flag_gems.device)


def _make_inputs(T, N, S, C, dtype):
    """Build one CTC batch, returning int-list and tensor length forms.

    _ctc_loss and _ctc_loss.out take int[] lengths, while _ctc_loss.Tensor and
    _ctc_loss.Tensor_out take them as tensors, so both forms are handed back.
    """
    raw = torch.randn(T, N, C, dtype=torch.float32, device=flag_gems.device)
    log_probs = _log_probs(raw, dtype)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = [T] * N
    target_lengths = [S] * N
    ref_log_probs = _reference_probs(log_probs)
    ref_targets = utils.to_reference(targets)
    gems_args = (log_probs, targets, input_lengths, target_lengths)
    ref_args = (ref_log_probs, ref_targets, input_lengths, target_lengths)
    return (gems_args, ref_args)


@pytest.mark.underscore_ctc_loss_out
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_out(dtype):
    """The out variant writes both results in place and returns those tensors."""
    T, N, S, C = (20, 4, 8, 10)
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype)
    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(*ref_args, 0, False)
    ref_log_alpha = _reference_alpha(ref_log_alpha, ref_args[0], ref_args[3])
    # The oracle may be upcast; candidate output buffers keep the input dtype.
    res_out0 = _empty_output(ref_nll.shape, dtype)
    res_out1 = _empty_output(ref_log_alpha.shape, dtype)
    got0, got1 = flag_gems._ctc_loss_out(
        *gems_args, 0, False, out0=res_out0, out1=res_out1
    )
    assert got0 is res_out0
    assert got1 is res_out1
    assert got0.data_ptr() == res_out0.data_ptr()
    assert got1.data_ptr() == res_out1.data_ptr()
    reduce_dim = 8
    _assert_ctc_result(res_out0, ref_nll, dtype, reduce_dim=reduce_dim)
    _assert_ctc_result(res_out1, ref_log_alpha, dtype, reduce_dim=reduce_dim)


@pytest.mark.underscore_ctc_loss_out
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_out_resize(dtype):
    """Empty out tensors must be resized to the produced shapes."""
    T, N, S, C = (20, 4, 8, 10)
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype)
    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(*ref_args, 0, False)
    ref_log_alpha = _reference_alpha(ref_log_alpha, ref_args[0], ref_args[3])
    res_out0 = _empty_output((0,), dtype)
    res_out1 = _empty_output((0,), dtype)
    flag_gems._ctc_loss_out(*gems_args, 0, False, out0=res_out0, out1=res_out1)
    assert res_out0.shape == ref_nll.shape
    assert res_out1.shape == ref_log_alpha.shape
    reduce_dim = 8
    _assert_ctc_result(res_out0, ref_nll, dtype, reduce_dim=reduce_dim)
    _assert_ctc_result(res_out1, ref_log_alpha, dtype, reduce_dim=reduce_dim)
