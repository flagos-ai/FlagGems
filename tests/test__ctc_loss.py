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


def _create_valid_state_mask(target_lengths, max_target, batch_size, T):
    """Create a mask for valid states in log_alpha output.

    Invalid states (beyond 2*target_len+1) contain uninitialized or
    implementation-specific values and should not be compared.
    """
    state_counts = 2 * target_lengths + 1
    mask = torch.zeros(
        (batch_size, T, 2 * max_target + 1),
        dtype=torch.bool,
        device=target_lengths.device,
    )
    for i in range(batch_size):
        mask[i, :, : state_counts[i]] = True
    return mask


@pytest.mark.ctc_loss_internal
@pytest.mark.parametrize("T", [20, 50])
@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("C", [10, 20])
def test__ctc_loss_accuracy(T, N, C, caplog):
    """Test _ctc_loss operator accuracy."""
    log_probs = torch.randn(T, N, C, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

    targets_list = []
    target_lengths = torch.randint(1, T // 2, (N,), device=flag_gems.device)

    for i in range(N):
        tgt_len = target_lengths[i].item()
        targets_list.extend(torch.randint(1, C, (tgt_len,)).tolist())

    targets = torch.tensor(targets_list, dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    # Prepare reference inputs
    ref_log_probs = utils.to_reference(log_probs, upcast=False)
    ref_targets = utils.to_reference(targets)
    ref_input_lengths = utils.to_reference(input_lengths)
    ref_target_lengths = utils.to_reference(target_lengths)

    # Reference computation
    ref_out = torch.ops.aten._ctc_loss(
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        blank=0,
        zero_infinity=False,
    )

    # FlagGems computation
    with caplog.at_level("DEBUG", logger="flag_gems.ops._ctc_loss"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten._ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                zero_infinity=False,
            )

    assert "GEMS _CTC_LOSS" in caplog.text

    # Verify results
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2

    # Compare neg_log_likelihood
    utils.gems_assert_close(
        res_out[0], ref_out[0], dtype=log_probs.dtype, equal_nan=True
    )

    # Compare log_alpha only for valid states
    max_target = target_lengths.max().item()
    mask = _create_valid_state_mask(target_lengths, max_target, N, T)
    ref_mask = utils.to_reference(mask)

    res_la_valid = res_out[1][mask]
    ref_la_valid = ref_out[1][ref_mask]
    utils.gems_assert_close(
        res_la_valid, ref_la_valid, dtype=log_probs.dtype, equal_nan=True
    )


@pytest.mark.ctc_loss_internal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__ctc_loss_dtypes(dtype):
    """Test _ctc_loss with different dtypes."""
    T, N, C = 30, 4, 15

    log_probs = torch.randn(T, N, C, device=flag_gems.device, dtype=dtype)
    log_probs = torch.nn.functional.log_softmax(log_probs.float(), dim=-1).to(dtype)

    targets_list = []
    target_lengths = torch.randint(5, 15, (N,), device=flag_gems.device)

    for i in range(N):
        tgt_len = target_lengths[i].item()
        targets_list.extend(torch.randint(1, C, (tgt_len,)).tolist())

    targets = torch.tensor(targets_list, dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_log_probs = utils.to_reference(log_probs, upcast=False).to(torch.float32)
    ref_targets = utils.to_reference(targets)
    ref_input_lengths = utils.to_reference(input_lengths)
    ref_target_lengths = utils.to_reference(target_lengths)

    ref_out = torch.ops.aten._ctc_loss(
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten._ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

    utils.gems_assert_close(res_out[0], ref_out[0], dtype=dtype, equal_nan=True)

    max_target = target_lengths.max().item()
    mask = _create_valid_state_mask(target_lengths, max_target, N, T)
    ref_mask = utils.to_reference(mask)
    utils.gems_assert_close(
        res_out[1][mask], ref_out[1][ref_mask], dtype=dtype, equal_nan=True
    )


@pytest.mark.ctc_loss_internal
def test__ctc_loss_2d_targets(caplog):
    """Test _ctc_loss with 2D padded targets (Tensor variant)."""
    T, N, C = 50, 8, 20

    log_probs = torch.randn(T, N, C, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

    max_target_len = 25
    targets = torch.randint(1, C, (N, max_target_len), device=flag_gems.device)
    target_lengths = torch.randint(10, max_target_len, (N,), device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_log_probs = utils.to_reference(log_probs, upcast=False).to(torch.float32)
    ref_targets = utils.to_reference(targets)
    ref_input_lengths = utils.to_reference(input_lengths)
    ref_target_lengths = utils.to_reference(target_lengths)

    ref_out = torch.ops.aten._ctc_loss.Tensor(
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
    )

    with caplog.at_level("DEBUG", logger="flag_gems.ops._ctc_loss"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten._ctc_loss.Tensor(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

    assert "GEMS _CTC_LOSS" in caplog.text

    utils.gems_assert_close(
        res_out[0], ref_out[0], dtype=log_probs.dtype, equal_nan=True
    )

    max_target = target_lengths.max().item()
    mask = _create_valid_state_mask(target_lengths, max_target, N, T)
    ref_mask = utils.to_reference(mask)
    utils.gems_assert_close(
        res_out[1][mask], ref_out[1][ref_mask], dtype=log_probs.dtype, equal_nan=True
    )


@pytest.mark.ctc_loss_internal
def test__ctc_loss_int_list_lengths():
    """The default/out schemas take int[] lengths, not tensors."""
    T, N, C = 20, 3, 10

    log_probs = torch.randn(T, N, C, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

    target_lengths = [4, 7, 5]
    input_lengths = [T] * N
    targets = torch.randint(
        1, C, (sum(target_lengths),), dtype=torch.long, device=flag_gems.device
    )

    ref_log_probs = utils.to_reference(log_probs, upcast=False).to(torch.float32)
    ref_targets = utils.to_reference(targets)

    ref_out = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten._ctc_loss(
            log_probs, targets, input_lengths, target_lengths
        )

    utils.gems_assert_close(
        res_out[0], ref_out[0], dtype=log_probs.dtype, equal_nan=True
    )

    length_tensor = torch.tensor(target_lengths, device=flag_gems.device)
    mask = _create_valid_state_mask(length_tensor, max(target_lengths), N, T)
    ref_mask = utils.to_reference(mask)
    utils.gems_assert_close(
        res_out[1][mask],
        ref_out[1][ref_mask],
        dtype=log_probs.dtype,
        equal_nan=True,
    )


@pytest.mark.ctc_loss_internal
def test__ctc_loss_out_variant_out(caplog):
    """The .out variant must write into and return the caller's buffers."""
    T, N, C = 20, 3, 10

    log_probs = torch.randn(T, N, C, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

    target_lengths = [4, 7, 5]
    input_lengths = [T] * N
    max_target = max(target_lengths)

    targets = torch.randint(
        1, C, (sum(target_lengths),), dtype=torch.long, device=flag_gems.device
    )

    ref_log_probs = utils.to_reference(log_probs, upcast=False).to(torch.float32)
    ref_targets = utils.to_reference(targets)

    ref_out = torch.ops.aten._ctc_loss.out(
        ref_log_probs,
        ref_targets,
        input_lengths,
        target_lengths,
        out0=torch.empty(0, device=ref_log_probs.device),
        out1=torch.empty(0, device=ref_log_probs.device),
    )

    out0 = torch.empty(0, device=flag_gems.device)
    out1 = torch.empty(0, device=flag_gems.device)
    with caplog.at_level("DEBUG", logger="flag_gems.ops._ctc_loss"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten._ctc_loss.out(
                log_probs, targets, input_lengths, target_lengths, out0=out0, out1=out1
            )

    assert "GEMS _CTC_LOSS" in caplog.text

    assert res_out[0].data_ptr() == out0.data_ptr()
    assert res_out[1].data_ptr() == out1.data_ptr()

    utils.gems_assert_close(
        res_out[0], ref_out[0], dtype=log_probs.dtype, equal_nan=True
    )

    length_tensor = torch.tensor(target_lengths, device=flag_gems.device)
    mask = _create_valid_state_mask(length_tensor, max_target, N, T)
    ref_mask = utils.to_reference(mask)
    utils.gems_assert_close(
        res_out[1][mask],
        ref_out[1][ref_mask],
        dtype=log_probs.dtype,
        equal_nan=True,
    )


@pytest.mark.ctc_loss_internal
def test__ctc_loss_out_variant_Tensor_out(caplog):
    """The .Tensor_out variant must write into and return the caller's buffers."""
    T, N, C = 20, 3, 10

    log_probs = torch.randn(T, N, C, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

    target_lengths = [4, 7, 5]
    input_lengths = [T] * N
    max_target = max(target_lengths)

    targets = torch.randint(
        1, C, (N, max_target), dtype=torch.long, device=flag_gems.device
    )
    input_lengths_t = torch.tensor(input_lengths, device=flag_gems.device)
    target_lengths_t = torch.tensor(target_lengths, device=flag_gems.device)

    ref_log_probs = utils.to_reference(log_probs, upcast=False).to(torch.float32)
    ref_targets = utils.to_reference(targets)
    ref_input_lengths = utils.to_reference(input_lengths_t)
    ref_target_lengths = utils.to_reference(target_lengths_t)

    ref_out = torch.ops.aten._ctc_loss.Tensor_out(
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        out0=torch.empty(0, device=ref_log_probs.device),
        out1=torch.empty(0, device=ref_log_probs.device),
    )

    out0 = torch.empty(0, device=flag_gems.device)
    out1 = torch.empty(0, device=flag_gems.device)
    with caplog.at_level("DEBUG", logger="flag_gems.ops._ctc_loss"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten._ctc_loss.Tensor_out(
                log_probs,
                targets,
                input_lengths_t,
                target_lengths_t,
                out0=out0,
                out1=out1,
            )

    assert "GEMS _CTC_LOSS" in caplog.text

    assert res_out[0].data_ptr() == out0.data_ptr()
    assert res_out[1].data_ptr() == out1.data_ptr()

    utils.gems_assert_close(
        res_out[0], ref_out[0], dtype=log_probs.dtype, equal_nan=True
    )

    length_tensor = torch.tensor(target_lengths, device=flag_gems.device)
    mask = _create_valid_state_mask(length_tensor, max_target, N, T)
    ref_mask = utils.to_reference(mask)
    utils.gems_assert_close(
        res_out[1][mask],
        ref_out[1][ref_mask],
        dtype=log_probs.dtype,
        equal_nan=True,
    )
