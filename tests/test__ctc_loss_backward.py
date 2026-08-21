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

import logging

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close
from .conftest import QUICK_MODE

# CPU doesn't support Half for CTC loss backward
BACKWARD_DTYPES = [torch.float32, torch.float64]


@pytest.mark.ctc_loss_backward_internal
@pytest.mark.parametrize("T", [50] if QUICK_MODE else [50, 100])
@pytest.mark.parametrize("N", [16] if QUICK_MODE else [16, 32])
@pytest.mark.parametrize("S", [30] if QUICK_MODE else [20, 30])
@pytest.mark.parametrize("C", [20] if QUICK_MODE else [20, 40])
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test__ctc_loss_backward_accuracy(T, N, S, C, dtype, caplog):
    # Forward pass to get neg_log_likelihood and log_alpha
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.randint(
        1, S + 1, (N,), dtype=torch.long, device=flag_gems.device
    )

    ref_log_probs = log_probs.cpu().clone().detach()
    ref_targets = targets.cpu()
    ref_input_lengths = input_lengths.cpu()
    ref_target_lengths = target_lengths.cpu()

    # Forward on CPU reference
    ref_neg_log_likelihood, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, ref_input_lengths, ref_target_lengths
    )

    # Forward on CUDA
    with flag_gems.use_gems():
        neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
            log_probs, targets, input_lengths, target_lengths
        )

    # Upstream gradient
    grad_output = torch.randn_like(neg_log_likelihood)
    ref_grad_output = grad_output.cpu()

    # Backward on CPU reference
    ref_grad = torch.ops.aten._ctc_loss_backward(
        ref_grad_output,
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        ref_neg_log_likelihood,
        ref_log_alpha,
        blank=0,
    )

    # Backward on CUDA with FlagGems
    with flag_gems.use_gems():
        with caplog.at_level(logging.DEBUG):
            res_grad = torch.ops.aten._ctc_loss_backward(
                grad_output,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                neg_log_likelihood,
                log_alpha,
                blank=0,
            )

    assert "GEMS _CTC_LOSS_BACKWARD" in caplog.text
    gems_assert_close(res_grad.cpu(), ref_grad, dtype, equal_nan=True)


@pytest.mark.ctc_loss_backward_internal
@pytest.mark.parametrize("T", [64])
@pytest.mark.parametrize("N", [8])
@pytest.mark.parametrize("S", [32])
@pytest.mark.parametrize("C", [28])
@pytest.mark.parametrize("dtype", [torch.float32])
def test__ctc_loss_backward_int_list_lengths(T, N, S, C, dtype, caplog):
    """Test backward with int[] lengths (default overload)."""
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = [T] * N
    target_lengths = [S - i % 5 for i in range(N)]

    ref_log_probs = log_probs.cpu().clone().detach()
    ref_targets = targets.cpu()

    # Forward on CPU
    ref_neg_log_likelihood, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    # Forward on CUDA
    with flag_gems.use_gems():
        neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
            log_probs, targets, input_lengths, target_lengths
        )

    grad_output = torch.randn_like(neg_log_likelihood)
    ref_grad_output = grad_output.cpu()

    # Backward on CPU (int[] overload)
    ref_grad = torch.ops.aten._ctc_loss_backward(
        ref_grad_output,
        ref_log_probs,
        ref_targets,
        input_lengths,
        target_lengths,
        ref_neg_log_likelihood,
        ref_log_alpha,
        blank=0,
    )

    # Backward on CUDA (int[] overload)
    with flag_gems.use_gems():
        with caplog.at_level(logging.DEBUG):
            res_grad = torch.ops.aten._ctc_loss_backward(
                grad_output,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                neg_log_likelihood,
                log_alpha,
                blank=0,
            )

    assert "GEMS _CTC_LOSS_BACKWARD" in caplog.text
    gems_assert_close(res_grad.cpu(), ref_grad, dtype, equal_nan=True)


@pytest.mark.ctc_loss_backward_internal
@pytest.mark.parametrize("dtype", [torch.float32])
def test__ctc_loss_backward_out_variant(dtype, caplog):
    """Test .out overload."""
    T, N, S, C = 50, 16, 30, 20
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = [T] * N
    target_lengths = [S - i % 5 for i in range(N)]

    ref_log_probs = log_probs.cpu().clone().detach()
    ref_targets = targets.cpu()

    # Forward
    ref_neg_log_likelihood, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    with flag_gems.use_gems():
        neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
            log_probs, targets, input_lengths, target_lengths
        )

    grad_output = torch.randn_like(neg_log_likelihood)
    ref_grad_output = grad_output.cpu()

    # Backward reference (functional)
    ref_grad = torch.ops.aten._ctc_loss_backward(
        ref_grad_output,
        ref_log_probs,
        ref_targets,
        input_lengths,
        target_lengths,
        ref_neg_log_likelihood,
        ref_log_alpha,
        blank=0,
    )

    # Backward with out= variant
    out = torch.empty_like(log_probs)
    with flag_gems.use_gems():
        with caplog.at_level(logging.DEBUG):
            res_grad = torch.ops.aten._ctc_loss_backward(
                grad_output,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                neg_log_likelihood,
                log_alpha,
                blank=0,
                out=out,
            )

    assert "GEMS _CTC_LOSS_BACKWARD" in caplog.text
    assert res_grad is out
    gems_assert_close(res_grad.cpu(), ref_grad, dtype, equal_nan=True)


@pytest.mark.ctc_loss_backward_internal
@pytest.mark.parametrize("dtype", [torch.float32])
def test__ctc_loss_backward_zero_infinity(dtype):
    """Test zero_infinity flag."""
    T, N, S, C = 50, 16, 30, 20
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.randint(
        10, T + 1, (N,), dtype=torch.long, device=flag_gems.device
    )
    target_lengths = torch.randint(
        5, S + 1, (N,), dtype=torch.long, device=flag_gems.device
    )

    # Make some sequences impossible (target longer than input)
    input_lengths[0] = 5
    target_lengths[0] = 20

    ref_log_probs = log_probs.cpu().clone().detach()
    ref_targets = targets.cpu()
    ref_input_lengths = input_lengths.cpu()
    ref_target_lengths = target_lengths.cpu()

    # Forward with zero_infinity
    ref_neg_log_likelihood, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        zero_infinity=True,
    )

    with flag_gems.use_gems():
        neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
            log_probs, targets, input_lengths, target_lengths, zero_infinity=True
        )

    grad_output = torch.randn_like(neg_log_likelihood)
    ref_grad_output = grad_output.cpu()

    # Backward with zero_infinity
    ref_grad = torch.ops.aten._ctc_loss_backward(
        ref_grad_output,
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        ref_neg_log_likelihood,
        ref_log_alpha,
        blank=0,
        zero_infinity=True,
    )

    with flag_gems.use_gems():
        res_grad = torch.ops.aten._ctc_loss_backward(
            grad_output,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            neg_log_likelihood,
            log_alpha,
            blank=0,
            zero_infinity=True,
        )

    gems_assert_close(res_grad.cpu(), ref_grad, dtype, equal_nan=True)


@pytest.mark.ctc_loss_backward_internal
@pytest.mark.parametrize("dtype", [torch.float32])
def test__ctc_loss_backward_unbatched(dtype):
    """Test unbatched (2D) log_probs - must be 3D for _ctc_loss."""
    T, C, S = 50, 20, 30
    # CTC loss requires 3D input: (T, N, C) where N=1 for unbatched
    log_probs = torch.randn(T, 1, C, dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (1, S), dtype=torch.long, device=flag_gems.device)
    input_length = T
    target_length = S

    ref_log_probs = log_probs.cpu().clone().detach()
    ref_targets = targets.cpu()

    # Forward unbatched
    ref_neg_log_likelihood, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, [input_length], [target_length]
    )

    with flag_gems.use_gems():
        neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
            log_probs, targets, [input_length], [target_length]
        )

    grad_output = torch.randn_like(neg_log_likelihood)
    ref_grad_output = grad_output.cpu()

    # Backward unbatched
    ref_grad = torch.ops.aten._ctc_loss_backward(
        ref_grad_output,
        ref_log_probs,
        ref_targets,
        [input_length],
        [target_length],
        ref_neg_log_likelihood,
        ref_log_alpha,
        blank=0,
    )

    with flag_gems.use_gems():
        res_grad = torch.ops.aten._ctc_loss_backward(
            grad_output,
            log_probs,
            targets,
            [input_length],
            [target_length],
            neg_log_likelihood,
            log_alpha,
            blank=0,
        )

    gems_assert_close(res_grad.cpu(), ref_grad, dtype, equal_nan=True)
