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

# CPU does not implement CTC loss for Half, so the reference comparison is
# limited to the float types it supports.
BACKWARD_DTYPES = [torch.float32, torch.float64]


def _make_case(T, N, S, C, dtype):
    """Build a CTC batch and the forward outputs its backward pass needs."""
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.randint(
        1, S + 1, (N,), dtype=torch.long, device=flag_gems.device
    )

    ref_log_probs = utils.to_reference(log_probs.clone().detach())
    ref_targets = utils.to_reference(targets)
    ref_input_lengths = utils.to_reference(input_lengths)
    ref_target_lengths = utils.to_reference(target_lengths)

    # _ctc_loss returns (neg_log_likelihood, log_alpha); the backward needs both.
    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, ref_input_lengths, ref_target_lengths
    )
    nll = ref_nll.to(device=log_probs.device, dtype=log_probs.dtype)
    log_alpha = ref_log_alpha.to(device=log_probs.device, dtype=log_probs.dtype)

    grad_output = torch.randn_like(nll)

    # The default and .out overloads take int[] lengths; only .Tensor takes
    # tensors. Build both forms so each overload gets what its schema wants.
    len_list = (input_lengths.tolist(), target_lengths.tolist())

    gems_args = (
        grad_output,
        log_probs,
        targets,
        *len_list,
        nll,
        log_alpha,
    )
    ref_args = (
        utils.to_reference(grad_output),
        ref_log_probs,
        ref_targets,
        *len_list,
        ref_nll,
        ref_log_alpha,
    )
    gems_args_tensor = (
        grad_output,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        nll,
        log_alpha,
    )
    ref_args_tensor = (
        utils.to_reference(grad_output),
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        ref_nll,
        ref_log_alpha,
    )
    return gems_args, ref_args, gems_args_tensor, ref_args_tensor


@pytest.mark.ctc_loss_backward
@pytest.mark.parametrize("T", [50] if QUICK_MODE else [50, 100])
@pytest.mark.parametrize("N", [16] if QUICK_MODE else [16, 32])
@pytest.mark.parametrize("S", [30] if QUICK_MODE else [20, 30])
@pytest.mark.parametrize("C", [20] if QUICK_MODE else [20, 40])
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test__ctc_loss_backward_accuracy(T, N, S, C, dtype):
    gems_args, ref_args, _, _ = _make_case(T, N, S, C, dtype)

    ref_grad = torch.ops.aten._ctc_loss_backward(*ref_args, blank=0)
    res_grad = flag_gems.ops._ctc_loss_backward(*gems_args, blank=0)

    assert res_grad.shape == ref_grad.shape
    gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.ctc_loss_backward
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test__ctc_loss_backward_zero_infinity(dtype):
    """zero_infinity must zero the gradient of unreachable alignments.

    A target longer than the input cannot be aligned, so the loss is infinite
    and the gradient has to come back finite (zeroed) rather than NaN.
    """
    T, N, S, C = 4, 2, 6, 5
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    # target_lengths > input_lengths makes the alignment impossible
    target_lengths = torch.full((N,), S, dtype=torch.long, device=flag_gems.device)

    len_args = (input_lengths.tolist(), target_lengths.tolist())

    ref_log_probs = utils.to_reference(log_probs.clone().detach())
    ref_targets = utils.to_reference(targets)
    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, *len_args, 0, True
    )
    nll = ref_nll.to(device=log_probs.device, dtype=log_probs.dtype)
    log_alpha = ref_log_alpha.to(device=log_probs.device, dtype=log_probs.dtype)

    grad_output = torch.randn_like(nll)

    ref_grad = torch.ops.aten._ctc_loss_backward(
        utils.to_reference(grad_output),
        ref_log_probs,
        ref_targets,
        *len_args,
        ref_nll,
        ref_log_alpha,
        blank=0,
        zero_infinity=True,
    )
    res_grad = flag_gems.ops._ctc_loss_backward(
        grad_output,
        log_probs,
        targets,
        *len_args,
        nll,
        log_alpha,
        blank=0,
        zero_infinity=True,
    )

    assert torch.isfinite(res_grad).all()
    gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.ctc_loss_backward
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test__ctc_loss_backward_tensor_overload(dtype):
    """The .Tensor overload takes the lengths as tensors instead of int lists."""
    _, _, gems_args, ref_args = _make_case(20, 4, 8, 10, dtype)

    ref_grad = torch.ops.aten._ctc_loss_backward.Tensor(*ref_args, blank=0)
    res_grad = flag_gems.ops._ctc_loss_backward(*gems_args, blank=0)

    gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.ctc_loss_backward_out
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test__ctc_loss_backward_out(dtype):
    gems_args, ref_args, _, _ = _make_case(20, 4, 8, 10, dtype)

    ref_out = torch.empty_like(ref_args[1])
    torch.ops.aten._ctc_loss_backward.out(*ref_args, blank=0, out=ref_out)

    res_out = torch.empty_like(gems_args[1])
    returned = flag_gems.ops._ctc_loss_backward_out(*gems_args, blank=0, out=res_out)

    # the out variant must write in place and hand back the same tensor
    assert returned.data_ptr() == res_out.data_ptr()
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ctc_loss_backward_out
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test__ctc_loss_backward_out_resize(dtype):
    """An empty ``out`` must be resized to the gradient shape."""
    gems_args, ref_args, _, _ = _make_case(20, 4, 8, 10, dtype)

    # ref_args[1] is the reference log_probs, so its device follows --ref
    ref_out = torch.empty(0, dtype=dtype, device=ref_args[1].device)
    torch.ops.aten._ctc_loss_backward.out(*ref_args, blank=0, out=ref_out)

    res_out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    flag_gems.ops._ctc_loss_backward_out(*gems_args, blank=0, out=res_out)

    assert res_out.shape == gems_args[1].shape
    gems_assert_close(res_out, ref_out, dtype)
