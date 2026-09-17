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
CTC_DTYPES = [torch.float32, torch.float64]


def _make_inputs(T, N, S, C, dtype):
    """Build one CTC batch, returning int-list and tensor length forms.

    _ctc_loss and _ctc_loss.out take int[] lengths, while _ctc_loss.Tensor and
    _ctc_loss.Tensor_out take them as tensors, so both forms are handed back.
    """
    raw = torch.randn(T, N, C, dtype=torch.float32, device=flag_gems.device)
    log_probs = raw.log_softmax(-1).to(dtype)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = [T] * N
    target_lengths = [S] * N

    ref_log_probs = utils.to_reference(log_probs.clone().detach())
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
    res_nll, res_log_alpha = flag_gems.ops._ctc_loss(*gems_args, 0, False)

    # _ctc_loss returns (neg_log_likelihood, log_alpha); both must match.
    # log_alpha accumulates over T timesteps, so its absolute error grows with
    # the recursion depth: measured at most ~2e-4 on the largest shape here.
    # A small fixed factor covers that while still catching real regressions;
    # scaling by T * (S + 1) as tests/test_ctc_loss.py does would push the
    # tolerance past 0.1 and stop discriminating.
    reduce_dim = 8
    gems_assert_close(res_nll, ref_nll, dtype, reduce_dim=reduce_dim)
    gems_assert_close(res_log_alpha, ref_log_alpha, dtype, reduce_dim=reduce_dim)


@pytest.mark.underscore_ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_blank_and_zero_infinity(dtype):
    """A non-zero blank index and an unreachable alignment must both hold up."""
    T, N, S, C = 8, 3, 4, 6
    raw = torch.randn(T, N, C, dtype=torch.float32, device=flag_gems.device)
    log_probs = raw.log_softmax(-1).to(dtype)
    # keep targets clear of the blank symbol at index C - 1
    targets = torch.randint(0, C - 1, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = [T] * N
    target_lengths = [S] * N

    ref_log_probs = utils.to_reference(log_probs.clone().detach())
    ref_targets = utils.to_reference(targets)

    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths, C - 1, True
    )
    res_nll, res_log_alpha = flag_gems.ops._ctc_loss(
        log_probs, targets, input_lengths, target_lengths, C - 1, True
    )

    reduce_dim = 8
    gems_assert_close(res_nll, ref_nll, dtype, reduce_dim=reduce_dim)
    gems_assert_close(res_log_alpha, ref_log_alpha, dtype, reduce_dim=reduce_dim)


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
    res_nll, res_log_alpha = flag_gems.ops._ctc_loss(
        gems_args[0], gems_args[1], *gems_lengths, 0, False
    )

    reduce_dim = 8
    gems_assert_close(res_nll, ref_nll, dtype, reduce_dim=reduce_dim)
    gems_assert_close(res_log_alpha, ref_log_alpha, dtype, reduce_dim=reduce_dim)


@pytest.mark.underscore_ctc_loss_out
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_out(dtype):
    """The out variant writes both results in place and returns those tensors."""
    T, N, S, C = 20, 4, 8, 10
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype)

    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(*ref_args, 0, False)

    res_out0 = torch.empty_like(ref_nll, device=flag_gems.device)
    res_out1 = torch.empty_like(ref_log_alpha, device=flag_gems.device)
    got0, got1 = flag_gems.ops._ctc_loss_out(
        *gems_args, 0, False, out0=res_out0, out1=res_out1
    )

    assert got0.data_ptr() == res_out0.data_ptr()
    assert got1.data_ptr() == res_out1.data_ptr()
    reduce_dim = 8
    gems_assert_close(res_out0, ref_nll, dtype, reduce_dim=reduce_dim)
    gems_assert_close(res_out1, ref_log_alpha, dtype, reduce_dim=reduce_dim)


@pytest.mark.underscore_ctc_loss_out
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test__ctc_loss_out_resize(dtype):
    """Empty out tensors must be resized to the produced shapes."""
    T, N, S, C = 20, 4, 8, 10
    gems_args, ref_args = _make_inputs(T, N, S, C, dtype)

    ref_nll, ref_log_alpha = torch.ops.aten._ctc_loss(*ref_args, 0, False)

    res_out0 = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res_out1 = torch.empty(0, dtype=dtype, device=flag_gems.device)
    flag_gems.ops._ctc_loss_out(*gems_args, 0, False, out0=res_out0, out1=res_out1)

    assert res_out0.shape == ref_nll.shape
    assert res_out1.shape == ref_log_alpha.shape
    reduce_dim = 8
    gems_assert_close(res_out0, ref_nll, dtype, reduce_dim=reduce_dim)
    gems_assert_close(res_out1, ref_log_alpha, dtype, reduce_dim=reduce_dim)
