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

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SHAPES = (
    (1, 1, 1),
    (17, 8, 2),
    (257, 16, 4),
    (1024, 64, 2),
)


def _reference(gate_logits, top_k, attention_mask=None):
    """Compute Ne * sum_i(f_i * P_i) on the device selected by ``--ref``."""
    logits = utils.to_reference(gate_logits.detach(), True)
    num_tokens, num_experts = logits.shape
    probabilities = torch.softmax(logits, dim=-1)
    selected_experts = torch.topk(logits, top_k, dim=-1).indices

    if attention_mask is None:
        valid = torch.ones(num_tokens, dtype=logits.dtype, device=logits.device)
    else:
        valid = utils.to_reference(attention_mask.detach().reshape(-1)).to(logits.dtype)

    total_valid = valid.sum()
    if total_valid == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)

    counts = torch.zeros(
        num_experts,
        dtype=logits.dtype,
        device=logits.device,
    )
    counts.scatter_add_(
        0,
        selected_experts.reshape(-1),
        valid.repeat_interleave(top_k),
    )
    frequency = counts / total_valid
    mean_probability = (probabilities * valid.unsqueeze(-1)).sum(dim=0) / total_valid
    return num_experts * torch.sum(frequency * mean_probability)


def _assert_close(actual, expected, num_experts):
    # The operator contract returns float32 regardless of the logits dtype.
    utils.gems_assert_close(
        actual,
        expected,
        torch.float32,
        reduce_dim=num_experts,
    )


@pytest.mark.moe_load_balance_loss
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("num_tokens,num_experts,top_k", SHAPES)
def test_moe_load_balance_loss_matches_reference(
    num_tokens,
    num_experts,
    top_k,
    masked,
    dtype,
):
    torch.manual_seed(2026)
    logits = torch.randn(
        num_tokens,
        num_experts,
        dtype=dtype,
        device=flag_gems.device,
    )
    attention_mask = None
    if masked:
        attention_mask = torch.ones(
            1,
            num_tokens,
            dtype=torch.int64,
            device=flag_gems.device,
        )
        attention_mask[..., 1::4] = 0

    actual = flag_gems.moe_load_balance_loss(
        logits,
        top_k=top_k,
        attention_mask=attention_mask,
    )
    expected = _reference(logits, top_k, attention_mask)

    assert actual.shape == torch.Size([])
    assert actual.dtype == torch.float32
    _assert_close(actual, expected, num_experts)


@pytest.mark.moe_load_balance_loss
@pytest.mark.parametrize("dtype", DTYPES)
def test_tle_path_with_irregular_large_shape(dtype):
    """Exercise the pipelined histogram path without a benchmark-specific shape."""
    torch.manual_seed(2026)
    logits = torch.randn(2051, 96, dtype=dtype, device=flag_gems.device)
    attention_mask = torch.ones(2051, dtype=torch.bool, device=flag_gems.device)
    attention_mask[5::7] = False

    expected = _reference(logits, 5, attention_mask)
    actual = flag_gems.moe_load_balance_loss(logits, 5, attention_mask)

    _assert_close(actual, expected, logits.shape[1])


@pytest.mark.moe_load_balance_loss
@pytest.mark.parametrize("dtype", DTYPES)
def test_noncontiguous_logits_and_mask(dtype):
    torch.manual_seed(2027)
    logits = torch.randn(
        97,
        24,
        dtype=dtype,
        device=flag_gems.device,
    )[:, ::2]
    mask_storage = torch.ones(
        97,
        2,
        dtype=torch.uint8,
        device=flag_gems.device,
    )
    mask_storage[::5, 0] = 0
    attention_mask = mask_storage[:, 0]
    assert not logits.is_contiguous()
    assert not attention_mask.is_contiguous()

    actual = flag_gems.moe_load_balance_loss(logits, 3, attention_mask)
    expected = _reference(logits, 3, attention_mask)
    _assert_close(actual, expected, logits.shape[1])


@pytest.mark.moe_load_balance_loss
@pytest.mark.parametrize("dtype", DTYPES)
def test_all_masked_tokens_return_zero(dtype):
    logits = torch.randn(33, 8, dtype=dtype, device=flag_gems.device)
    attention_mask = torch.zeros(
        33,
        dtype=torch.bool,
        device=flag_gems.device,
    )
    actual = flag_gems.moe_load_balance_loss(logits, 2, attention_mask)
    assert actual.item() == 0.0


@pytest.mark.moe_load_balance_loss
def test_default_top_k_is_two():
    logits = torch.randn(19, 8, device=flag_gems.device)
    actual = flag_gems.moe_load_balance_loss(logits)
    expected = _reference(logits, 2)
    _assert_close(actual, expected, logits.shape[1])


@pytest.mark.moe_load_balance_loss
def test_moe_load_balance_loss_validation():
    device = flag_gems.device
    logits = torch.randn(8, 4, device=device)

    with pytest.raises(ValueError, match=r"shape \[T, N_e\]"):
        flag_gems.moe_load_balance_loss(logits.unsqueeze(0))
    with pytest.raises(TypeError, match="supports only"):
        flag_gems.moe_load_balance_loss(logits.long())
    with pytest.raises(ValueError, match="greater than zero"):
        flag_gems.moe_load_balance_loss(torch.empty(0, 4, device=device))
    with pytest.raises(TypeError, match="Python int"):
        flag_gems.moe_load_balance_loss(logits, 2.0)
    with pytest.raises(ValueError, match="1 <= top_k <= N_e"):
        flag_gems.moe_load_balance_loss(logits, 0)
    with pytest.raises(ValueError, match="1 <= top_k <= N_e"):
        flag_gems.moe_load_balance_loss(logits, 5)
    with pytest.raises(ValueError, match="must contain T elements"):
        flag_gems.moe_load_balance_loss(
            logits,
            2,
            torch.ones(7, dtype=torch.bool, device=device),
        )
    with pytest.raises(TypeError, match="attention_mask supports only"):
        flag_gems.moe_load_balance_loss(logits, 2, torch.ones(8, device=device))
