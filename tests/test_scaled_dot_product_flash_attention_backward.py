# Copyright 2026, The FlagOS Contributors.
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

"""Accuracy tests for the ``_scaled_dot_product_flash_attention_backward``
operator (registered as ``scaled_dot_product_flash_attention_backward``).

The backward op is exercised through the same tensors the ATen interface passes
around (``[B, H, S, D]`` layout): the FlagGems forward
``_scaled_dot_product_flash_attention`` followed by
``scaled_dot_product_flash_attention_backward``.

Causal is only combined with equal ``q_seq_len == kv_seq_len`` shapes: the fused
flash kernels align the causal diagonal between query and key and therefore match
``torch.nn.functional.scaled_dot_product_attention`` semantics exactly for that
case.
"""

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

device = flag_gems.device

if QUICK_MODE:
    BACKWARD_SHAPES = [(1, 2, 2, 64, 64, 64, False)]
else:
    BACKWARD_SHAPES = [
        (1, 2, 2, 64, 64, 64, False),
        (1, 2, 2, 64, 64, 64, True),
        (1, 2, 2, 64, 96, 64, False),
        (1, 2, 2, 96, 64, 64, False),
        (2, 4, 4, 128, 128, 128, False),
        (2, 4, 4, 128, 128, 128, True),
        (2, 4, 2, 128, 128, 64, False),
        (2, 4, 2, 128, 128, 64, True),
    ]
BACKWARD_DTYPES = [torch.float16, torch.bfloat16]


def make_input(
    batch,
    num_q_head,
    num_kv_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    dtype,
    current_device,
):
    random_utils.set_philox_state(
        42 if flag_gems.vendor_name == "cambricon" else 1234567890, 0, current_device
    )
    q_shape = (batch, num_q_head, q_seq_len, head_size)
    kv_shape = (batch, num_kv_head, kv_seq_len, head_size)
    q = torch.empty(q_shape, dtype=dtype, device=current_device).uniform_(-0.05, 0.05)
    k = torch.empty(kv_shape, dtype=dtype, device=current_device).uniform_(-0.05, 0.05)
    v = torch.empty(kv_shape, dtype=dtype, device=current_device).uniform_(-0.05, 0.05)
    return q, k, v


def _sdpa_autograd_grads(q, k, v, grad_out, scale, is_causal, enable_gqa):
    """Reference SDPA gradients computed through PyTorch's own autograd."""
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    out.backward(grad_out)
    return q.grad, k.grad, v.grad


@pytest.mark.scaled_dot_product_flash_attention_backward
@pytest.mark.parametrize(
    "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, is_causal",
    BACKWARD_SHAPES,
)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_scaled_dot_product_flash_attention_backward(
    batch,
    num_q_head,
    num_kv_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    dtype,
):
    current_device = torch_device_fn.current_device()
    scale = float(1.0 / np.sqrt(head_size))
    q, k, v = make_input(
        batch,
        num_q_head,
        num_kv_head,
        q_seq_len,
        kv_seq_len,
        head_size,
        dtype,
        current_device,
    )
    enable_gqa = num_q_head != num_kv_head

    # Reference gradients from PyTorch's own SDPA autograd. By default the
    # reference runs on the GPU at the same dtype as the operator under test;
    # with ``--ref cpu`` (TO_CPU) it is computed through the CPU math backend
    # in float64 for higher precision.
    grad_out = torch.randn_like(q)
    if utils.TO_CPU:
        ref_q = utils.to_reference(q, True)
        ref_k = utils.to_reference(k, True)
        ref_v = utils.to_reference(v, True)
        ref_go = utils.to_reference(grad_out, True)
    else:
        ref_q, ref_k, ref_v, ref_go = q, k, v, grad_out
    ref_q_g, ref_k_g, ref_v_g = _sdpa_autograd_grads(
        ref_q, ref_k, ref_v, ref_go, scale, is_causal, enable_gqa
    )

    # Gradients through FlagGems' flash-attention forward/backward pair.
    philox_seed = torch.empty(0, dtype=torch.long, device=current_device)
    philox_offset = torch.empty(0, dtype=torch.long, device=current_device)
    out, logsumexp = flag_gems._scaled_dot_product_flash_attention(
        q, k, v, 0.0, is_causal, False, scale=scale
    )[:2]
    dq, dk, dv = flag_gems.scaled_dot_product_flash_attention_backward(
        grad_out,
        q,
        k,
        v,
        out,
        logsumexp,
        None,
        None,
        q_seq_len,
        kv_seq_len,
        0.0,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )

    utils.gems_assert_close(dq, ref_q_g, dtype, equal_nan=True)
    utils.gems_assert_close(dk, ref_k_g, dtype, equal_nan=True)
    # dV is more sensitive to softmax recomputation errors in the flash
    # backward (no centering term), mirroring the existing SDPA tests.
    if enable_gqa:
        if dtype == torch.bfloat16:
            v_atol = 2e-2
        else:
            v_atol = 4e-3
    else:
        if dtype == torch.bfloat16:
            v_atol = 5e-3
        else:
            v_atol = 2e-3
    utils.gems_assert_close(dv, ref_v_g, dtype, equal_nan=True, atol=v_atol)
