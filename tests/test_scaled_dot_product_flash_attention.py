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

import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device

# Input shapes: (batch, num_head, q_seq_len, kv_seq_len, head_size).
# Tensors are in (B, H, S, D) layout — the ATen _scaled_dot_product_flash_attention
# convention (heads-first), which differs from the seqlen-first layout used internally
# by flash_attention_forward.
if cfg.QUICK_MODE:
    SHAPES = [
        (1, 8, 128, 128, 64),
    ]
    CAUSAL_CHOICES = [False]
    FLOAT_DTYPES = [torch.float16]
else:
    SHAPES = [
        # square seqlen, various head sizes
        (1, 8, 128, 128, 64),
        (2, 8, 256, 256, 128),
        (4, 16, 512, 512, 64),
        (2, 4, 1024, 1024, 128),
        # non-square Q/K seqlen
        (1, 4, 128, 256, 64),
        (2, 8, 17, 1030, 128),
        (1, 1, 128, 2048, 64),
        # single-token decode (split-KV like)
        (1, 4, 1, 1024, 128),
    ]
    CAUSAL_CHOICES = [False, True]
    FLOAT_DTYPES = [torch.float16, torch.bfloat16]


def make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device):
    """Create Q/K/V in (B, H, S, D) layout — ATen SDPA heads-first convention."""
    random_utils.set_philox_state(1234567890, 0, device)
    q = torch.empty(
        (batch, num_head, q_seq_len, head_size), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, num_head, kv_seq_len, head_size), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, num_head, kv_seq_len, head_size), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    return q, k, v


@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.parametrize("batch, num_head, q_seq_len, kv_seq_len, head_size", SHAPES)
@pytest.mark.parametrize("is_causal", CAUSAL_CHOICES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scaled_dot_product_flash_attention(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    # is_causal requires q_seq_len <= kv_seq_len; skip mismatched shapes.
    if is_causal and q_seq_len > kv_seq_len:
        pytest.skip("is_causal requires q_seq_len <= kv_seq_len")

    device = torch_device_fn.current_device()
    scale = float(1.0 / math.sqrt(head_size))

    q, k, v = make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device)

    # Reference: PyTorch native _scaled_dot_product_flash_attention.
    # Called directly (no flag_gems.enable()) to avoid routing to our impl.
    ref_result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q, k, v, 0.0, is_causal, False, scale=scale
    )
    ref_out = ref_result[0]  # output tensor, shape (B, H, S, D)

    # FlagGems implementation (direct call, not via dispatcher).
    gems_result = flag_gems.scaled_dot_product_flash_attention(
        q, k, v, 0.0, is_causal, False, scale=scale
    )
    gems_out = gems_result[0]  # output tensor, shape (B, H, S, D)

    utils.gems_assert_close(gems_out, ref_out, dtype)
