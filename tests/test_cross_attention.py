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
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils

DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# Reuse the numeric coverage from test_scaled_dot_product_attention.py's
# LEGACY_SHAPES. SDPA's (B, Hq, Hkv, Lq, Lk, D, enable_gqa) maps to this
# operator's (B, Hq, Hkv, Lq, Lk, Dqk, Dv), with Dqk == Dv == D. GQA/MQA is
# implied by Hq != Hkv, so the enable_gqa flag is not part of this contract.
CROSS_ATTENTION_SHAPES = (
    (1, 16, 16, 16, 16, 128, 128),
    (4, 8, 8, 128, 128, 64, 64),
    (2, 16, 16, 256, 128, 64, 64),
    (2, 32, 32, 128, 512, 128, 128),
    (1, 16, 16, 257, 513, 128, 128),
    (4, 8, 8, 1024, 1024, 64, 64),
    (4, 8, 8, 1024, 1024, 128, 128),
    (4, 8, 8, 2048, 256, 64, 64),
    (4, 8, 8, 2048, 256, 128, 128),
    (4, 8, 8, 17, 1030, 64, 64),
    (4, 8, 8, 17, 1030, 128, 128),
    (2, 4, 4, 512, 612, 128, 128),
    (2, 4, 4, 1024, 1034, 64, 64),
    (2, 4, 4, 2048, 2048, 32, 32),
    (2, 4, 4, 4096, 4096, 16, 16),
    (2, 4, 4, 4001, 4001, 32, 32),
    (2, 4, 4, 4001, 4096, 64, 64),
    (2, 4, 4, 4096, 4000, 128, 128),
    (1, 2, 2, 8192, 8202, 16, 16),
    (1, 2, 2, 8192, 8192, 32, 32),
    (2, 4, 2, 512, 612, 128, 128),
    (2, 4, 1, 1024, 1034, 64, 64),
    (2, 4, 2, 2048, 2048, 32, 32),
    (2, 4, 1, 4096, 4096, 16, 16),
    (2, 4, 2, 4001, 4001, 32, 32),
    (2, 4, 1, 4001, 4096, 64, 64),
    (2, 4, 2, 4096, 4000, 128, 128),
    (1, 2, 1, 8192, 8202, 16, 16),
    (1, 2, 1, 8192, 8192, 32, 32),
)


def _reference(query, key, value, attn_mask=None, scale=None):
    ref_query = utils.to_reference(query, False)
    ref_key = utils.to_reference(key, False)
    ref_value = utils.to_reference(value, False)
    ref_mask = utils.to_reference(attn_mask, False)

    # PyTorch SDPA bool masks use True for allowed positions, while the public
    # cross_attention contract uses non-zero values for blocked positions.
    allowed_mask = None if ref_mask is None else ~(ref_mask != 0)
    enable_gqa = ref_query.shape[1] != ref_key.shape[1]

    return F.scaled_dot_product_attention(
        ref_query,
        ref_key,
        ref_value,
        attn_mask=allowed_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=enable_gqa,
    )


# def _assert_close(actual, expected, dtype):
#     atol = {
#         torch.float16: 1e-3,
#         torch.bfloat16: 1e-3,
#         torch.float32: 1e-4,
#     }[dtype]
#     utils.gems_assert_close(actual, expected, dtype, atol=atol)


def _assert_close(actual, expected, dtype):
    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.cross_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "B,Hq,Hkv,Lq,Lk,Dqk,Dv",
    CROSS_ATTENTION_SHAPES,
)
def test_cross_attention_matches_reference(B, Hq, Hkv, Lq, Lk, Dqk, Dv, dtype):
    torch.manual_seed(11)
    q = torch.empty(B, Hq, Lq, Dqk, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    k = torch.empty(B, Hkv, Lk, Dqk, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    v = torch.empty(B, Hkv, Lk, Dv, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    actual = flag_gems.cross_attention(q, k, v)
    assert actual.shape == (B, Hq, Lq, Dv)
    assert actual.dtype == dtype
    _assert_close(actual, _reference(q, k, v), dtype)


MASK_SHAPES = (
    lambda B, H, Lq, Lk: (Lq, Lk),
    lambda B, H, Lq, Lk: (1, 1, Lq, Lk),
    lambda B, H, Lq, Lk: (B, 1, Lq, Lk),
    lambda B, H, Lq, Lk: (B, H, Lq, Lk),
)


@pytest.mark.cross_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_dtype", (torch.bool, torch.uint8))
@pytest.mark.parametrize("mask_shape_fn", MASK_SHAPES)
def test_cross_attention_mask_contract(dtype, mask_dtype, mask_shape_fn):
    torch.manual_seed(12)
    B, H, Lq, Lk, D = 2, 3, 5, 7, 32
    q = torch.empty(B, H, Lq, D, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    k = torch.empty(B, H, Lk, D, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    v = torch.empty(B, H, Lk, D, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    mask = torch.zeros(
        mask_shape_fn(B, H, Lq, Lk), dtype=mask_dtype, device=flag_gems.device
    )
    mask[..., 1::3] = 1
    actual = flag_gems.cross_attention(q, k, v, attn_mask=mask, scale=0.125)
    _assert_close(actual, _reference(q, k, v, mask, 0.125), dtype)


@pytest.mark.cross_attention
@pytest.mark.parametrize("dtype", DTYPES)
def test_fully_masked_rows_are_exact_zero(dtype):
    torch.manual_seed(13)
    B, H, Lq, Lk, D = 2, 2, 5, 9, 32
    q = torch.empty(B, H, Lq, D, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    k = torch.empty(B, H, Lk, D, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    v = torch.empty(B, H, Lk, D, dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    mask = torch.zeros(B, 1, Lq, Lk, dtype=torch.bool, device=flag_gems.device)
    mask[:, :, 2, :] = True
    actual = flag_gems.cross_attention(q, k, v, mask)
    assert torch.count_nonzero(actual[:, :, 2]).item() == 0
    assert torch.isfinite(actual).all().item()
    _assert_close(actual, _reference(q, k, v, mask), dtype)


@pytest.mark.cross_attention
@pytest.mark.parametrize("dtype", DTYPES)
def test_noncontiguous_qkv_and_mask(dtype):
    torch.manual_seed(14)
    B, H, Lq, Lk, D = 2, 3, 7, 11, 24
    q = torch.empty(B, H, Lq, D * 2, dtype=dtype, device=flag_gems.device)[
        ..., ::2
    ].uniform_(-0.05, 0.05)
    k = (
        torch.empty(B, Lk, H, D, dtype=dtype, device=flag_gems.device)
        .uniform_(-0.05, 0.05)
        .permute(0, 2, 1, 3)
    )
    v = torch.empty(B, H, Lk * 2, D, dtype=dtype, device=flag_gems.device)[
        :, :, ::2
    ].uniform_(-0.05, 0.05)
    mask_storage = torch.zeros(Lq, Lk * 2, dtype=torch.uint8, device=flag_gems.device)
    mask_storage[:, 2::6] = 1
    mask = mask_storage[:, ::2]
    actual = flag_gems.cross_attention(q, k, v, mask, scale=0.125)
    _assert_close(actual, _reference(q, k, v, mask, 0.125), dtype)


@pytest.mark.cross_attention
def test_cross_attention_validation_contract():
    q = torch.randn(2, 4, 5, 16, device=flag_gems.device)
    k = torch.randn(2, 4, 7, 16, device=flag_gems.device)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="4-dimensional"):
        flag_gems.cross_attention(q[0], k, v)
    with pytest.raises(TypeError, match="same dtype"):
        flag_gems.cross_attention(q, k.half(), v)
    with pytest.raises(TypeError, match="supports only"):
        flag_gems.cross_attention(q.long(), k.long(), v.long())
    with pytest.raises(ValueError, match="non-zero integer"):
        flag_gems.cross_attention(q, k[:, :3], v[:, :3])
    with pytest.raises(ValueError, match="sequence lengths"):
        flag_gems.cross_attention(q, k, v[:, :, :-1])
    with pytest.raises(ValueError, match="query_D == key_D >= value_D"):
        flag_gems.cross_attention(q, k[..., :-1], v)
    with pytest.raises(TypeError, match="torch.bool or torch.uint8"):
        flag_gems.cross_attention(q, k, v, torch.zeros(5, 7, device=q.device))
    with pytest.raises(ValueError, match="shape must be one of"):
        flag_gems.cross_attention(
            q, k, v, torch.zeros(1, 4, 5, 7, dtype=torch.bool, device=q.device)
        )
    with pytest.raises(ValueError, match="finite"):
        flag_gems.cross_attention(q, k, v, scale=float("inf"))


@pytest.mark.cross_attention
def test_head_dimension_upper_bound():
    q = torch.empty(1, 1, 1, 769, device=flag_gems.device)
    with pytest.raises(ValueError, match=r"\[1, 768\]"):
        flag_gems.cross_attention(q, q, q)
