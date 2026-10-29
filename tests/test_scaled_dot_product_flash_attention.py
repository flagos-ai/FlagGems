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

from . import accuracy_utils as utils
from . import conftest as cfg

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [
    (1, 2, 16, 16, 32),
    (2, 4, 17, 31, 64),
    (1, 1, 1, 65, 128),
]


def _make_qkv(batch, heads, q_len, kv_len, head_dim, dtype, *, noncontiguous=False):
    torch.manual_seed(0)
    if noncontiguous:
        query = torch.randn(
            (batch, q_len, heads, head_dim), dtype=dtype, device=flag_gems.device
        ).transpose(1, 2)
        key = torch.randn(
            (batch, kv_len, heads, head_dim), dtype=dtype, device=flag_gems.device
        ).transpose(1, 2)
        value = torch.randn(
            (batch, kv_len, heads, head_dim), dtype=dtype, device=flag_gems.device
        ).transpose(1, 2)
    else:
        query = torch.randn(
            (batch, heads, q_len, head_dim), dtype=dtype, device=flag_gems.device
        )
        key = torch.randn(
            (batch, heads, kv_len, head_dim), dtype=dtype, device=flag_gems.device
        )
        value = torch.randn(
            (batch, heads, kv_len, head_dim), dtype=dtype, device=flag_gems.device
        )
    return query, key, value


def _cpu_flash_reference(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("FlashAttention only supports float16 and bfloat16 inputs")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise RuntimeError("FlashAttention expects rank-4 query, key and value")
    if (
        query.shape[:2] != key.shape[:2]
        or key.shape != value.shape
        or query.shape[-1] != key.shape[-1]
        or query.dtype != key.dtype
        or key.dtype != value.dtype
        or query.device != key.device
        or key.device != value.device
    ):
        raise RuntimeError("FlashAttention query, key and value metadata mismatch")
    if not 0 <= dropout_p < 1:
        raise RuntimeError("dropout probability must be in [0, 1)")
    if return_debug_mask and dropout_p == 0:
        raise RuntimeError("return_softmax is only supported when p_dropout > 0.0")
    q, k, v = (x.detach().cpu().double() for x in (query, key, value))
    q_len, kv_len = q.shape[-2], k.shape[-2]
    scores = q @ k.transpose(-2, -1)
    scores *= 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    if is_causal:
        q_index = torch.arange(q_len)[:, None] + kv_len - q_len
        k_index = torch.arange(kv_len)[None, :]
        scores.masked_fill_(k_index > q_index, float("-inf"))
    logsumexp = torch.logsumexp(scores, dim=-1).float()
    probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    debug_mask = torch.empty((0,), dtype=query.dtype)
    # This oracle uses the actual CPU generator. Its state prefix is opaque
    # CPU metadata and does not claim compatibility with CUDA Philox replay.
    rng_state = torch.random.get_rng_state()[:16].clone().view(torch.uint64)
    if dropout_p:
        keep = torch.rand(probabilities.shape) >= dropout_p
        if return_debug_mask:
            debug_mask = torch.zeros(
                (
                    *query.shape[:2],
                    math.ceil(q_len / 128) * 128,
                    math.ceil(kv_len / 128) * 128,
                ),
                dtype=query.dtype,
            )
            debug_mask[..., :q_len, :kv_len] = torch.where(
                keep, probabilities, -probabilities
            )
        probabilities = probabilities * keep / (1.0 - dropout_p)
    output = (probabilities @ v).to(query.dtype)
    if flag_gems.vendor_name == "hygon":
        # Match separate signed scalar fields using real CPU RNG state bytes.
        cpu_rng_words = rng_state.view(torch.int64)
        return (
            output,
            logsumexp,
            None,
            None,
            q_len,
            kv_len,
            cpu_rng_words[0],
            cpu_rng_words[1],
            debug_mask,
        )
    return (
        output,
        logsumexp,
        None,
        None,
        q_len,
        kv_len,
        rng_state,
        torch.zeros((), dtype=torch.uint64),
        debug_mask,
    )


def _call(query, key, value, dropout_p, is_causal, return_debug_mask, scale):
    try:
        if cfg.TO_CPU or flag_gems.vendor_name in {
            "ascend",
            "mthreads",
            "kunlunxin",
            "iluvatar",
        }:
            return _cpu_flash_reference(
                query, key, value, dropout_p, is_causal, return_debug_mask, scale=scale
            )
        return torch.ops.aten._scaled_dot_product_flash_attention.default(
            query, key, value, dropout_p, is_causal, return_debug_mask, scale=scale
        )
    except RuntimeError as error:
        if (
            return_debug_mask
            and dropout_p == 0
            and str(error) == "return_softmax is only supported when p_dropout > 0.0"
        ):
            pytest.skip(str(error))
        raise


def _assert_metadata(actual, expected, query):
    assert len(actual) == len(expected) == 9
    assert actual[0].device == query.device
    assert actual[0].dtype == query.dtype
    assert actual[0].shape == expected[0].shape
    assert actual[1].device == query.device
    assert actual[1].dtype == torch.float32
    assert actual[1].shape == expected[1].shape
    assert len(actual) == len(expected) == 9
    assert actual[2] is None and expected[2] is None
    assert actual[3] is None and expected[3] is None
    assert actual[4:6] == expected[4:6]
    for index in (6, 7, 8):
        expected_device = (
            query.device
            if cfg.TO_CPU
            or flag_gems.vendor_name in {"ascend", "mthreads", "kunlunxin", "iluvatar"}
            else expected[index].device
        )
        assert actual[index].device == expected_device
        assert actual[index].shape == expected[index].shape
        assert actual[index].dtype == expected[index].dtype


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.underscore_scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale_mode", ["default", "explicit"])
def test_scaled_dot_product_flash_attention(dtype, shape, is_causal, scale_mode):
    batch, heads, q_len, kv_len, head_dim = shape
    query, key, value = _make_qkv(batch, heads, q_len, kv_len, head_dim, dtype)
    scale = None if scale_mode == "default" else 0.5 / math.sqrt(head_dim)
    ref_query = utils.to_reference(query, False)
    ref_key = utils.to_reference(key, False)
    ref_value = utils.to_reference(value, False)

    expected = _call(ref_query, ref_key, ref_value, 0.0, is_causal, False, scale)
    actual = flag_gems._scaled_dot_product_flash_attention(
        query, key, value, 0.0, is_causal, False, scale=scale
    )

    _assert_metadata(actual, expected, query)
    utils.gems_assert_close(actual[0].cpu(), expected[0].cpu(), dtype)
    utils.gems_assert_close(actual[1].cpu(), expected[1].cpu(), torch.float32)
    assert actual[0].shape == query.shape
    assert actual[0].untyped_storage().data_ptr() != query.untyped_storage().data_ptr()


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.underscore_scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_dot_product_flash_attention_noncontiguous(dtype):
    query, key, value = _make_qkv(2, 4, 17, 31, 64, dtype, noncontiguous=True)
    assert not query.is_contiguous()
    ref_query = utils.to_reference(query, False)
    ref_key = utils.to_reference(key, False)
    ref_value = utils.to_reference(value, False)

    expected = _call(ref_query, ref_key, ref_value, 0.0, False, False, None)
    actual = flag_gems._scaled_dot_product_flash_attention(
        query, key, value, 0.0, False, False, scale=None
    )

    _assert_metadata(actual, expected, query)
    utils.gems_assert_close(actual[0].cpu(), expected[0].cpu(), dtype)
    utils.gems_assert_close(actual[1].cpu(), expected[1].cpu(), torch.float32)


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.underscore_scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_dot_product_flash_attention_debug_mask_metadata(dtype):
    query, key, value = _make_qkv(1, 2, 16, 16, 32, dtype)
    ref_query = utils.to_reference(query, False)
    ref_key = utils.to_reference(key, False)
    ref_value = utils.to_reference(value, False)

    expected = _call(ref_query, ref_key, ref_value, 0.0, False, True, None)
    actual = flag_gems._scaled_dot_product_flash_attention(
        query, key, value, 0.0, False, True, scale=None
    )

    _assert_metadata(actual, expected, query)
    utils.gems_assert_close(actual[0].cpu(), expected[0].cpu(), dtype)
    utils.gems_assert_close(actual[1].cpu(), expected[1].cpu(), torch.float32)
    if actual[8].numel() and expected[8].numel():
        utils.gems_assert_equal(actual[8].cpu(), expected[8].cpu())


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.underscore_scaled_dot_product_flash_attention
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
def test_scaled_dot_product_flash_attention_dropout_contract(dropout_p):
    query, key, value = _make_qkv(1, 2, 16, 16, 32, torch.float16)
    result = flag_gems._scaled_dot_product_flash_attention(
        query, key, value, dropout_p, False, True, scale=None
    )
    assert result[0].device == query.device and result[0].dtype == query.dtype
    assert result[1].device == query.device and result[1].dtype == torch.float32

    assert len(result) == 9
    assert result[0].shape == query.shape
    assert result[1].shape == query.shape[:-1]
    assert result[6].numel() > 0
    assert result[8].dtype in (torch.bool, torch.uint8, query.dtype)


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.underscore_scaled_dot_product_flash_attention
@pytest.mark.parametrize(
    "bad_query,bad_key,bad_value",
    [
        ((2, 16, 32), (1, 2, 16, 32), (1, 2, 16, 32)),
        ((1, 2, 16, 32), (1, 3, 16, 32), (1, 2, 16, 32)),
        ((1, 2, 16, 32), (1, 2, 16, 32), (1, 2, 17, 32)),
    ],
)
def test_scaled_dot_product_flash_attention_rejects_invalid_shapes(
    bad_query, bad_key, bad_value
):
    torch.manual_seed(0)
    query = torch.randn(bad_query, dtype=torch.float16, device=flag_gems.device)
    key = torch.randn(bad_key, dtype=torch.float16, device=flag_gems.device)
    value = torch.randn(bad_value, dtype=torch.float16, device=flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems._scaled_dot_product_flash_attention(
            query, key, value, 0.0, False, False, scale=None
        )


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.underscore_scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scaled_dot_product_flash_attention_rejects_unsupported_dtype(dtype):
    query = torch.ones((1, 2, 16, 32), dtype=dtype, device=flag_gems.device)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems._scaled_dot_product_flash_attention(
            query, key, value, 0.0, False, False, scale=None
        )
