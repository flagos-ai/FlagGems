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
import triton

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils
from . import conftest as cfg

DTYPES = [torch.float16, torch.bfloat16]
DENSE_CASES = [
    (1, 2, 2, 16, 16, 32),
    (2, 4, 4, 17, 31, 64),
    (1, 4, 2, 33, 65, 128),
]


def _set_seed():
    torch.manual_seed(0)
    set_philox_state(1234567890, 0, torch_device_fn.current_device())


def _make_qkv(batch, q_heads, kv_heads, q_len, kv_len, head_dim, dtype):
    _set_seed()
    query = torch.empty(
        (batch, q_len, q_heads, head_dim), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    key = torch.empty(
        (batch, kv_len, kv_heads, head_dim), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    return query, key, value


def _cpu_flash_forward(
    query,
    key,
    value,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
):
    """CPU suite oracle with an actual CPU RNG state, without CUDA Philox replay."""
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("FlashAttention only supports float16 and bfloat16 inputs")
    if (cum_seq_q is None) != (cum_seq_k is None):
        raise RuntimeError("Both cumulative sequence tensors must be provided together")
    varlen = cum_seq_q is not None
    rank = 3 if varlen else 4
    if query.ndim != rank or key.ndim != rank or value.ndim != rank:
        raise RuntimeError("FlashAttention input rank does not match sequence metadata")
    if (
        query.dtype != key.dtype
        or key.dtype != value.dtype
        or query.device != key.device
        or key.device != value.device
        or key.shape != value.shape
        or query.shape[-1] != key.shape[-1]
        or query.shape[-2] % key.shape[-2] != 0
    ):
        raise RuntimeError("FlashAttention query/key/value metadata mismatch")
    if not 0 <= dropout_p < 1:
        raise RuntimeError("dropout probability must be in [0, 1)")
    if return_debug_mask and dropout_p == 0:
        raise RuntimeError("return_softmax is only supported when p_dropout > 0.0")
    q, k, v = (x.detach().cpu().double() for x in (query, key, value))
    if varlen:
        cq, ck = cum_seq_q.cpu().long(), cum_seq_k.cpu().long()
        if (
            cq.ndim != 1
            or ck.ndim != 1
            or cq.numel() != ck.numel()
            or cq.numel() < 2
            or cq[0] != 0
            or ck[0] != 0
            or cq[-1] != q.shape[0]
            or ck[-1] != k.shape[0]
            or bool((cq.diff() < 0).any())
            or bool((ck.diff() < 0).any())
            or bool((cq.diff() > max_q).any())
            or bool((ck.diff() > max_k).any())
        ):
            raise RuntimeError("Invalid cumulative sequence metadata")
        batch = cq.numel() - 1
        qparts = [q[int(cq[i]) : int(cq[i + 1])] for i in range(batch)]
        kparts = [k[int(ck[i]) : int(ck[i + 1])] for i in range(batch)]
        vparts = [v[int(ck[i]) : int(ck[i + 1])] for i in range(batch)]
    else:
        if q.shape[0] != k.shape[0]:
            raise RuntimeError("FlashAttention batch sizes differ")
        batch = q.shape[0]
        qparts, kparts, vparts = list(q), list(k), list(v)
    used = seqused_k.detach().cpu().long() if varlen and seqused_k is not None else None
    if used is not None and (used.shape != (batch,) or bool((used < 0).any())):
        raise RuntimeError("Invalid used key lengths")
    slopes = alibi_slopes.detach().cpu().double() if alibi_slopes is not None else None
    if slopes is not None and slopes.shape not in [
        (query.shape[-2],),
        (batch, query.shape[-2]),
    ]:
        raise RuntimeError("Invalid ALiBi slopes shape")
    # PPU ATen exposes signed int64 seed/offset tensors. Preserve the real CPU
    # RNG bytes while representing them with the target native metadata ABI.
    rng_dtype = torch.int64 if flag_gems.vendor_name == "thead" else torch.uint64
    rng_state = torch.random.get_rng_state()[:16].clone().view(rng_dtype)
    debug = (
        torch.zeros(
            (
                batch,
                query.shape[-2],
                math.ceil(max_q / 128) * 128,
                math.ceil(max_k / 128) * 128,
            ),
            dtype=query.dtype,
        )
        if return_debug_mask
        else torch.empty((0,), dtype=query.dtype)
    )
    outputs, lses = [], []
    for i, (qi, ki, vi) in enumerate(zip(qparts, kparts, vparts)):
        if used is not None:
            ki, vi = ki[: int(used[i])], vi[: int(used[i])]
        qi = qi.transpose(0, 1)
        ki = ki.transpose(0, 1).repeat_interleave(qi.shape[0] // ki.shape[1], dim=0)
        vi = vi.transpose(0, 1).repeat_interleave(qi.shape[0] // vi.shape[1], dim=0)
        scores = qi @ ki.transpose(-2, -1)
        scores *= 1 / math.sqrt(qi.shape[-1]) if scale is None else scale
        qpos = torch.arange(qi.shape[-2])[:, None] + ki.shape[-2] - qi.shape[-2]
        kpos = torch.arange(ki.shape[-2])[None, :]
        if slopes is not None:
            slope = slopes if slopes.ndim == 1 else slopes[i]
            scores -= slope[:, None, None] * (qpos - kpos).abs()
        left = -1 if window_size_left is None else window_size_left
        right = -1 if window_size_right is None else window_size_right
        if is_causal:
            right = 0 if right < 0 else min(right, 0)
        mask = torch.zeros((qi.shape[-2], ki.shape[-2]), dtype=torch.bool)
        if left >= 0:
            mask |= kpos < qpos - left
        if right >= 0:
            mask |= kpos > qpos + right
        scores.masked_fill_(mask, float("-inf"))
        lses.append(torch.logsumexp(scores, -1).float())
        probabilities = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)
        if dropout_p:
            keep = torch.rand(probabilities.shape) >= dropout_p
            if return_debug_mask:
                debug[i, :, : qi.shape[-2], : ki.shape[-2]] = torch.where(
                    keep, probabilities, -probabilities
                )
            probabilities = probabilities * keep / (1 - dropout_p)
        outputs.append((probabilities @ vi).transpose(0, 1).to(query.dtype))
    output = torch.cat(outputs, dim=0) if varlen else torch.stack(outputs)
    lse = torch.cat(lses, dim=1) if varlen else torch.stack(lses)
    return output, lse, rng_state, torch.zeros((), dtype=rng_dtype), debug


def _call(
    query,
    key,
    value,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    **kwargs,
):
    if cfg.TO_CPU or flag_gems.vendor_name in {
        "ascend",
        "kunlunxin",
        "metax",
        "mthreads",
    }:
        return _cpu_flash_forward(
            query,
            key,
            value,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            return_debug_mask,
            **kwargs,
        )
    try:
        return torch.ops.aten._flash_attention_forward.default(
            query,
            key,
            value,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            return_debug_mask,
            **kwargs,
        )
    except RuntimeError as error:
        if (
            kwargs.get("alibi_slopes") is not None
            and str(error) == "This flash attention build does not support alibi."
        ):
            pytest.skip(str(error))
        raise


def _reference_args(query, key, value, cum_seq_q=None, cum_seq_k=None):
    return (
        utils.to_reference(query, False),
        utils.to_reference(key, False),
        utils.to_reference(value, False),
        utils.to_reference(cum_seq_q, False) if cum_seq_q is not None else None,
        utils.to_reference(cum_seq_k, False) if cum_seq_k is not None else None,
    )


def _assert_all_outputs(actual, expected, dtype, query):
    assert len(actual) == len(expected) == 5
    assert actual[0].device == query.device and actual[0].dtype == query.dtype
    assert actual[0].shape == expected[0].shape
    assert actual[1].device == query.device and actual[1].dtype == torch.float32
    assert actual[1].shape == expected[1].shape
    for index in (2, 3, 4):
        expected_device = (
            query.device
            if cfg.TO_CPU
            or flag_gems.vendor_name in {"ascend", "kunlunxin", "metax", "mthreads"}
            else expected[index].device
        )
        assert actual[index].device == expected_device
        assert actual[index].shape == expected[index].shape
        assert actual[index].dtype == expected[index].dtype
    utils.gems_assert_close(actual[0].cpu(), expected[0].cpu(), dtype)
    utils.gems_assert_close(actual[1].cpu(), expected[1].cpu(), torch.float32)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", DENSE_CASES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale_mode", ["default", "explicit"])
def test__flash_attention_forward(dtype, shape, is_causal, scale_mode):
    batch, q_heads, kv_heads, q_len, kv_len, head_dim = shape
    query, key, value = _make_qkv(
        batch, q_heads, kv_heads, q_len, kv_len, head_dim, dtype
    )
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    scale = None if scale_mode == "default" else 0.5 / math.sqrt(head_dim)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        q_len,
        kv_len,
        0.0,
        is_causal,
        False,
        scale=scale,
    )
    actual = flag_gems._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        q_len,
        kv_len,
        0.0,
        is_causal,
        False,
        scale=scale,
    )

    _assert_all_outputs(actual, expected, dtype, query)
    assert actual[0].shape == query.shape
    assert actual[0].untyped_storage().data_ptr() != query.untyped_storage().data_ptr()


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.skipif(
    flag_gems.vendor_name
    in {"hygon", "metax", "mthreads", "kunlunxin", "cambricon", "tsingmicro"},
    reason="Existing FlashAttention sliding-window backend limitations: issues #2810, #2811, #2812, #2814, #5254, #4083",
)
@pytest.mark.parametrize(
    "window_size_left,window_size_right", [(8, 0), (8, 8), (None, 4)]
)
def test__flash_attention_forward_windowed(window_size_left, window_size_right):
    dtype = torch.float16
    query, key, value = _make_qkv(1, 2, 2, 17, 31, 64, dtype)
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    kwargs = {
        "scale": None,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
    }
    expected = _call(
        ref_query, ref_key, ref_value, None, None, 17, 31, 0.0, False, False, **kwargs
    )
    actual = flag_gems._flash_attention_forward(
        query, key, value, None, None, 17, 31, 0.0, False, False, **kwargs
    )

    _assert_all_outputs(actual, expected, dtype, query)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.skipif(
    flag_gems.vendor_name in {"hygon", "mthreads", "kunlunxin", "tsingmicro"},
    reason="Existing FlashAttention GQA/ALiBi backend limitations: issues #2810, #2812, #2814, #4083",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test__flash_attention_forward_gqa_alibi(is_causal):
    dtype = torch.float16
    query, key, value = _make_qkv(2, 4, 2, 16, 31, 64, dtype)
    alibi_slopes = torch.linspace(
        0.1, 0.4, steps=8, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 4)
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    ref_alibi = utils.to_reference(alibi_slopes, False)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        16,
        31,
        0.0,
        is_causal,
        False,
        scale=None,
        alibi_slopes=ref_alibi,
    )
    actual = flag_gems._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        16,
        31,
        0.0,
        is_causal,
        False,
        scale=None,
        alibi_slopes=alibi_slopes,
    )

    _assert_all_outputs(actual, expected, dtype, query)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
def test__flash_attention_forward_noncontiguous_bshd(dtype):
    _set_seed()
    query = (
        torch.empty((2, 4, 17, 64), dtype=dtype, device=flag_gems.device)
        .uniform_(-0.05, 0.05)
        .transpose(1, 2)
    )
    key = (
        torch.empty((2, 4, 31, 64), dtype=dtype, device=flag_gems.device)
        .uniform_(-0.05, 0.05)
        .transpose(1, 2)
    )
    value = (
        torch.empty((2, 4, 31, 64), dtype=dtype, device=flag_gems.device)
        .uniform_(-0.05, 0.05)
        .transpose(1, 2)
    )
    assert not query.is_contiguous()
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        17,
        31,
        0.0,
        False,
        False,
        scale=None,
    )
    actual = flag_gems._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        17,
        31,
        0.0,
        False,
        False,
        scale=None,
    )

    _assert_all_outputs(actual, expected, dtype, query)


@pytest.mark.underscore_flash_attention_forward
def test__flash_attention_forward_seqused_k_metadata():
    dtype = torch.float16
    query, key, value = _make_qkv(2, 2, 2, 4, 8, 32, dtype)
    seqused_k = torch.tensor([5, 8], dtype=torch.int32, device=flag_gems.device)
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    ref_seqused_k = utils.to_reference(seqused_k, False)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        4,
        8,
        0.0,
        False,
        False,
        scale=None,
        seqused_k=ref_seqused_k,
    )
    actual = flag_gems._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        4,
        8,
        0.0,
        False,
        False,
        scale=None,
        seqused_k=seqused_k,
    )

    _assert_all_outputs(actual, expected, dtype, query)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
def test__flash_attention_forward_varlen(dtype):
    _set_seed()
    cum_seq_q = torch.tensor([0, 3, 8], dtype=torch.int32, device=flag_gems.device)
    cum_seq_k = torch.tensor([0, 5, 9], dtype=torch.int32, device=flag_gems.device)
    query = torch.empty((8, 2, 32), dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    key = torch.empty((9, 2, 32), dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    ref_query, ref_key, ref_value, ref_cum_q, ref_cum_k = _reference_args(
        query, key, value, cum_seq_q, cum_seq_k
    )

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        ref_cum_q,
        ref_cum_k,
        5,
        5,
        0.0,
        False,
        False,
        scale=None,
    )
    actual = flag_gems._flash_attention_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        5,
        5,
        0.0,
        False,
        False,
        scale=None,
    )

    _assert_all_outputs(actual, expected, dtype, query)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.skipif(triton.__version__ < "3.1", reason="Dropout requires Triton >= 3.1")
@pytest.mark.skipif(
    flag_gems.vendor_name in {"hygon", "mthreads", "kunlunxin"},
    reason="Existing FlashAttention dropout backend limitations: issues #2810, #2812, #2814",
)
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
def test__flash_attention_forward_dropout_debug_mask(dropout_p):
    dtype = torch.float16
    query, key, value = _make_qkv(1, 2, 2, 128, 128, 32, dtype)

    _set_seed()
    actual = flag_gems._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        128,
        128,
        dropout_p,
        False,
        True,
        scale=None,
    )
    _set_seed()
    replay = flag_gems._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        128,
        128,
        dropout_p,
        False,
        True,
        scale=None,
    )

    _assert_all_outputs(actual, replay, dtype, query)
    assert len(actual) == len(replay) == 5
    utils.gems_assert_equal(actual[0].cpu(), replay[0].cpu())
    utils.gems_assert_equal(actual[1].cpu(), replay[1].cpu())
    # Packed native RNG state contains both seed and offset; output 3 then
    # has an unspecified payload. Keep its ABI check above and preserve full
    # value replay checks for the legacy separate scalar seed/offset ABI.
    packed_rng_state = (
        flag_gems.vendor_name in {"nvidia", "thead"}
        and actual[2].shape == replay[2].shape == (2,)
        and actual[2].dtype in (torch.int64, torch.uint64)
        and actual[3].shape == replay[3].shape == ()
        and actual[3].dtype == actual[2].dtype
        and actual[2].device == actual[3].device == query.device
    )
    for index in ((2, 4) if packed_rng_state else (2, 3, 4)):
        result, repeated = actual[index], replay[index]
        assert result.shape == repeated.shape
        assert result.dtype == repeated.dtype
        if result.numel():
            utils.gems_assert_equal(result.cpu(), repeated.cpu())
    debug_mask = actual[4].cpu()
    kept_or_dropped = torch.sum(debug_mask != 0)
    assert kept_or_dropped.item() > 0
    observed_dropout = (
        torch.sum(debug_mask < 0).float() / kept_or_dropped.float()
    ).item()
    assert abs(observed_dropout - dropout_p) < 0.03


@pytest.mark.underscore_flash_attention_forward
def test__flash_attention_forward_rejects_partial_varlen_metadata():
    query, key, value = _make_qkv(1, 2, 2, 16, 16, 32, torch.float16)
    cum_seq_q = torch.tensor([0, 16], dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems._flash_attention_forward(
            query,
            key,
            value,
            cum_seq_q,
            None,
            16,
            16,
            0.0,
            False,
            False,
            scale=None,
        )


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__flash_attention_forward_rejects_unsupported_dtype(dtype):
    query = torch.ones((1, 16, 2, 32), dtype=dtype, device=flag_gems.device)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems._flash_attention_forward(
            query,
            key,
            value,
            None,
            None,
            16,
            16,
            0.0,
            False,
            False,
            scale=None,
        )
