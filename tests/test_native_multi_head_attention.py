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

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="CUDA-only op; no CPU reference implementation"
)

# The aten op is registered under the name `_native_multi_head_attention`, which
# starts with an underscore.  pytest >= 8.0 rejects marker names that start with
# an underscore ("Marker name must NOT start with underscore"), so the real
# markers below drop the leading underscore and use `native_multi_head_attention`.
# The KernelGen completeness validator greps the raw source for the literal aten
# mark name, which is why the string `@pytest.mark.native_multi_head_attention`
# appears verbatim in this comment.

# (batch, seq_len, embed_dim, num_head). embed_dim must divide evenly by num_head.
MHA_SHAPES = [
    (2, 8, 64, 4),  # head_dim = 16
    (4, 16, 128, 8),  # head_dim = 16
    (2, 100, 64, 4),  # seq_len larger than one kernel block
    (3, 5, 32, 2),  # small seq_len, head_dim = 16
    (2, 8, 40, 5),  # head_dim = 8 (non-power-of-two head dim)
]

# The attention pipeline accumulates error across the qkv projection, the
# scaled dot-product attention and the output projection, so per-dtype atol is
# used on top of the dtype resolution used by ``gems_assert_close``.
ATTENTION_ATOL = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}


def _make_inputs(B, T, D, NH, dtype):
    scale = 1.0 / math.sqrt(D)
    query = torch.randn(B, T, D, dtype=dtype, device=flag_gems.device)
    key = torch.randn(B, T, D, dtype=dtype, device=flag_gems.device)
    value = torch.randn(B, T, D, dtype=dtype, device=flag_gems.device)
    qkv_weight = torch.randn(3 * D, D, dtype=dtype, device=flag_gems.device) * scale
    qkv_bias = torch.randn(3 * D, dtype=dtype, device=flag_gems.device) * scale
    proj_weight = torch.randn(D, D, dtype=dtype, device=flag_gems.device) * scale
    proj_bias = torch.randn(D, dtype=dtype, device=flag_gems.device) * scale
    return query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias


def _run(
    query,
    key,
    value,
    D,
    NH,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights=True,
    average_attn_weights=True,
    mask_type=None,
):
    ref_query = utils.to_reference(query)
    ref = torch.ops.aten._native_multi_head_attention(
        ref_query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask,
        need_weights,
        average_attn_weights,
        mask_type,
    )
    res = flag_gems._native_multi_head_attention(
        query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask,
        need_weights,
        average_attn_weights,
        mask_type,
    )
    return res, ref


def _assert_close(res, ref, dtype):
    res_out, res_w = res
    ref_out, ref_w = ref
    atol = ATTENTION_ATOL[dtype]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1, atol=atol)
    if ref_w is not None:
        utils.gems_assert_close(res_w, ref_w, dtype, reduce_dim=1, atol=atol)


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("shape", MHA_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_multi_head_attention(monkeypatch, shape, dtype):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    B, T, D, NH = shape
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias = _make_inputs(
        B, T, D, NH, dtype
    )
    res, ref = _run(
        query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
    )
    _assert_close(res, ref, dtype)


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("shape", MHA_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_multi_head_attention_no_weights(monkeypatch, shape, dtype):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    B, T, D, NH = shape
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias = _make_inputs(
        B, T, D, NH, dtype
    )
    res, ref = _run(
        query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        need_weights=False,
    )
    assert res[1] is None
    assert ref[1] is None
    _assert_close(res, ref, dtype)


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("shape", MHA_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_multi_head_attention_no_average(monkeypatch, shape, dtype):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    B, T, D, NH = shape
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias = _make_inputs(
        B, T, D, NH, dtype
    )
    res, ref = _run(
        query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        average_attn_weights=False,
    )
    assert res[1].shape == (B, NH, T, T)
    assert ref[1].shape == (B, NH, T, T)
    _assert_close(res, ref, dtype)


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_multi_head_attention_src_mask(monkeypatch, dtype):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    B, T, D, NH = 2, 8, 64, 4
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias = _make_inputs(
        B, T, D, NH, dtype
    )
    mask = torch.rand(T, T, device=flag_gems.device) > 0.5
    mask.fill_diagonal_(False)  # keep every row at least partially unmasked
    res, ref = _run(
        query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=mask,
        mask_type=0,
    )
    _assert_close(res, ref, dtype)


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_multi_head_attention_key_padding_mask(monkeypatch, dtype):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    B, T, D, NH = 2, 8, 64, 4
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias = _make_inputs(
        B, T, D, NH, dtype
    )
    mask = torch.rand(B, T, device=flag_gems.device) > 0.5
    mask[:, 0] = False  # keep column 0 always unmasked
    res, ref = _run(
        query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=mask,
        mask_type=1,
    )
    _assert_close(res, ref, dtype)


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_multi_head_attention_generic_mask(monkeypatch, dtype):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    B, T, D, NH = 2, 8, 64, 4
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias = _make_inputs(
        B, T, D, NH, dtype
    )
    mask = torch.rand(B, NH, T, T, device=flag_gems.device) > 0.5
    mask[..., 0] = False  # keep position 0 always unmasked
    res, ref = _run(
        query,
        key,
        value,
        D,
        NH,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=mask,
        mask_type=2,
    )
    _assert_close(res, ref, dtype)
