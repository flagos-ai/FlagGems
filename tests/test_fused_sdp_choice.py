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

# SDPBackend integer values (torch.nn.attention.SDPBackend).
_SDP_ERROR = -1
_SDP_MATH = 0
_SDP_FLASH = 1
_SDP_EFFICIENT = 2
_SDP_CUDNN = 3

# (batch, num_heads, seq_len, head_dim) attention shapes covering typical
# inference/training scenarios and the fused-kernel boundary conditions.
ATTENTION_SHAPES = [
    (2, 8, 128, 64),
    (1, 16, 256, 64),
    (4, 8, 512, 128),
    (2, 8, 1024, 128),
    (1, 1, 1, 64),
    (2, 8, 128, 256),
    (2, 8, 128, 50),
    (2, 8, 1, 64),
]

# dtypes the SDP backends discriminate on.  float32 falls back to the
# memory-efficient kernel, float64 to the math kernel, while the low-precision
# fp16/bf16 prefer cuDNN/Flash on this hardware.
ATTENTION_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

# Backends that reject float64: only the low-precision fp16/bf16 paths exercise
# cuDNN/Flash here, so the 64-bit math-backend case is excluded from these cases.
LOW_PREC_DTYPES = [torch.float16, torch.bfloat16]

# float16/bf16/fp32 sweep for the general dispatch cases (float64 routes to the
# math backend and is covered separately by ATTENTION_DTYPES).
MID_PREC_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _make_qkv(shape, dtype, device):
    q = torch.randn(shape, dtype=dtype, device=device)
    return q, q, q


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test_fused_sdp_choice(shape, dtype):
    """Default backend selection across attention shapes and dtypes."""
    # ``_fused_sdp_choice`` returns the SDPBackend that PyTorch would dispatch to
    # for *the inputs' own device*.  With ``--ref=cpu`` the reference tensors are
    # moved to CPU (where aten can only ever select the math backend), while the
    # Gems kernel still runs on the GPU and selects a fused backend -- the two
    # integers are device-dependent and not comparable, so the cross-device
    # reference path is skipped.
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    q, k, v = _make_qkv(shape, dtype, flag_gems.device)
    ref_q = utils.to_reference(q)
    ref_k = utils.to_reference(k)
    ref_v = utils.to_reference(v)

    ref_out = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v)
    res_out = flag_gems._fused_sdp_choice(q, k, v)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", MID_PREC_DTYPES)
def test_fused_sdp_choice_is_causal(shape, dtype):
    """Selection under the ``is_causal`` flag."""
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    q, k, v = _make_qkv(shape, dtype, flag_gems.device)
    ref_q, ref_k, ref_v = (
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
    )

    ref_out = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v, is_causal=True)
    res_out = flag_gems._fused_sdp_choice(q, k, v, is_causal=True)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", MID_PREC_DTYPES)
def test_fused_sdp_choice_with_attn_mask(shape, dtype):
    """Selection when a boolean attention mask is supplied."""
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    batch, heads, seq_q, head_dim = shape
    seq_k = seq_q
    q = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    k = torch.randn(
        (batch, heads, seq_k, head_dim), dtype=dtype, device=flag_gems.device
    )
    v = k
    attn_mask = torch.zeros(
        (batch, heads, seq_q, seq_k), dtype=torch.bool, device=flag_gems.device
    )
    ref_q, ref_k, ref_v = (
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
    )
    ref_mask = utils.to_reference(attn_mask)

    ref_out = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v, attn_mask=ref_mask)
    res_out = flag_gems._fused_sdp_choice(q, k, v, attn_mask=attn_mask)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("dtype", MID_PREC_DTYPES)
def test_fused_sdp_choice_dropout(shape, dtype):
    """Selection under a non-zero dropout probability."""
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    q, k, v = _make_qkv(shape, dtype, flag_gems.device)
    ref_q, ref_k, ref_v = (
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
    )

    ref_out = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v, dropout_p=0.1)
    res_out = flag_gems._fused_sdp_choice(q, k, v, dropout_p=0.1)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("dtype", MID_PREC_DTYPES)
def test_fused_sdp_choice_gqa(dtype):
    """Grouped-query attention with ``enable_gqa`` toggled on and off."""
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    shape_q = (2, 8, 128, 64)
    shape_kv = (2, 2, 128, 64)
    q = torch.randn(shape_q, dtype=dtype, device=flag_gems.device)
    k = torch.randn(shape_kv, dtype=dtype, device=flag_gems.device)
    v = k
    ref_q, ref_k, ref_v = (
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
    )

    # Without enable_gqa the mismatched num_heads block the fused kernels.
    ref_off = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v)
    res_off = flag_gems._fused_sdp_choice(q, k, v)
    utils.gems_assert_equal(res_off, ref_off)

    ref_on = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v, enable_gqa=True)
    res_on = flag_gems._fused_sdp_choice(q, k, v, enable_gqa=True)
    utils.gems_assert_equal(res_on, ref_on)


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("dtype", MID_PREC_DTYPES)
def test_fused_sdp_choice_non_contiguous(dtype):
    """A non-stride-1 last dimension must drive selection to the math backend."""
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    # Slice every other element along the last dim so stride(-1) == 2.
    full = torch.randn((2, 8, 128, 128), dtype=dtype, device=flag_gems.device)
    q = full[..., ::2]
    k = q
    v = q
    ref_q, ref_k, ref_v = (
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
    )

    ref_out = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v)
    res_out = flag_gems._fused_sdp_choice(q, k, v)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("dtype", LOW_PREC_DTYPES)
def test_fused_sdp_choice_singleton_head_dim(dtype):
    """A singleton head dim is tolerated by cuDNN/Flash even when stride != 1."""
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    # head_dim == 1 reached by slicing; the permissive ignore_singleton branch
    # should let cuDNN/Flash through (the efficient kernel still rejects it).
    full = torch.randn((2, 8, 128, 8), dtype=dtype, device=flag_gems.device)
    q = full[..., ::8]
    k = q
    v = q
    ref_q, ref_k, ref_v = (
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
    )

    ref_out = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v)
    res_out = flag_gems._fused_sdp_choice(q, k, v)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fused_sdp_choice
@pytest.mark.parametrize("dtype", MID_PREC_DTYPES)
def test_fused_sdp_choice_scale(dtype):
    """The ``scale`` keyword must not affect the backend choice."""
    if utils.TO_CPU:
        pytest.skip("_fused_sdp_choice is device-dependent; CPU ref is always math")
    q, k, v = _make_qkv((2, 8, 128, 64), dtype, flag_gems.device)
    ref_q, ref_k, ref_v = (
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
    )

    ref_out = torch.ops.aten._fused_sdp_choice(ref_q, ref_k, ref_v, scale=0.1)
    res_out = flag_gems._fused_sdp_choice(q, k, v, scale=0.1)
    utils.gems_assert_equal(res_out, ref_out)
