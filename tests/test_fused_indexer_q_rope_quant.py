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

HEAD_DIM = 128
ROPE_DIM = 64
NUM_HEADS = 8
MAX_POS = 4096
MXFP4_BLOCK_SIZE = 32

HAS_NATIVE_FP8 = hasattr(torch, "float8_e4m3fn") and (
    flag_gems.SUPPORTED_FP8_DTYPE == torch.float8_e4m3fn
)


def _supports_mxfp4_ptx():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _make_cos_sin_cache(max_pos, rope_dim, device, dtype):
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def _rotate_tail_gptj(q, positions, cos_sin_cache):
    nope_dim = HEAD_DIM - ROPE_DIM
    half = ROPE_DIM // 2
    cos_sin = cos_sin_cache.index_select(0, positions).float()
    cos = cos_sin[:, :half].unsqueeze(1)
    sin = cos_sin[:, half:].unsqueeze(1)

    q_rot = q.float()
    tail = q_rot[..., nope_dim:]
    even = tail[..., 0::2]
    odd = tail[..., 1::2]
    rot_even = (even * cos - odd * sin).to(torch.bfloat16).float()
    rot_odd = (odd * cos + even * sin).to(torch.bfloat16).float()
    q_rot[..., nope_dim::2] = rot_even
    q_rot[..., nope_dim + 1 :: 2] = rot_odd
    return q_rot


def _reference_fp8(q_rot, weights, softmax_scale, head_scale):
    amax = q_rot.abs().amax(dim=-1)
    q_scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-4) / 448.0)))
    q_fp8 = (q_rot / q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights_out = weights.float() * q_scale * softmax_scale * head_scale
    return q_fp8, weights_out


def _quantize_to_mxfp4(x):
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    n_blocks = head_dim // MXFP4_BLOCK_SIZE
    x_f32 = x.float().reshape(-1, n_blocks, MXFP4_BLOCK_SIZE)

    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=6 * (2**-126))
    log2_ratio = (amax / 6.0).log2().ceil().clamp(-127.0, 127.0)
    scale = log2_ratio.exp2()
    ue8m0 = (log2_ratio + 127.0).to(torch.uint8)

    x_scaled = (x_f32 / scale).clamp(-6.0, 6.0)
    abs_x = x_scaled.abs()
    code = torch.zeros_like(abs_x, dtype=torch.int32)
    code = torch.where(abs_x > 0.25, 1, code)
    code = torch.where(abs_x >= 0.75, 2, code)
    code = torch.where(abs_x > 1.25, 3, code)
    code = torch.where(abs_x >= 1.75, 4, code)
    code = torch.where(abs_x > 2.5, 5, code)
    code = torch.where(abs_x >= 3.5, 6, code)
    code = torch.where(abs_x > 5.0, 7, code)
    sign = ((x_scaled.view(torch.int32) >> 31) & 1).to(torch.uint8)
    nibble = code.to(torch.uint8) | (sign << 3)

    nibble_flat = nibble.reshape(-1, head_dim)
    packed = (nibble_flat[:, 0::2] | (nibble_flat[:, 1::2] << 4)).contiguous()
    packed = packed.reshape(*orig_shape[:-1], head_dim // 2)
    scales = ue8m0.view(*orig_shape[:-1], n_blocks)
    return packed, scales


def _reference_mxfp4(q_rot, weights, softmax_scale, head_scale):
    q_packed, q_scale = _quantize_to_mxfp4(q_rot)
    weights_out = weights.float() * softmax_scale * head_scale
    return (q_packed, q_scale.view(torch.int32).squeeze(-1)), weights_out


@pytest.mark.fused_indexer_q_rope_quant
@pytest.mark.skipif(flag_gems.device != "cuda", reason="requires CUDA")
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_fp4", [False, True])
@torch.inference_mode()
def test_fused_indexer_q_rope_quant(num_tokens, cache_dtype, use_fp4):
    if use_fp4 and not _supports_mxfp4_ptx():
        pytest.skip("MXFP4 E2M1 PTX conversion requires sm100 or newer")

    device = flag_gems.device
    torch.manual_seed(0)

    q = torch.randn(
        num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = _make_cos_sin_cache(MAX_POS, ROPE_DIM, device, cache_dtype)
    weights = torch.randn(num_tokens, NUM_HEADS, dtype=torch.bfloat16, device=device)
    softmax_scale = HEAD_DIM**-0.5
    head_scale = NUM_HEADS**-0.5

    q_rot = _rotate_tail_gptj(q, positions, cos_sin_cache)
    if use_fp4:
        q_ref, weights_ref = _reference_mxfp4(q_rot, weights, softmax_scale, head_scale)
    else:
        q_ref, weights_ref = _reference_fp8(q_rot, weights, softmax_scale, head_scale)

    q_out, weights_out = flag_gems.fused_indexer_q_rope_quant(
        positions,
        q,
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        use_fp4=use_fp4,
    )

    if use_fp4:
        q_ref_values, q_ref_scales = q_ref
        q_out_values, q_out_scales = q_out
        assert torch.equal(q_ref_scales, q_out_scales)
        assert torch.equal(q_ref_values, q_out_values)
    else:
        assert torch.equal(q_ref.view(torch.int8), q_out.view(torch.int8))

    assert weights_out.dtype == torch.float32
    torch.testing.assert_close(weights_out, weights_ref, rtol=0, atol=0)
