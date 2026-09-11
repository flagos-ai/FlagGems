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

from . import base

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
    q_rot[..., nope_dim::2] = (even * cos - odd * sin).to(torch.bfloat16).float()
    q_rot[..., nope_dim + 1 :: 2] = (odd * cos + even * sin).to(torch.bfloat16).float()
    return q_rot


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


def _reference_op(
    positions,
    q,
    cos_sin_cache,
    weights,
    softmax_scale,
    head_scale,
    use_fp4,
):
    q_rot = _rotate_tail_gptj(q, positions, cos_sin_cache)
    if use_fp4:
        q_packed, q_scale = _quantize_to_mxfp4(q_rot)
        weights_out = weights.float() * softmax_scale * head_scale
        return (q_packed, q_scale.view(torch.int32).squeeze(-1)), weights_out

    amax = q_rot.abs().amax(dim=-1)
    q_scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-4) / 448.0)))
    q_fp8 = (q_rot / q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights_out = weights.float() * q_scale * softmax_scale * head_scale
    return q_fp8, weights_out


def _input_fn(shape, dtype, device):
    num_tokens, num_heads, use_fp4 = shape
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = _make_cos_sin_cache(MAX_POS, ROPE_DIM, device, torch.bfloat16)
    weights = torch.randn(num_tokens, num_heads, dtype=dtype, device=device)
    yield (
        positions,
        q,
        cos_sin_cache,
        weights,
        HEAD_DIM**-0.5,
        num_heads**-0.5,
        use_fp4,
    )


class FusedIndexerQRopeQuantBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (1, NUM_HEADS, False),
        (32, NUM_HEADS, False),
        (32, NUM_HEADS, True),
    ]
    DEFAULT_SHAPE_DESC = "num_tokens, num_heads, use_fp4"

    def init_user_config(self):
        super().init_user_config()
        if any(len(shape) != 3 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES
        if not _supports_mxfp4_ptx():
            self.shapes = [shape for shape in self.shapes if not shape[2]]
        if not self.shapes:
            self.shapes = [(1, NUM_HEADS, False)]


@pytest.mark.fused_indexer_q_rope_quant
@pytest.mark.skipif(flag_gems.device != "cuda", reason="requires CUDA")
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
def test_fused_indexer_q_rope_quant():
    bench = FusedIndexerQRopeQuantBenchmark(
        op_name="fused_indexer_q_rope_quant",
        input_fn=_input_fn,
        torch_op=_reference_op,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.fused_indexer_q_rope_quant)
    bench.run()
