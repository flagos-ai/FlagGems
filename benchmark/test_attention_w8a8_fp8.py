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
from flag_gems.ops.attention_w8a8_fp8 import attention_w8a8_fp8

from . import base, utils


def _supports_hopper_fp8() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _get_fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    return dtype


def _hadamard_matrix(dim, device):
    assert dim > 0 and dim & (dim - 1) == 0, "head_size must be a power of two"
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < dim:
        h = torch.cat(
            (
                torch.cat((h, h), dim=1),
                torch.cat((h, -h), dim=1),
            ),
            dim=0,
        )
    return h / math.sqrt(dim)


def _apply_incoherent_qk(x):
    h = _hadamard_matrix(x.shape[-1], x.device).to(torch.float32)
    return torch.matmul(x.float(), h).to(x.dtype)


def _quantize_per_block_fp8(x, fp8_dtype, block_size=128):
    batch, seq_len, num_head, _ = x.shape
    fp8_max = float(torch.finfo(fp8_dtype).max)
    nblocks = triton.cdiv(seq_len, block_size)

    out = torch.empty_like(x, dtype=fp8_dtype)
    descale = torch.empty(
        (batch, num_head, nblocks),
        device=x.device,
        dtype=torch.float32,
    )

    for block_idx in range(nblocks):
        lo = block_idx * block_size
        hi = min(seq_len, lo + block_size)
        tile = x[:, lo:hi, :, :].float()
        scale = (tile.abs().amax(dim=(1, 3)) / fp8_max).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        out[:, lo:hi, :, :] = torch.clamp(
            tile / scale[:, None, :, None],
            -fp8_max,
            fp8_max,
        ).to(fp8_dtype)
        descale[:, :, block_idx] = scale

    return out.contiguous(), descale.contiguous()


def _quantize_qkv_w8a8(q, k, v):
    fp8_dtype = _get_fp8_dtype()
    q_fp8, q_descale = _quantize_per_block_fp8(_apply_incoherent_qk(q), fp8_dtype)
    k_fp8, k_descale = _quantize_per_block_fp8(_apply_incoherent_qk(k), fp8_dtype)
    v_fp8, v_descale = _quantize_per_block_fp8(v, fp8_dtype)
    return (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    )


def torch_attention_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    scale,
    is_causal,
):
    result = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        0.0,
        is_causal,
        False,
        scale=scale,
    )
    return result[0]


def gems_attention_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    scale,
    is_causal,
):
    out = torch.empty_like(q, dtype=torch.float16)
    return attention_w8a8_fp8(
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale=scale,
        is_causal=is_causal,
        out=out,
    )


class AttentionW8A8FP8Benchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = []

        for batch in (1, 2, 4, 8):
            self.shapes.extend(
                [
                    (batch, 512, 16, 128, False),
                    (batch, 512, 32, 64, False),
                    (batch, 512, 16, 128, True),
                    (batch, 512, 32, 64, True),
                ]
            )

        for batch in (1, 2, 4, 8):
            for seq_len in (1024, 2048, 4096, 8192):
                self.shapes.extend(
                    [
                        (batch, seq_len, 16, 128, False),
                        (batch, seq_len, 32, 64, False),
                    ]
                )

    def set_more_shapes(self):
        return []


def attention_w8a8_fp8_input_fn(config, dtype, device):
    batch, seq_len, num_head, head_size, is_causal = config
    q = torch.empty(
        (batch, seq_len, num_head, head_size),
        device=device,
        dtype=dtype,
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, seq_len, num_head, head_size),
        device=device,
        dtype=dtype,
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, seq_len, num_head, head_size),
        device=device,
        dtype=dtype,
    ).uniform_(-0.05, 0.05)
    scale = 1.0 / math.sqrt(head_size)

    (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    ) = _quantize_qkv_w8a8(q, k, v)

    yield (
        q,
        k,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        scale,
        is_causal,
    )


@pytest.mark.skipif(utils.SkipVersion("torch", "<2.4"), reason="Low Pytorch Version.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.skipif(flag_gems.vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.attention_w8a8_fp8
def test_attention_w8a8_fp8():
    bench = AttentionW8A8FP8Benchmark(
        op_name="attention_w8a8_fp8",
        input_fn=attention_w8a8_fp8_input_fn,
        torch_op=torch_attention_w8a8_fp8,
        dtypes=[torch.float16],
    )
    bench.set_gems(gems_attention_w8a8_fp8)
    bench.run()
