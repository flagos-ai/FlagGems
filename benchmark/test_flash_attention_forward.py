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
from flag_gems.ops import flash_attn_varlen_func_w8a8_fp8 as w8a8_varlen
from flag_gems.ops.attention import flash_attn_varlen_func as fa2_varlen

from . import base, utils


def torch_flash_attention_forward(
    q, k, v, scale, is_causal, dropout_p=0.0, return_debug_mask=False, **extra_kwargs
):
    return torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )


def gems_flash_attention_forward(
    q, k, v, scale, is_causal, dropout_p=0.0, return_debug_mask=False, **extra_kwargs
):
    return flag_gems.flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )


def torch_flash_attention_supports_alibi(device: str) -> bool:
    if device == "cpu" or not torch.cuda.is_available():
        return False

    try:
        q = torch.randn((1, 16, 1, 64), device=device, dtype=torch.float16)
        k = torch.randn((1, 16, 1, 64), device=device, dtype=torch.float16)
        v = torch.randn((1, 16, 1, 64), device=device, dtype=torch.float16)
        scale = float(1.0 / math.sqrt(64))
        alibi_slopes = torch.ones((1, 1), device=device, dtype=torch.float32) * 0.3
        torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            None,
            None,
            q.shape[-3],
            k.shape[-3],
            0.0,
            False,
            False,
            scale=scale,
            alibi_slopes=alibi_slopes,
        )
        return True
    except RuntimeError as e:
        if "does not support alibi" in str(e).lower():
            return False
        raise


class FlashAttentionForwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = []
        for head_size in (64, 128, 192, 256):
            for is_causal in (False, True):
                self.shapes.append(
                    (
                        4,
                        8,
                        8,
                        1024,
                        128,
                        head_size,
                        is_causal,
                        0.0,
                        False,
                        None,
                        None,
                        False,
                    )
                )

        for batch, num_head, q_seq_len, kv_seq_len in (
            (1, 1, 128, 2048),
            (4, 8, 17, 1030),
        ):
            for is_causal in (False, True):
                self.shapes.append(
                    (
                        batch,
                        num_head,
                        num_head,
                        q_seq_len,
                        kv_seq_len,
                        128,
                        is_causal,
                        0.0,
                        False,
                        None,
                        None,
                        False,
                    )
                )

        supports_alibi = torch_flash_attention_supports_alibi(self.device)
        if supports_alibi:
            # GQA + alibi cases
            for head_size in (128, 192):
                for is_causal in (False, True):
                    self.shapes.append(
                        (
                            4,
                            8,
                            2,
                            1024,
                            1024,
                            head_size,
                            is_causal,
                            0.0,
                            False,
                            None,
                            None,
                            True,
                        )
                    )
            for is_causal in (False, True):
                self.shapes.append(
                    (4, 4, 4, 1, 519, 128, is_causal, 0.0, False, None, None, True)
                )

        # Split-KV like cases (q_seq_len=1, num_head_k < num_head).
        for is_causal in (False, True):
            self.shapes.append(
                (1, 4, 1, 1, 1024, 128, is_causal, 0.0, False, None, None, False)
            )
            if supports_alibi:
                self.shapes.append(
                    (1, 4, 1, 1, 1024, 128, is_causal, 0.0, False, None, None, True)
                )

        # Sliding window attention.
        for batch, num_head, q_seq_len, kv_seq_len in (
            (1, 1, 128, 2048),
            (8, 32, 1024, 1024),
            (8, 32, 1024, 128),
            (8, 32, 17, 1030),
        ):
            for window_size_left, window_size_right in ((256, 0), (128, 128)):
                self.shapes.append(
                    (
                        batch,
                        num_head,
                        num_head,
                        q_seq_len,
                        kv_seq_len,
                        128,
                        False,
                        0.0,
                        False,
                        window_size_left,
                        window_size_right,
                        False,
                    )
                )
        self.shapes.append(
            (8, 32, 32, 1024, 1024, 192, False, 0.0, False, 256, 0, False)
        )

        for is_causal in (False, True):
            self.shapes.append(
                (1, 1, 1, 1024, 1024, 128, is_causal, 0.2, True, None, None, False)
            )

    def set_more_shapes(self):
        return []


def flash_attention_forward_input_fn(config, dtype, device):
    (
        batch,
        num_head,
        num_head_k,
        q_seq_len,
        kv_seq_len,
        head_size,
        is_causal,
        dropout_p,
        return_debug_mask,
        window_size_left,
        window_size_right,
        use_alibi,
    ) = config

    q = torch.empty(
        (batch, q_seq_len, num_head, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, kv_seq_len, num_head_k, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, kv_seq_len, num_head_k, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    scale = float(1.0 / math.sqrt(head_size))

    extra_kwargs = {}
    if window_size_left is not None or window_size_right is not None:
        extra_kwargs.update(
            {
                "window_size_left": window_size_left,
                "window_size_right": window_size_right,
            }
        )
    if use_alibi:
        extra_kwargs["alibi_slopes"] = (
            torch.ones(batch, num_head, device=device, dtype=torch.float32) * 0.3
        )

    yield q, k, v, scale, is_causal, dropout_p, return_debug_mask, extra_kwargs


def _flash_attention_forward_input_fn(config, dtype, device):
    for (
        q,
        k,
        v,
        scale,
        is_causal,
        dropout_p,
        return_debug_mask,
        kwargs,
    ) in flash_attention_forward_input_fn(config, dtype, device):
        kwargs["scale"] = scale
        yield (
            q,
            k,
            v,
            None,
            None,
            q.shape[-3],
            k.shape[-3],
            dropout_p,
            is_causal,
            return_debug_mask,
            kwargs,
        )


@pytest.mark.underscore_flash_attention_forward
def test__flash_attention_forward():
    bench = FlashAttentionForwardBenchmark(
        op_name="_flash_attention_forward",
        input_fn=_flash_attention_forward_input_fn,
        torch_op=torch.ops.aten._flash_attention_forward.default,
        # FlashAttention supports CUDA float16 and bfloat16 inputs.
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


@pytest.mark.skipif(utils.SkipVersion("torch", "<2.4"), reason="Low Pytorch Version.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.flash_attention_forward
def test_flash_attention_forward():
    bench = FlashAttentionForwardBenchmark(
        op_name="flash_attention_forward",
        input_fn=flash_attention_forward_input_fn,
        torch_op=torch_flash_attention_forward,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(gems_flash_attention_forward)
    bench.run()


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


def _quantize_varlen_per_block_fp8(x, seq_lens, fp8_dtype, block_size=128):
    total_tokens, num_head, _ = x.shape
    assert total_tokens == sum(seq_lens)

    fp8_max = float(torch.finfo(fp8_dtype).max)
    nblocks = triton.cdiv(max(seq_lens), block_size)

    out = torch.empty_like(x, dtype=fp8_dtype)
    descale = torch.ones(
        (len(seq_lens), num_head, nblocks),
        device=x.device,
        dtype=torch.float32,
    )

    token_offset = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        for block_idx in range(triton.cdiv(seq_len, block_size)):
            lo = token_offset + block_idx * block_size
            hi = min(token_offset + seq_len, lo + block_size)
            tile = x[lo:hi].float()
            scale = (tile.abs().amax(dim=(0, 2)) / fp8_max).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            out[lo:hi] = torch.clamp(
                tile / scale[None, :, None],
                -fp8_max,
                fp8_max,
            ).to(fp8_dtype)
            descale[batch_idx, :, block_idx] = scale
        token_offset += seq_len

    return out.contiguous(), descale.contiguous()


def _quantize_qkv_w8a8(q, k, v, q_seq_lens, kv_seq_lens):
    fp8_dtype = _get_fp8_dtype()
    q_fp8, q_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(q), q_seq_lens, fp8_dtype
    )
    k_fp8, k_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(k), kv_seq_lens, fp8_dtype
    )
    v_fp8, v_descale = _quantize_varlen_per_block_fp8(v, kv_seq_lens, fp8_dtype)
    return (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    )


def baseline_flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    scale,
    is_causal,
    baseline_out,
    w8a8_out,
):
    return fa2_varlen(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale=scale,
        causal=is_causal,
        out=baseline_out,
    )


def gems_flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    scale,
    is_causal,
    baseline_out,
    w8a8_out,
):
    return w8a8_varlen(
        q_fp8,
        k_fp8,
        v_fp8,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale=scale,
        causal=is_causal,
        out=w8a8_out,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


class FlashAttnVarlenFuncW8A8FP8Benchmark(base.GenericBenchmark):
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


def _make_deterministic_ragged_lengths(batch, max_seq_len):
    if batch == 1:
        return (max_seq_len,), (max_seq_len,)

    step = max(1, max_seq_len // (2 * batch))
    q_seq_lens = tuple(
        max(1, max_seq_len - batch_idx * step - batch_idx % 3)
        for batch_idx in range(batch)
    )
    kv_seq_lens = q_seq_lens[1:] + q_seq_lens[:1]
    return q_seq_lens, kv_seq_lens


def _make_cu_seqlens(seq_lens, device):
    return torch.tensor(
        (0,) + seq_lens,
        device=device,
        dtype=torch.int32,
    ).cumsum(dim=0, dtype=torch.int32)


def flash_attn_varlen_func_w8a8_fp8_input_fn(config, dtype, device):
    batch, seq_len, num_head, head_size, is_causal = config
    q_seq_lens, kv_seq_lens = _make_deterministic_ragged_lengths(batch, seq_len)
    q = torch.empty(
        (sum(q_seq_lens), num_head, head_size),
        device=device,
        dtype=dtype,
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (sum(kv_seq_lens), num_head, head_size),
        device=device,
        dtype=dtype,
    ).uniform_(-0.05, 0.05)
    v = torch.empty_like(k).uniform_(-0.05, 0.05)
    scale = 1.0 / math.sqrt(head_size)

    (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    ) = _quantize_qkv_w8a8(q, k, v, q_seq_lens, kv_seq_lens)
    cu_seqlens_q = _make_cu_seqlens(q_seq_lens, device)
    cu_seqlens_k = _make_cu_seqlens(kv_seq_lens, device)
    baseline_out = torch.empty_like(q)
    w8a8_out = torch.empty_like(q)

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
        max(q_seq_lens),
        cu_seqlens_q,
        max(kv_seq_lens),
        cu_seqlens_k,
        scale,
        is_causal,
        baseline_out,
        w8a8_out,
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
@pytest.mark.flash_attn_varlen_func_w8a8_fp8
def test_flash_attn_varlen_func_w8a8_fp8():
    bench = FlashAttnVarlenFuncW8A8FP8Benchmark(
        op_name="flash_attn_varlen_func_w8a8_fp8",
        input_fn=flash_attn_varlen_func_w8a8_fp8_input_fn,
        torch_op=baseline_flash_attn_varlen_func_w8a8_fp8,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(gems_flash_attn_varlen_func_w8a8_fp8)
    bench.run()
