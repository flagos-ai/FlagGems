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


def fp32_math_attention_forward(
    q,
    k,
    v,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    **extra_kwargs,
):
    del (
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        max_q,
        max_k,
        dropout_p,
        return_debug_mask,
        extra_kwargs,
    )
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scale_tensor = q.new_full((), scale)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale_tensor
    if is_causal:
        causal_mask = torch.ones(
            (q.shape[1], k.shape[1]), dtype=torch.bool, device=q.device
        ).triu(diagonal=1)
        scores.masked_fill_(causal_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probabilities, v)
    return out, lse, None, None, None


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


class HygonFP32FlashAttentionForwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        del shape_file_path
        self.shapes = [
            (
                1,
                8,
                8,
                seq_len,
                seq_len,
                head_size,
                is_causal,
                0.0,
                False,
                None,
                None,
                False,
            )
            for seq_len in (256, 1024, 4096)
            for head_size in (64, 128)
            for is_causal in (False, True)
        ]

    def set_more_shapes(self):
        return []


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


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon-specific FP32 path")
@pytest.mark.underscore_flash_attention_forward
def test_hygon_fp32_flash_attention_forward():
    bench = HygonFP32FlashAttentionForwardBenchmark(
        op_name="_flash_attention_forward",
        input_fn=_flash_attention_forward_input_fn,
        torch_op=fp32_math_attention_forward,
        dtypes=[torch.float32],
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
