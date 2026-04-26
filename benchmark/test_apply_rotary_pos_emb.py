from typing import Optional

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark


class RopeBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return []


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flag_gems.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def rope_input_fn(shape, dtype, device):
    batch_size = 4
    q_heads = 8
    k_heads = 1
    head_dim = 64

    seq_len = shape[0]
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=device
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=device
    )
    cos, sin = get_rope_cos_sin(seq_len, head_dim, dtype, device=device)
    yield q, k, cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_fn(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
):
    q = q.float()
    k = k.float()
    cos = cos[None, : q.size(-3), None, :]
    sin = sin[None, : q.size(-3), None, :]
    cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
    sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


@pytest.mark.apply_rotary_pos_emb
def test_apply_rotary_pos_emb():
    bench = RopeBenchmark(
        input_fn=rope_input_fn,
        op_name="apply_rotary_pos_emb",
        torch_op=torch_apply_rotary_pos_emb,
        gems_op=flag_gems.apply_rotary_pos_emb,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
