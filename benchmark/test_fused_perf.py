from typing import Optional

import pytest
import torch

import flag_gems

from .attri_util import FLOAT_DTYPES
from .performance_utils import (
    GenericBenchmark,
    GenericBenchmarkExcluse1D,
    binary_input_fn,
)


@pytest.mark.gelu_and_mul
def test_perf_gelu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.gelu(x), y)

    gems_op = flag_gems.gelu_and_mul
    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="gelu_and_mul",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.silu_and_mul
def test_perf_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    gems_op = flag_gems.silu_and_mul

    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="silu_and_mul",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.skip_layernorm
def test_perf_skip_layernorm():
    def skip_layernorm_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        layer_shape = (shape[-1],)
        weight = torch.randn(layer_shape, dtype=dtype, device=device)
        bias = torch.randn(layer_shape, dtype=dtype, device=device)
        yield inp, residual, layer_shape, weight, bias

    def torch_op(inp, residual, layer_shape, weight, bias):
        return torch.layer_norm(inp + residual, layer_shape, weight, bias)

    gems_op = flag_gems.skip_layer_norm

    bench = GenericBenchmarkExcluse1D(
        input_fn=skip_layernorm_input_fn,
        op_name="skip_layernorm",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.fused_add_rms_norm
def test_perf_fused_add_rms_norm():
    def fused_add_rms_norm_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        layer_shape = (shape[-1],)
        weight = torch.randn(layer_shape, dtype=dtype, device=device)
        yield inp, residual, layer_shape, weight, 1e-5

    def torch_op(x, residual, layer_shape, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    gems_op = flag_gems.fused_add_rms_norm

    bench = GenericBenchmarkExcluse1D(
        input_fn=fused_add_rms_norm_input_fn,
        op_name="fused_add_rms_norm",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flag_gems.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


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


class RopeBenchmark(GenericBenchmark):
    """
    benchmark for rope
    """

    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return None


@pytest.mark.apply_rotary_pos_emb
def test_perf_apply_rotary_pos_emb():
    batch_size = 4
    q_heads = 8
    k_heads = 1
    head_dim = 64

    def rope_input_fn(shape, dtype, device):
        seq_len = shape[0]
        q = torch.randn(
            (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=device
        )
        k = torch.randn(
            (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=device
        )
        cos, sin = get_rope_cos_sin(seq_len, head_dim, dtype, device=device)
        yield q, k, cos, sin

    torch_op = torch_apply_rotary_pos_emb
    gems_op = flag_gems.apply_rotary_pos_emb

    bench = RopeBenchmark(
        input_fn=rope_input_fn,
        op_name="apply_rotary_pos_emb",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()
