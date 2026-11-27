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

"""NVIDIA native baseline for `efficient_attention_backward`.

Mirrors benchmark/test_flash_attention_backward.py
(EfficientAttentionBackwardBenchmark, a GenericBenchmark subclass):
    op_name  = "efficient_attention_backward"
    torch_op = _efficient_attn_bwd_aten (wraps
               torch.ops.aten._efficient_attention_backward)
    dtypes   = [torch.float16, torch.bfloat16]

The class overrides set_shapes to build its own
(batch, num_heads, seq_len, head_size, custom_mask_type, has_bias, _) list and
returns None from set_more_shapes, so core_shapes.yaml is not consulted. That
list is reproduced below in the same order.

custom_mask_type 1 means causal. has_bias adds a
[batch, num_heads, seq_len, seq_len] bias and sets bias_requires_grad.
dropout_p is 0.0 for every core row, and scale is 1/sqrt(head_size).

The saved forward state (out, lse_aligned, lse, philox seed/offset) comes from
a real torch.ops.aten._efficient_attention_forward call, which build_inputs
performs outside the timed region exactly as the benchmark's input_fn does;
the philox state is pinned with the benchmark's seed (1234567890, 0) first.
The forward runs in BHSD layout and the backward operands are permuted to BSHD,
as in the benchmark.

The wrapper passes scale as a keyword and num_splits_key=0, matching
_efficient_attn_bwd_aten. The metax/hygon/thead native-probe path in
_AttentionBackwardBenchmark.get_input_iter is not reproduced; it does not apply
to NVIDIA.
"""

import math

import torch

OP_NAME = "efficient_attention_backward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._efficient_attention_backward is unavailable on this build"
)

# EfficientAttentionBackwardBenchmark.set_shapes, in build order.
# (batch, num_heads, seq_len, head_size, custom_mask_type, has_bias, _)
_SAFE_HEAD_SIZES = (64, 128)


def _build_shapes():
    shapes = []
    for head_size in _SAFE_HEAD_SIZES:
        shapes.append((4, 16, 1024, head_size, 0, False, False))
        shapes.append((4, 16, 1024, head_size, 1, False, False))
        shapes.append((2, 16, 2048, head_size, 0, False, False))
        shapes.append((2, 16, 2048, head_size, 1, False, False))
    for head_size in (64, 128):
        shapes.append((2, 8, 512, head_size, 0, True, False))
        shapes.append((2, 8, 1024, head_size, 0, True, False))
    for batch, num_heads, seq_len in (
        (2, 16, 512),
        (4, 16, 1024),
        (1, 32, 1024),
        (8, 8, 1024),
        (1, 8, 2048),
        (2, 16, 2048),
    ):
        shapes.append((batch, num_heads, seq_len, 128, 0, False, False))
        shapes.append((batch, num_heads, seq_len, 128, 1, False, False))
    shapes.append((1, 8, 4096, 128, 0, False, False))
    shapes.append((1, 8, 4096, 64, 1, False, False))
    shapes.append((2, 16, 4096, 128, 0, False, False))
    shapes.append((4, 16, 4096, 128, 0, False, False))
    return shapes


_SHAPES = _build_shapes()

_PHILOX_SEED = 1234567890
_PHILOX_OFFSET = 0
_DROPOUT_P = 0.0
_NUM_SPLITS_KEY = 0


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _set_philox_state(seed, offset, device):
    """flag_gems.utils.random_utils.set_philox_state, CUDA branch only."""
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    gen = torch.cuda.default_generators[index]
    state_copy = gen.get_state()
    state_copy.view(torch.int64)[0] = seed
    state_copy.view(torch.int64)[1] = offset
    gen.set_state(state_copy)


def _efficient_attn_forward_native(q, k, v, bias, is_causal, scale):
    """benchmark::efficient_attn_forward_native -- the saved backward state."""
    q_fa = q.permute(0, 2, 1, 3).contiguous()
    k_fa = k.permute(0, 2, 1, 3).contiguous()
    v_fa = v.permute(0, 2, 1, 3).contiguous()
    results = torch.ops.aten._efficient_attention_forward(
        q_fa,
        k_fa,
        v_fa,
        bias,
        None,
        None,
        q.shape[2],
        k.shape[2],
        dropout_p=_DROPOUT_P,
        custom_mask_type=1 if is_causal else 0,
        compute_log_sumexp=True,
        scale=scale,
    )
    out_fa = results[0]
    lse_aligned = results[1]
    philox_seed = results[2]
    philox_offset = results[3]
    out = out_fa.permute(0, 2, 1, 3).contiguous()
    lse = lse_aligned[:, :, : q.shape[2]].float()
    return out, lse_aligned.float(), lse, philox_seed, philox_offset


def _efficient_attn_bwd_aten(
    d_out_bshd,
    q_bshd,
    k_bshd,
    v_bshd,
    attn_bias,
    out_bshd,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_aligned,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    lse,
    scale=None,
):
    """benchmark::_efficient_attn_bwd_aten -- `lse` is accepted but unused."""
    return torch.ops.aten._efficient_attention_backward(
        d_out_bshd,
        q_bshd,
        k_bshd,
        v_bshd,
        attn_bias,
        out_bshd,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        lse_aligned,
        dropout_p,
        philox_seed,
        philox_offset,
        custom_mask_type,
        bias_requires_grad,
        scale=scale,
        num_splits_key=_NUM_SPLITS_KEY,
    )


def native():
    if _resolve("torch.ops.aten._efficient_attention_backward") is None:
        return None
    return _efficient_attn_bwd_aten


def grid():
    # `ordinal` disambiguates the key: EfficientAttentionBackwardBenchmark's own
    # shape list repeats four (batch, heads, seq_len, 128, mask, bias) rows
    # between its head-size loop and its (batch, heads, seq_len) loop, and two
    # grid points must not collapse onto one JSON key.
    return [
        {
            "ordinal": i,
            "batch": batch,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_size": head_size,
            "custom_mask_type": custom_mask_type,
            "has_bias": has_bias,
        }
        for i, (
            batch,
            num_heads,
            seq_len,
            head_size,
            custom_mask_type,
            has_bias,
            _,
        ) in enumerate(_SHAPES)
    ]


def unsupported_reason(binding, dtype):
    # benchmark::_is_unsafe_decode_causal -- seq_len 1 with a causal mask.
    if binding["seq_len"] == 1 and binding["custom_mask_type"] == 1:
        return "seq_len == 1 with a causal mask is an unsafe decode shape"
    return None


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    num_heads = binding["num_heads"]
    seq_len = binding["seq_len"]
    head_size = binding["head_size"]
    custom_mask_type = binding["custom_mask_type"]
    has_bias = binding["has_bias"]
    is_causal = custom_mask_type == 1
    scale = 1.0 / math.sqrt(head_size)

    _set_philox_state(_PHILOX_SEED, _PHILOX_OFFSET, device)

    q_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    k_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    v_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    bias = (
        torch.randn(batch, num_heads, seq_len, seq_len, device=device, dtype=dtype)
        * 0.1
        if has_bias
        else None
    )
    d_out_bhsd = torch.randn(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    )

    (
        out_bhsd_tmp,
        lse_aligned,
        lse,
        philox_seed,
        philox_offset,
    ) = _efficient_attn_forward_native(q_bhsd, k_bhsd, v_bhsd, bias, is_causal, scale)

    q_bshd = q_bhsd.permute(0, 2, 1, 3).contiguous()
    k_bshd = k_bhsd.permute(0, 2, 1, 3).contiguous()
    v_bshd = v_bhsd.permute(0, 2, 1, 3).contiguous()
    out_bshd = out_bhsd_tmp.permute(0, 2, 1, 3).contiguous()
    d_out_bshd = d_out_bhsd.permute(0, 2, 1, 3).contiguous()

    return (
        d_out_bshd,
        q_bshd,
        k_bshd,
        v_bshd,
        bias,
        out_bshd,
        None,
        None,
        seq_len,
        seq_len,
        lse_aligned,
        _DROPOUT_P,
        philox_seed,
        philox_offset,
        custom_mask_type,
        has_bias,
        lse,
    ), {"scale": scale}


def key_shape(binding):
    return (
        f"c{binding['ordinal']:02d}"
        f"_b{binding['batch']}_h{binding['num_heads']}"
        f"_s{binding['seq_len']}_d{binding['head_size']}"
        f"_mask{binding['custom_mask_type']}"
        f"_bias{int(binding['has_bias'])}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    num_heads = binding["num_heads"]
    seq_len = binding["seq_len"]
    head_size = binding["head_size"]
    has_bias = binding["has_bias"]
    dt = str(dtype)
    bshd = [batch, seq_len, num_heads, head_size]
    bias_shape = [batch, num_heads, seq_len, seq_len] if has_bias else None
    return {
        "inputs": {
            "args": [
                {"shape": bshd, "dtype": dt, "note": "grad_out, BSHD"},
                {"shape": bshd, "dtype": dt, "note": "query, BSHD"},
                {"shape": bshd, "dtype": dt, "note": "key, BSHD"},
                {"shape": bshd, "dtype": dt, "note": "value, BSHD"},
                ({"shape": bias_shape, "dtype": dt} if has_bias else None),
                {"shape": bshd, "dtype": dt, "note": "forward out, BSHD"},
                None,
                None,
                {"scalar": seq_len},
                {"scalar": seq_len},
                {"dtype": "torch.float32", "note": "lse_aligned from forward"},
                {"scalar": _DROPOUT_P},
                {"note": "philox seed from forward"},
                {"note": "philox offset from forward"},
                {"scalar": binding["custom_mask_type"]},
                {"scalar": has_bias},
                {"dtype": "torch.float32", "note": "lse (unused by the aten call)"},
            ],
            "kwargs": {"scale": {"scalar": 1.0 / math.sqrt(head_size)}},
        },
        "outputs": {
            "grad_query": {"shape": bshd, "dtype": dt},
            "grad_key": {"shape": bshd, "dtype": dt},
            "grad_value": {"shape": bshd, "dtype": dt},
            "grad_bias": ({"shape": bias_shape, "dtype": dt} if has_bias else None),
        },
        "dims": {
            "ordinal": binding["ordinal"],
            "batch": batch,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_size": head_size,
            "custom_mask_type": binding["custom_mask_type"],
            "has_bias": has_bias,
        },
    }
