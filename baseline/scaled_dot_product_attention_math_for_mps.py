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

"""NVIDIA native baseline for `scaled_dot_product_attention_math_for_mps`.

Mirrors benchmark/test_scaled_dot_product_attention_math_for_mps.py
(ScaledDotProductAttentionMathForMpsBenchmark, a GenericBenchmark subclass):
    op_name  = "scaled_dot_product_attention_math_for_mps"
    torch_op = torch_ref (defined inside the test function)
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes (not set_more_shapes) with its own 4-D
attention shapes so the generic core_shapes list cannot replace them.

The test is parametrized over is_causal in [False, True] under one op_name, so
each shape contributes two collection points; the grid carries the flag.

This operator is the MPS math fallback and has no CUDA kernel. The benchmark's
reference is an explicit matmul / causal-mask / softmax / matmul composition,
reproduced verbatim below, and it returns (output, attn_weights). The measured
work is therefore several kernels, not one. See Q&A.md.
"""

import torch

OP_NAME = "scaled_dot_product_attention_math_for_mps"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.matmul is unavailable on this torch build"

# ScaledDotProductAttentionMathForMpsBenchmark.set_shapes
# (batch, num_heads, seq_len, head_size)
_SHAPES = [
    (2, 8, 64, 64),
    (4, 8, 128, 64),
    (2, 8, 128, 128),
    (1, 16, 256, 64),
]

_DROPOUT_P = 0.0


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


def _torch_ref(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    dropout_mask=None,
    scale=None,
):
    """benchmark::torch_ref -- explicit math attention, returns (out, weights)."""
    head_dim = query.shape[-1]
    scale_factor = 1.0 / (head_dim**0.5) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=query.device).tril(
            diagonal=seq_k - seq_q
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


def native():
    if _resolve("torch.matmul") is None:
        return None
    return _torch_ref


def grid():
    return [
        {"shape": list(s), "is_causal": is_causal}
        for s in _SHAPES
        for is_causal in (False, True)
    ]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    return (
        query,
        key,
        value,
        None,
        _DROPOUT_P,
        binding["is_causal"],
        None,
    ), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_causal{int(binding['is_causal'])}"


def config(binding, dtype):
    shape = list(binding["shape"])
    batch, heads, seq_len, head_size = shape
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "query"},
                {"shape": shape, "dtype": dt, "note": "key"},
                {"shape": shape, "dtype": dt, "note": "value"},
                None,
                {"scalar": _DROPOUT_P},
                {"scalar": binding["is_causal"]},
                None,
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": dt},
            "attn_weights": {
                "shape": [batch, heads, seq_len, seq_len],
                "dtype": dt,
            },
        },
        "dims": {
            "batch": batch,
            "num_heads": heads,
            "seq_len": seq_len,
            "head_size": head_size,
            "is_causal": binding["is_causal"],
        },
    }
