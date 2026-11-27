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

"""NVIDIA native baseline for `sparse_semi_structured_linear`.

Mirrors benchmark/test_sparse_semi_structured_linear.py
(SparseSemiStructuredLinearBenchmark, a base.Benchmark subclass):
    op_name  = "sparse_semi_structured_linear"
    torch_op = _torch_ref (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES (the class passes it in __init__)

The class overrides set_shapes with its own (M, K) list and declares shape_desc
"M, K"; set_more_shapes only applies at --level comprehensive, so the two extra
shapes are not in this grid. N equals K -- output features match input
features.

Two tests share this op_name: `use_bias=False` and `use_bias=True`. Both
record under "sparse_semi_structured_linear", so the grid carries the flag and
both variants are collected.

CAUTION: the reference is not a sparse kernel. `_torch_ref` applies the 2:4
meta mask to the dense weight and runs a dense torch.matmul, so the baseline
measures mask + dense GEMM rather than a semi-structured sparse GEMM. That
composition is reproduced here. See Q&A.md.

`_build_2_4_weight` builds the weight and its meta together: within each group
of four columns the `choice` bit keeps either the first two or the last two.
"""

import torch

OP_NAME = "sparse_semi_structured_linear"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.matmul is unavailable on this torch build"

# SparseSemiStructuredLinearBenchmark.set_shapes -- (M, K); K is a multiple of 4
_SHAPES = [
    (16, 32),
    (64, 128),
    (256, 512),
]


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


def _build_2_4_weight(n, k, dtype, device):
    """benchmark::_build_2_4_weight -- a 2:4 weight and its meta selector."""
    k4 = k // 4
    base_w = torch.randn(n, k, dtype=dtype, device=device)
    choice = torch.randint(0, 2, (n, k4), device=device).bool()
    w = torch.zeros(n, k, dtype=dtype, device=device)
    for g in range(k4):
        keep = choice[:, g]
        w[:, 4 * g] = torch.where(
            keep, base_w[:, 4 * g], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 1] = torch.where(
            keep, base_w[:, 4 * g + 1], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 2] = torch.where(
            ~keep, base_w[:, 4 * g + 2], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 3] = torch.where(
            ~keep, base_w[:, 4 * g + 3], torch.zeros_like(keep, dtype=dtype)
        )
    return w, choice.to(torch.int8)


def _torch_ref(inp, weight, meta):
    """benchmark::_torch_ref -- dense matmul with the 2:4 meta applied."""
    k4 = weight.shape[1] // 4
    choice = meta.to(torch.bool)
    w = weight.view(weight.shape[0], k4, 4)
    keep = torch.cat(
        [
            choice.unsqueeze(2),
            choice.unsqueeze(2),
            (~choice).unsqueeze(2),
            (~choice).unsqueeze(2),
        ],
        dim=2,
    )
    masked = torch.where(keep, w, torch.zeros_like(w)).reshape(weight.shape[0], -1)
    return torch.matmul(inp, masked.t())


def _torch_ref_with_bias(inp, weight, meta, bias):
    """benchmark::_torch_ref_with_bias."""
    return _torch_ref(inp, weight, meta) + bias


def _dispatch(inp, weight, meta, bias=None):
    """Route to the reference the matching test would use."""
    if bias is None:
        return _torch_ref(inp, weight, meta)
    return _torch_ref_with_bias(inp, weight, meta, bias)


def native():
    if _resolve("torch.matmul") is None:
        return None
    return _dispatch


def grid():
    return [
        {"m": m, "k": k, "use_bias": use_bias}
        for m, k in _SHAPES
        for use_bias in (False, True)
    ]


def build_inputs(binding, dtype, device):
    m, k = binding["m"], binding["k"]
    n = k  # output features equal input features
    inp = torch.randn(m, k, dtype=dtype, device=device)
    weight, meta = _build_2_4_weight(n, k, dtype, device)
    if binding["use_bias"]:
        bias = torch.randn(n, dtype=dtype, device=device)
        return (inp, weight, meta, bias), {}
    return (inp, weight, meta), {}


def key_shape(binding):
    return f"m{binding['m']}_k{binding['k']}_bias{int(binding['use_bias'])}"


def config(binding, dtype):
    m, k = binding["m"], binding["k"]
    n = k
    dt = str(dtype)
    args = [
        {"shape": [m, k], "dtype": dt},
        {"shape": [n, k], "dtype": dt, "note": "2:4 sparse"},
        {"shape": [n, k // 4], "dtype": "torch.int8", "note": "meta"},
    ]
    if binding["use_bias"]:
        args.append({"shape": [n], "dtype": dt})
    return {
        "inputs": {"args": args, "kwargs": {}},
        "outputs": {"out": {"shape": [m, n], "dtype": dt}},
        "dims": {"M": m, "K": k, "N": n, "use_bias": binding["use_bias"]},
    }
