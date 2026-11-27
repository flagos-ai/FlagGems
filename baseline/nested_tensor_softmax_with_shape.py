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

"""NVIDIA native baseline for `nested_tensor_softmax_with_shape`.

Mirrors benchmark/test_nested_tensor_softmax_with_shape.py
(NestedTensorSoftmaxWithShapeBenchmark, a base.Benchmark subclass):
    op_name  = "nested_tensor_softmax_with_shape"
    torch_op = _torch_ref (the module-local reference below)
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with NESTED_SOFTMAX_SHAPES -- read as
(B, H, T_max, seq_lens) -- so core_shapes.yaml is not consulted.

This operator has no native CUDA kernel. The benchmark's reference pads the
jagged tensor and runs a per-batch Python loop of torch.softmax over the valid
block, which is reproduced verbatim below. The measured latency is therefore a
host loop of B softmax launches, not one kernel. See Q&A.md.

Inputs are built as torch.nested jagged tensors from per-batch components, as
the benchmark builds them.
"""

import torch

OP_NAME = "nested_tensor_softmax_with_shape"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nested.nested_tensor is unavailable on this torch build"
)

# test_nested_tensor_softmax_with_shape.py::NESTED_SOFTMAX_SHAPES
# (B, H, T_max, seq_lens)
_SHAPES = [
    (2, 2, 16, [8, 16]),
    (4, 2, 32, [16, 24, 32, 32]),
    (8, 4, 64, [16, 32, 48, 64, 16, 32, 48, 64]),
]

_QUERY_DIM = 4


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


def _torch_ref(self_nt, query_nt):
    """benchmark::_torch_ref -- masked softmax via padding and a Python loop."""
    attn = self_nt.to_padded_tensor(0.0)
    offsets = query_nt.offsets()
    seq_lens = offsets[1:] - offsets[:-1]
    b_size = attn.shape[0]
    out = torch.zeros_like(attn)
    for b in range(b_size):
        t = int(seq_lens[b])
        if t <= 0:
            continue
        out[b, :t, :, :t] = torch.softmax(attn[b, :t, :, :t], dim=-1)
    return out


def native():
    if _resolve("torch.nested.nested_tensor") is None:
        return None
    return _torch_ref


def grid():
    return [
        {"B": b, "H": h, "T_max": t_max, "seq_lens": list(seq_lens)}
        for b, h, t_max, seq_lens in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    b_size = binding["B"]
    heads = binding["H"]
    t_max = binding["T_max"]
    seq_lens = binding["seq_lens"]
    self_components = [
        torch.randn(seq_lens[b], heads, t_max, dtype=dtype, device=device)
        for b in range(b_size)
    ]
    self_nt = torch.nested.nested_tensor(
        self_components, layout=torch.jagged, device=device
    )
    query_components = [
        torch.randn(seq_lens[b], _QUERY_DIM, dtype=dtype, device=device)
        for b in range(b_size)
    ]
    query_nt = torch.nested.nested_tensor(
        query_components, layout=torch.jagged, device=device
    )
    return (self_nt, query_nt), {}


def key_shape(binding):
    return f"B{binding['B']}_H{binding['H']}_T{binding['T_max']}"


def config(binding, dtype):
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "dtype": dt,
                    "note": "jagged nested tensor [sum(seq_lens), H, T_max]",
                },
                {
                    "dtype": dt,
                    "note": f"jagged nested tensor [sum(seq_lens), {_QUERY_DIM}]",
                },
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"dtype": dt, "note": "padded [B, T_max, H, T_max]"}},
        "dims": {
            "B": binding["B"],
            "H": binding["H"],
            "T_max": binding["T_max"],
            "seq_lens": list(binding["seq_lens"]),
        },
    }
