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

"""NVIDIA native baseline for `nested_from_padded`.

Mirrors benchmark/test_nested_from_padded.py (NestedFromPaddedBenchmark, a
base.Benchmark subclass):
    op_name  = "nested_from_padded"
    torch_op = torch.ops.aten._nested_from_padded
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with NESTED_FROM_PADDED_SHAPES -- read as
(batch_size, max_len, head_dim) -- so core_shapes.yaml is not consulted.

Per-batch sequence lengths are drawn from np.random.randint(1, max_len + 1)
under seed 42, with component 0 pinned to the full padded length so the layout
is deterministic. The benchmark uses numpy for this; the same values are
produced here with a seeded numpy Generator-free call so the collector's two
build_inputs invocations agree.
"""

import numpy as np
import torch

OP_NAME = "nested_from_padded"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._nested_from_padded is unavailable on this torch build"
)

# test_nested_from_padded.py::NESTED_FROM_PADDED_SHAPES
# (batch_size, max_len, head_dim)
_SHAPES = [
    (8, 8, 64),
    (16, 16, 64),
    (32, 32, 64),
    (64, 64, 64),
    (128, 64, 64),
    (256, 128, 64),
    (512, 256, 128),
]

_SEED = 42


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


def native():
    op = _resolve("torch.ops.aten._nested_from_padded")
    return op if callable(op) else None


def grid():
    return [{"batch_size": b, "max_len": m, "head_dim": d} for b, m, d in _SHAPES]


def build_inputs(binding, dtype, device):
    batch_size = binding["batch_size"]
    max_len = binding["max_len"]
    head_dim = binding["head_dim"]
    np.random.seed(_SEED)
    lengths = np.random.randint(1, max_len + 1, size=batch_size)
    lengths[0] = max_len
    padded = torch.randn((batch_size, max_len, head_dim), dtype=dtype, device=device)
    sizes = torch.stack(
        [
            torch.from_numpy(lengths).to(torch.int64),
            torch.full((batch_size,), head_dim, dtype=torch.int64),
        ],
        dim=1,
    )
    return (padded, sizes), {}


def key_shape(binding):
    return [binding["batch_size"], binding["max_len"], binding["head_dim"]]


def config(binding, dtype):
    b = binding["batch_size"]
    m = binding["max_len"]
    d = binding["head_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [b, m, d], "dtype": dt},
                {"shape": [b, 2], "dtype": "torch.int64"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"dtype": dt, "note": "nested tensor; no concrete shape"}},
        "dims": {"batch_size": b, "max_len": m, "head_dim": d},
    }
