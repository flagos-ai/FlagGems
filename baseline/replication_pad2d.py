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

"""NVIDIA native baseline for `replication_pad2d`.

Mirrors benchmark/test_replication_pad2d.py (ReplicationPad2dBenchmark, a
base.Benchmark subclass):
    op_name  = "replication_pad2d"
    torch_op = torch.ops.aten.replication_pad2d
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own (shape, padding) pairs and returns
None from set_more_shapes, so core_shapes.yaml is not consulted. The padding is
[left, right, top, bottom] and the last entry uses an asymmetric (1, 2, 3, 4).
"""

import torch

OP_NAME = "replication_pad2d"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.replication_pad2d is unavailable on this torch build"
)

# ReplicationPad2dBenchmark.set_shapes -- (shape, padding)
_SHAPES = [
    ((2, 3, 8, 8), (1, 1, 2, 2)),
    ((4, 8, 64, 64), (2, 2, 2, 2)),
    ((16, 32, 56, 56), (2, 2, 2, 2)),
    ((32, 64, 28, 28), (3, 3, 3, 3)),
    ((64, 128, 14, 14), (1, 2, 3, 4)),
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


def native():
    op = _resolve("torch.ops.aten.replication_pad2d")
    return op if callable(op) else None


def grid():
    return [
        {"shape": list(shape), "padding": list(padding)} for shape, padding in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    x = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (x, list(binding["padding"])), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    pad = "".join(str(p) for p in binding["padding"])
    return f"{shape}_pad{pad}"


def config(binding, dtype):
    shape = list(binding["shape"])
    padding = list(binding["padding"])
    pad_left, pad_right, pad_top, pad_bottom = padding
    out_shape = [
        *shape[:-2],
        shape[-2] + pad_top + pad_bottom,
        shape[-1] + pad_left + pad_right,
    ]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                padding,
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"shape": shape, "padding": padding},
    }
