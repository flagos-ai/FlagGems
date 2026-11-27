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

"""NVIDIA native baseline for `channel_shuffle`.

Mirrors benchmark/test_channel_shuffle.py (ChannelShuffleBenchmark, a
base.Benchmark subclass):
    op_name  = "channel_shuffle"
    torch_op = torch.ops.aten.channel_shuffle
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with its own (NCHW shape, groups) pairs and
returns None from set_more_shapes, so core_shapes.yaml is not consulted and
nothing is added at --level comprehensive either.
"""

import torch

OP_NAME = "channel_shuffle"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.channel_shuffle is unavailable on this torch build"
)

# ChannelShuffleBenchmark.set_shapes -- (NCHW shape, groups)
_SHAPES = [
    ((1, 4, 2, 2), 2),
    ((2, 8, 4, 4), 4),
    ((4, 16, 8, 8), 4),
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
    op = _resolve("torch.ops.aten.channel_shuffle")
    return op if callable(op) else None


def grid():
    return [{"shape": list(shape), "groups": groups} for shape, groups in _SHAPES]


def build_inputs(binding, dtype, device):
    x = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (x, binding["groups"]), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_g{binding['groups']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": binding["groups"]},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {
            "N": shape[0],
            "C": shape[1],
            "H": shape[2],
            "W": shape[3],
            "groups": binding["groups"],
        },
    }
