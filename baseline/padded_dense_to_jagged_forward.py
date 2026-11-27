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

"""NVIDIA native baseline for `padded_dense_to_jagged_forward`.

Mirrors benchmark/test_padded_dense_to_jagged_forward.py
(PaddedDenseToJaggedForwardBenchmark, a base.Benchmark subclass):
    op_name  = "padded_dense_to_jagged_forward"
    torch_op = torch.ops.aten._padded_dense_to_jagged_forward
    dtypes   = [float16, bfloat16, float32, float64, int64]
               (parametrized; the float16/float64 skips are mthreads- and
               iluvatar-only, so all five run on NVIDIA)

The class overrides set_shapes with PADDED_DENSE_TO_JAGGED_SHAPES -- read as
(batch_size, max_length, inner_size) -- so core_shapes.yaml is not consulted.

The test is also parametrized over offset_dtype in [int32, int64], which the
class reads off the instance; both are carried in the grid below.

Lengths come from a seeded numpy Generator (default_rng(42)) over
[max_length // 4, max_length], and the offsets are their cumulative sum
prefixed with 0. int64 inputs are drawn from randint(-1024, 1024) and the
float dtypes from randn.
"""

import numpy as np
import torch

OP_NAME = "padded_dense_to_jagged_forward"
DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int64,
]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._padded_dense_to_jagged_forward is unavailable on this build"
)

# test_padded_dense_to_jagged_forward.py::PADDED_DENSE_TO_JAGGED_SHAPES
# (batch_size, max_length, inner_size)
_SHAPES = [
    (8, 16, 64),
    (32, 64, 128),
    (128, 128, 64),
    (512, 256, 128),
    (1024, 512, 64),
]

_OFFSET_DTYPES = ("torch.int32", "torch.int64")
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


def _lengths(batch_size, max_length):
    rng = np.random.default_rng(_SEED)
    return rng.integers(max(1, max_length // 4), max_length + 1, size=batch_size)


def native():
    op = _resolve("torch.ops.aten._padded_dense_to_jagged_forward")
    return op if callable(op) else None


def grid():
    return [
        {
            "batch_size": b,
            "max_length": m,
            "inner_size": i,
            "offset_dtype": offset_dtype,
        }
        for b, m, i in _SHAPES
        for offset_dtype in _OFFSET_DTYPES
    ]


def build_inputs(binding, dtype, device):
    batch_size = binding["batch_size"]
    max_length = binding["max_length"]
    inner_size = binding["inner_size"]
    offset_dtype = (
        torch.int32 if binding["offset_dtype"] == "torch.int32" else torch.int64
    )
    lengths = _lengths(batch_size, max_length)
    offsets_values = np.concatenate(([0], np.cumsum(lengths)))
    offsets = torch.tensor(offsets_values, device=device, dtype=offset_dtype)
    if dtype == torch.int64:
        dense = torch.randint(
            -1024,
            1024,
            (batch_size, max_length, inner_size),
            dtype=dtype,
            device=device,
        )
    else:
        dense = torch.randn(
            (batch_size, max_length, inner_size), dtype=dtype, device=device
        )
    total_l = int(offsets_values[-1])
    return (dense, [offsets], total_l), {}


def key_shape(binding):
    offset = binding["offset_dtype"].split(".")[-1]
    return (
        f"b{binding['batch_size']}_m{binding['max_length']}"
        f"_i{binding['inner_size']}_{offset}"
    )


def config(binding, dtype):
    batch_size = binding["batch_size"]
    max_length = binding["max_length"]
    inner_size = binding["inner_size"]
    total_l = int(_lengths(batch_size, max_length).sum())
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [batch_size, max_length, inner_size], "dtype": dt},
                [{"shape": [batch_size + 1], "dtype": binding["offset_dtype"]}],
                {"scalar": total_l},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [total_l, inner_size], "dtype": dt}},
        "dims": {
            "batch_size": batch_size,
            "max_length": max_length,
            "inner_size": inner_size,
            "total_L": total_l,
            "offset_dtype": binding["offset_dtype"],
        },
    }
