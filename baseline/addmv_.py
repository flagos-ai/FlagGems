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

"""NVIDIA native baseline for `addmv_`.

Mirrors benchmark/test_addmv_.py (AddmvBenchmark, a GenericBenchmark2DOnly
subclass):
    op_name  = "addmv_"
    torch_op = lambda bias, mat, vec: bias.addmv_(mat, vec)
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml)

_input_fn reads each core shape as (m, n) and yields
(bias[m], mat[m, n], vec[n]); the result is accumulated into `bias` in place.

The benchmark leaves is_inplace unset even though Tensor.addmv_ writes back
into bias, so warmup iterations accumulate there. IS_INPLACE is set True here
so the collector restores bias outside every timed interval -- without it the
values diverge across repetitions and can reach inf in float16.
"""

import torch

OP_NAME = "addmv_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = "torch.Tensor.addmv_ is unavailable on this torch build"

# yaml:GenericBenchmark2DOnly; AddmvBenchmark.set_more_shapes returns []
# so the core list is used as-is and read as (m, n).
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
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


def _addmv_(bias, mat, vec):
    """benchmark torch_op: lambda bias, mat, vec: bias.addmv_(mat, vec)"""
    return bias.addmv_(mat, vec)


def native():
    if _resolve("torch.Tensor.addmv_") is None:
        return None
    return _addmv_


def grid():
    return [{"m": m, "n": n} for m, n in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    mat = torch.randn([m, n], dtype=dtype, device=device)
    vec = torch.randn([n], dtype=dtype, device=device)
    bias = torch.randn([m], dtype=dtype, device=device)
    return (bias, mat, vec), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [m], "dtype": dt, "note": "accumulated in place"},
                {"shape": [m, n], "dtype": dt},
                {"shape": [n], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"self": {"shape": [m], "dtype": dt}},
        "dims": {"M": m, "N": n},
    }
