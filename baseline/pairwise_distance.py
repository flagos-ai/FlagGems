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

"""NVIDIA native baseline for `pairwise_distance`.

Mirrors benchmark/test_pairwise_distance.py (PairwiseDistanceBenchmark, a
GenericBenchmark2DOnly subclass):
    op_name  = "pairwise_distance"
    torch_op = torch.pairwise_distance
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:GenericBenchmark2DOnly (benchmark/core_shapes.yaml)

The `composed_pairwise_distance` fallback in that file is selected only on
Ascend; on NVIDIA the benchmark calls torch.pairwise_distance.

The class's set_more_shapes adds the small-N-large-D shapes, but
set_more_shapes only runs at --level comprehensive, so this grid is the yaml
list alone.

x1 and x2 share the same (M, N) shape, so one p-distance is computed per row.
The input_fn sweeps p over (-inf, inf, 0.0, 1.0, 2.0, 6.6), giving six
collection points per shape; p is passed as a keyword.
"""

import torch

OP_NAME = "pairwise_distance"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.pairwise_distance is unavailable on this torch build"

# yaml:GenericBenchmark2DOnly
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_P_VALUES = (float("-inf"), float("inf"), 0.0, 1.0, 2.0, 6.6)

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_INT_DTYPES = (torch.int16, torch.int32)
_BOOL_DTYPES = (torch.bool,)
_COMPLEX_DTYPES = (torch.complex64,)


def _generate_tensor_input(shape, dtype, device):
    """benchmark/utils.py::generate_tensor_input, verbatim semantics."""
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in _INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    if dtype in _BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(device)
    if dtype in _COMPLEX_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    raise TypeError(f"unsupported dtype: {dtype}")


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
    op = _resolve("torch.pairwise_distance")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s), "p": p} for s in _SHAPES for p in _P_VALUES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp1 = _generate_tensor_input(shape, dtype, device)
    inp2 = _generate_tensor_input(shape, dtype, device)
    return (inp1, inp2), {"p": binding["p"]}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_p{binding['p']}"


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt},
            ],
            "kwargs": {"p": {"scalar": binding["p"]}},
        },
        "outputs": {"out": {"shape": [shape[0]], "dtype": dt}},
        "dims": {"M": shape[0], "D": shape[1], "p": binding["p"]},
    }
