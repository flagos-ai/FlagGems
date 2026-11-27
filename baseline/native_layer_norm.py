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

"""NVIDIA native baseline for `native_layer_norm`.

Mirrors benchmark/test_layer_norm.py (NormBenchmark, a GenericBenchmark
subclass):
    op_name  = "native_layer_norm"
    torch_op = torch.ops.aten.native_layer_norm.default
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:NormBenchmark (benchmark/core_shapes.yaml),
               shape_desc "N, C, *"

normalized_shape is shape[1:], so weight and bias carry every dimension but
the leading one. eps is 1e-5.
"""

import torch

OP_NAME = "native_layer_norm"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.native_layer_norm.default is unavailable on this torch build"
)

# yaml:NormBenchmark -- shape_desc "N, C, *"
_SHAPES = [
    (4, 16, 64, 4),
    (16, 16, 8, 48),
    (16, 16, 8, 88),
    (16, 16, 128),
    (20, 6, 65536),
]

_EPS = 1e-5


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
    op = _resolve("torch.ops.aten.native_layer_norm.default")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    normalized_shape = shape[1:]
    weight = torch.randn(normalized_shape, dtype=dtype, device=device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=device)
    return (inp, normalized_shape, weight, bias, _EPS), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    normalized_shape = shape[1:]
    dt = str(dtype)
    stat_shape = [shape[0]] + [1] * len(normalized_shape)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                normalized_shape,
                {"shape": normalized_shape, "dtype": dt},
                {"shape": normalized_shape, "dtype": dt},
                {"scalar": _EPS},
            ],
            "kwargs": {},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": dt},
            "mean": {"shape": stat_shape, "dtype": dt},
            "rstd": {"shape": stat_shape, "dtype": dt},
        },
        "dims": {
            "shape": shape,
            "normalized_shape": normalized_shape,
            "eps": _EPS,
        },
    }
