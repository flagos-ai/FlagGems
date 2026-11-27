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

"""NVIDIA native baseline for `reshape_as`.

Mirrors benchmark/test_reshape_as.py (base.GenericBenchmarkExcluse1D):
    op_name  = "reshape_as"
    torch_op = torch.Tensor.reshape_as
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:GenericBenchmarkExcluse1D (benchmark/core_shapes.yaml)

The input_fn deliberately makes every row hit the materializing copy: the
source is transposed, and `other` is a flat tensor of the same element count.
A 1-D source would be stride-expressible as any same-numel shape and would only
measure dispatch, which is why the benchmark excludes rank 1 and transposes.
"""

import torch

OP_NAME = "reshape_as"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.Tensor.reshape_as is unavailable on this torch build"

# yaml:GenericBenchmarkExcluse1D
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
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
    op = _resolve("torch.Tensor.reshape_as")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    numel = 1
    for s in shape:
        numel *= s
    other = torch.empty((numel,), dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device).transpose(0, -1)
    return (inp, other), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    numel = 1
    for d in shape:
        numel *= d
    transposed = list(shape)
    transposed[0], transposed[-1] = transposed[-1], transposed[0]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": transposed,
                    "dtype": dt,
                    "note": "transposed view of the core shape",
                },
                {"shape": [numel], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [numel], "dtype": dt}},
        "dims": {"shape": shape, "numel": numel},
    }
