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

"""NVIDIA native baseline for `inner`.

Mirrors benchmark/test_inner.py (InnerBenchmark, a base.Benchmark subclass):
    op_name  = "inner"
    torch_op = torch.inner
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with INNER_SHAPES and declares shape_desc
"input shape, other shape", so core_shapes.yaml is not consulted. The pairs
cover 1-D dot products, 2-D x 2-D contractions over the last axis, and a
2-D x 1-D case.
"""

import torch

OP_NAME = "inner"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.inner is unavailable on this torch build"

# test_inner.py::INNER_SHAPES -- (input shape, other shape)
_SHAPES = [
    ((4096,), (4096,)),
    ((16384,), (16384,)),
    ((512, 1024), (1024, 1024)),
    ((1024, 4096), (2048, 4096)),
    ((2048, 4096), (2048, 4096)),
    ((256, 128), (128,)),
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
    op = _resolve("torch.inner")
    return op if callable(op) else None


def grid():
    return [
        {"input_shape": list(inp), "other_shape": list(other)} for inp, other in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["input_shape"]), dtype=dtype, device=device)
    other = torch.randn(tuple(binding["other_shape"]), dtype=dtype, device=device)
    return (inp, other), {}


def key_shape(binding):
    inp = "x".join(str(d) for d in binding["input_shape"])
    other = "x".join(str(d) for d in binding["other_shape"])
    return f"in{inp}_other{other}"


def config(binding, dtype):
    inp = list(binding["input_shape"])
    other = list(binding["other_shape"])
    dt = str(dtype)
    out_shape = inp[:-1] + other[:-1]
    return {
        "inputs": {
            "args": [
                {"shape": inp, "dtype": dt},
                {"shape": other, "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"input_shape": inp, "other_shape": other},
    }
