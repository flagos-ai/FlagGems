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

"""NVIDIA native baseline for `bilinear`.

Mirrors benchmark/test_bilinear.py (BilinearBenchmark, a base.Benchmark
subclass):
    op_name  = "bilinear"
    torch_op = torch.nn.functional.bilinear
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes to assign BILINEAR_SHAPES, so core_shapes.yaml
is not consulted. Each entry is (M, in1, in2, out) and becomes
input1[M, in1], input2[M, in2], weight[out, in1, in2], bias[out].
"""

import torch

OP_NAME = "bilinear"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nn.functional.bilinear is unavailable on this torch build"
)

# test_bilinear.py::BILINEAR_SHAPES -- (M, in1, in2, out)
_SHAPES = [
    (256, 64, 64, 128),
    (512, 128, 128, 256),
    (1024, 64, 48, 96),
    (128, 256, 256, 128),
    (2048, 32, 32, 64),
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
    op = _resolve("torch.nn.functional.bilinear")
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "in1": in1, "in2": in2, "out": out} for m, in1, in2, out in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    m, in1, in2, out = binding["m"], binding["in1"], binding["in2"], binding["out"]
    input1 = torch.randn((m, in1), dtype=dtype, device=device)
    input2 = torch.randn((m, in2), dtype=dtype, device=device)
    weight = torch.randn((out, in1, in2), dtype=dtype, device=device)
    bias = torch.randn((out,), dtype=dtype, device=device)
    return (input1, input2, weight, bias), {}


def key_shape(binding):
    return [binding["m"], binding["in1"], binding["in2"], binding["out"]]


def config(binding, dtype):
    m, in1, in2, out = binding["m"], binding["in1"], binding["in2"], binding["out"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [m, in1], "dtype": dt},
                {"shape": [m, in2], "dtype": dt},
                {"shape": [out, in1, in2], "dtype": dt},
                {"shape": [out], "dtype": dt},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [m, out], "dtype": dt}},
        "dims": {"M": m, "in1": in1, "in2": in2, "out": out},
    }
