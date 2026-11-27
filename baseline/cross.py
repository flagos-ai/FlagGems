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

"""NVIDIA native baseline for `cross`.

Mirrors benchmark/test_cross.py (CrossBenchmark, a base.Benchmark subclass):
    op_name  = "cross"
    torch_op = torch.cross
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with CROSS_COMMON_CASES, so core_shapes.yaml is
not consulted. Each case is (input_shape, other_shape, dim) and mirrors the
layouts covered by tests/test_cross.py at the smallest size that clears fixed
kernel-launch overhead: broadcasting pairs, a non-trailing dim=1 case and
trailing dim=-1 cases. dim is passed as a keyword.
"""

import torch

OP_NAME = "cross"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.cross is unavailable on this torch build"

# test_cross.py::CROSS_COMMON_CASES -- (input_shape, other_shape, dim)
_SHAPES = [
    ((262144, 3, 4), (1, 3, 4), 1),
    ((1, 3), (131072, 3), -1),
    ((65536, 4, 3), (65536, 4, 3), -1),
    ((131072, 3), (131072, 3), -1),
    ((262144, 3), (262144, 3), -1),
    ((524288, 3), (524288, 3), -1),
    ((1048576, 3), (1048576, 3), -1),
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
    op = _resolve("torch.cross")
    return op if callable(op) else None


def grid():
    return [
        {"input_shape": list(inp), "other_shape": list(other), "dim": dim}
        for inp, other, dim in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["input_shape"]), dtype=dtype, device=device)
    other = torch.randn(tuple(binding["other_shape"]), dtype=dtype, device=device)
    return (inp, other), {"dim": binding["dim"]}


def key_shape(binding):
    inp = "x".join(str(d) for d in binding["input_shape"])
    other = "x".join(str(d) for d in binding["other_shape"])
    return f"in{inp}_other{other}_dim{binding['dim']}"


def config(binding, dtype):
    inp = list(binding["input_shape"])
    other = list(binding["other_shape"])
    dt = str(dtype)
    out_shape = list(torch.broadcast_shapes(tuple(inp), tuple(other)))
    return {
        "inputs": {
            "args": [
                {"shape": inp, "dtype": dt},
                {"shape": other, "dtype": dt},
            ],
            "kwargs": {"dim": {"scalar": binding["dim"]}},
        },
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {
            "input_shape": inp,
            "other_shape": other,
            "dim": binding["dim"],
        },
    }
