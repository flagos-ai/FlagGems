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

"""NVIDIA native baseline for `normal_tensor_float`.

Mirrors benchmark/test_normal.py (base.GenericBenchmark):
    op_name  = "normal_tensor_float"
    torch_op = torch.normal
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

This is the (Tensor mean, float std) overload: `loc` is a full tensor of 3.0
and `scale` the Python float 10.0, as normal_tensor_float_input_fn yields them.
"""

import torch

OP_NAME = "normal_tensor_float"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.normal is unavailable on this torch build"

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_LOC_FILL = 3.0
_SCALE = 10.0


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
    op = _resolve("torch.normal")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    loc = torch.full(
        tuple(binding["shape"]), fill_value=_LOC_FILL, dtype=dtype, device=device
    )
    return (loc, _SCALE), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": f"full({_LOC_FILL})"},
                {"scalar": _SCALE},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "mean": _LOC_FILL, "std": _SCALE},
    }
