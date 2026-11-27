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

"""NVIDIA native baseline for `new_zeros_out`.

Mirrors benchmark/test_new_zeros.py (NewZerosOutBenchmark, a GenericBenchmark
subclass using the two-phase case_fn/build_inputs_fn pair):
    op_name  = "new_zeros_out"
    torch_op = torch.ops.aten.new_zeros.out
    dtypes   = consts.FLOAT_DTYPES

The parent class overrides set_shapes with NEW_ZEROS_SHAPES, deliberately
bypassing core_shapes.yaml: this is a factory operator and the base fallback
shapes' allocation cost would dominate the measurement.

The source tensor only supplies dtype/device defaults, so the benchmark keeps
it at a single element; the out buffer is allocated in build_inputs so its
allocation stays outside the timed region and the measured work is the
in-place zero fill of `size`.
"""

import torch

OP_NAME = "new_zeros_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out",)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.new_zeros.out is unavailable on this torch build"
)

# test_new_zeros.py::NEW_ZEROS_SHAPES -- shape_desc "size"
_SHAPES = [
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
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
    op = _resolve("torch.ops.aten.new_zeros.out")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    size = list(binding["shape"])
    src = torch.empty(1, dtype=dtype, device=device)
    out = torch.empty(size, dtype=dtype, device=device)
    return (src, size), {"out": out}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    size = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [1], "dtype": dt, "note": "defaults source only"},
                size,
            ],
            "kwargs": {"out": {"shape": size, "dtype": dt}},
        },
        "outputs": {"out": {"shape": size, "dtype": dt}},
        "dims": {"size": size},
    }
