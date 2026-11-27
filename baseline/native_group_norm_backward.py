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

"""NVIDIA native baseline for `native_group_norm_backward`.

Mirrors benchmark/test_group_norm_backward.py (GroupNormBackwardBenchmark, a
GenericBenchmark subclass):
    op_name  = "native_group_norm_backward"
    torch_op = torch.ops.aten.native_group_norm_backward
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with GN_BWD_SHAPES and shape_desc "N, C, H, W",
so core_shapes.yaml is not consulted.

num_groups equals C in this benchmark (one group per channel), HxW is the
product of the two spatial dims and output_mask is [True, True, True].
N/C/HxW/group/output_mask are passed as keywords, as the input_fn yields them.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "native_group_norm_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.native_group_norm_backward is unavailable on this build"
)

# test_group_norm_backward.py::GN_BWD_SHAPES -- shape_desc "N, C, H, W"
_SHAPES = [
    (16, 3, 16, 16),
    (1, 64, 32, 32),
    (2, 64, 128, 128),
]

_OUTPUT_MASK = [True, True, True]


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
    op = _resolve("torch.ops.aten.native_group_norm_backward")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    n, c = shape[0], shape[1]
    hxw = shape[2] * shape[3]
    num_groups = c
    grad_out = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.randn([n, num_groups], dtype=dtype, device=device)
    rstd = torch.randn([n, num_groups], dtype=dtype, device=device).abs() + 0.1
    weight = torch.randn(c, dtype=dtype, device=device)
    return (grad_out, inp, mean, rstd, weight), {
        "N": n,
        "C": c,
        "HxW": hxw,
        "group": num_groups,
        "output_mask": list(_OUTPUT_MASK),
    }


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    n, c = shape[0], shape[1]
    hxw = shape[2] * shape[3]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt},
                {"shape": [n, c], "dtype": dt},
                {"shape": [n, c], "dtype": dt},
                {"shape": [c], "dtype": dt},
            ],
            "kwargs": {
                "N": {"scalar": n},
                "C": {"scalar": c},
                "HxW": {"scalar": hxw},
                "group": {"scalar": c},
                "output_mask": list(_OUTPUT_MASK),
            },
        },
        "outputs": {
            "grad_input": {"shape": shape, "dtype": dt},
            "grad_weight": {"shape": [c], "dtype": dt},
            "grad_bias": {"shape": [c], "dtype": dt},
        },
        "dims": {"N": n, "C": c, "HxW": hxw, "group": c, "shape": shape},
    }
