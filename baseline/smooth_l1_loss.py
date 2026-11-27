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

"""NVIDIA native baseline for `smooth_l1_loss`.

Mirrors benchmark/test_smooth_l1_loss.py (SmoothL1LossBenchmark, a
base.Benchmark subclass):
    op_name  = "smooth_l1_loss"
    torch_op = torch.ops.aten.smooth_l1_loss
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:smooth_l1_loss (benchmark/core_shapes.yaml),
               shape_desc "(B), M, N"

get_input_iter yields (inp, target, 1, 1.0): reduction 1 is mean and beta is
1.0.
"""

import torch

OP_NAME = "smooth_l1_loss"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten.smooth_l1_loss is unavailable on this torch build"
)

# yaml:smooth_l1_loss -- shape_desc "(B), M, N"
_SHAPES = [
    (262144,),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]

_REDUCTION = 1  # mean
_BETA = 1.0


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
    op = _resolve("torch.ops.aten.smooth_l1_loss")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, device=device, dtype=dtype)
    target = torch.randn(shape, device=device, dtype=dtype)
    return (inp, target, _REDUCTION, _BETA), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"shape": shape, "dtype": dt},
                {"scalar": _REDUCTION},
                {"scalar": _BETA},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [], "dtype": dt}},
        "dims": {"shape": shape, "reduction": _REDUCTION, "beta": _BETA},
    }
