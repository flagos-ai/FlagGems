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

"""NVIDIA native baseline for `is_all_true`.

Mirrors benchmark/test_is_all_true.py (IsAllTrueBenchmark, a base.Benchmark
subclass):
    op_name  = "is_all_true"
    torch_op = torch._is_all_true
    dtypes   = consts.BOOL_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

_is_all_true accepts bool tensors only and reduces over every element, so the
input is randint(0, 2) in torch.bool regardless of the dtype axis.
"""

import torch

OP_NAME = "is_all_true"
DTYPES = [torch.bool]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch._is_all_true is unavailable on this torch build"

# yaml:Benchmark -- the shared default shape list
_SHAPES = [
    (1073741824,),
    (64, 64),
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
    op = _resolve("torch._is_all_true")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    return (inp,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": "torch.bool"}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [], "dtype": "torch.bool"}},
        "dims": {"shape": shape},
    }
