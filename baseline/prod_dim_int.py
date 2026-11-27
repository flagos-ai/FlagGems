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

"""NVIDIA native baseline for `prod_dim_int`.

Mirrors benchmark/test_prod_dim_int.py (ProdDimIntBenchmark, a
base.UnaryReductionBenchmark subclass):
    op_name  = "prod_dim_int"
    torch_op = torch.prod
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:UnaryReductionBenchmark (benchmark/core_shapes.yaml)

The subclass replaces the base reduction iterator with a rank-dependent sweep:
every shape yields the no-dim whole-tensor reduction, a rank >= 2 shape adds
dim 0 and dim -1, and a rank >= 3 shape adds dim 1. The grid below expands that
per shape.
"""

import torch

OP_NAME = "prod_dim_int"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.prod is unavailable on this torch build"

# yaml:UnaryReductionBenchmark
_SHAPES = [
    (1048576,),
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
    op = _resolve("torch.prod")
    return op if callable(op) else None


def grid():
    bindings = []
    for shape in _SHAPES:
        bindings.append({"shape": list(shape), "dim": None})
        if len(shape) >= 2:
            bindings.append({"shape": list(shape), "dim": 0})
            bindings.append({"shape": list(shape), "dim": -1})
        if len(shape) >= 3:
            bindings.append({"shape": list(shape), "dim": 1})
    return bindings


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    if binding["dim"] is None:
        return (inp,), {}
    return (inp, binding["dim"]), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    dim = "all" if binding["dim"] is None else str(binding["dim"])
    return f"{shape}_dim{dim}"


def config(binding, dtype):
    shape = list(binding["shape"])
    dim = binding["dim"]
    dt = str(dtype)
    args = [{"shape": shape, "dtype": dt}]
    if dim is None:
        out_shape = []
    else:
        args.append({"scalar": dim})
        axis = dim if dim >= 0 else len(shape) + dim
        out_shape = [d for i, d in enumerate(shape) if i != axis]
    return {
        "inputs": {"args": args, "kwargs": {}},
        "outputs": {"out": {"shape": out_shape, "dtype": dt}},
        "dims": {"shape": shape, "dim": dim},
    }
