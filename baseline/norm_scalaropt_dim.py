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

"""NVIDIA native baseline for `norm_scalaropt_dim`.

Mirrors benchmark/test_norm.py (base.GenericBenchmarkExcluse1D):
    op_name  = "norm_scalaropt_dim"
    torch_op = torch.norm
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:GenericBenchmarkExcluse1D (benchmark/core_shapes.yaml)

norm_scalaropt_dim_input_fn yields (inp, 2, -1, False) -- an L2 norm
reduced over the last dimension without keepdim.
"""

import torch

OP_NAME = "norm_scalaropt_dim"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.norm is unavailable on this torch build"

# yaml:GenericBenchmarkExcluse1D
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_P = 2
_DIM = -1
_KEEPDIM = False


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
    op = _resolve("torch.norm")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp, _P, _DIM, _KEEPDIM), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": _P},
                {"scalar": _DIM},
                {"scalar": _KEEPDIM},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape[:-1], "dtype": dt}},
        "dims": {"shape": shape, "p": _P, "dim": _DIM, "keepdim": _KEEPDIM},
    }
