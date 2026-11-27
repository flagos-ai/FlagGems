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

"""NVIDIA native baseline for `standard_gamma_grad`.

Mirrors benchmark/test_standard_gamma_grad.py (base.GenericBenchmark):
    op_name  = "standard_gamma_grad"
    torch_op = torch._standard_gamma_grad
    dtypes   = consts.FLOAT_DTYPES
    shapes   = yaml:Benchmark (benchmark/core_shapes.yaml)

Both arguments must be positive for a Gamma distribution, so the input_fn draws
them from rand(shape) * 4.0 + 0.5 rather than randn.
"""

import torch

OP_NAME = "standard_gamma_grad"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch._standard_gamma_grad is unavailable on this torch build"
)

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
    op = _resolve("torch._standard_gamma_grad")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    # Both alpha (self_grad) and output must be positive.
    self_grad = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.5
    output = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.5
    return (self_grad, output), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "> 0"},
                {"shape": shape, "dtype": dt, "note": "> 0"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape},
    }
