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

"""NVIDIA native baseline for `special_chebyshev_polynomial_u`.

Mirrors benchmark/test_special_chebyshev_polynomial_u.py (base.GenericBenchmarkExcluse1D):
    op_name  = "special_chebyshev_polynomial_u"
    torch_op = torch.special.chebyshev_polynomial_u
    dtypes   = [torch.float32]
    shapes   = yaml:GenericBenchmarkExcluse1D (benchmark/core_shapes.yaml)

The degree n is the fixed Python int 3. The input is built in float32 directly by the
input_fn, and it is not remapped into [-1, 1].

float32 only: this special.* CUDA kernel supports no other dtype.
"""

import torch

OP_NAME = "special_chebyshev_polynomial_u"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.special.chebyshev_polynomial_u is unavailable on this torch build"
)

# yaml:GenericBenchmarkExcluse1D
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_N = 3


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
    op = _resolve("torch.special.chebyshev_polynomial_u")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    # The input_fn pins float32 regardless of the dtype axis.
    x = torch.randn(tuple(binding["shape"]), dtype=torch.float32, device=device)
    return (x, _N), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt},
                {"scalar": _N},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "n": _N},
    }
