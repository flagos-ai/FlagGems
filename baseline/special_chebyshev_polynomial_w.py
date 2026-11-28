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

"""NVIDIA native baseline for `special_chebyshev_polynomial_w`.

Mirrors benchmark/test_special_chebyshev_polynomial_w.py (ChebyshevPolynomialWBenchmark, a
base.Benchmark subclass):
    op_name  = "special_chebyshev_polynomial_w"
    torch_op = torch.special.chebyshev_polynomial_w
    dtypes   = [torch.float32]

The base class overrides set_shapes with its own modest special.* subset, so
core_shapes.yaml is not consulted; set_more_shapes only applies at --level
comprehensive.

The degree n is the fixed int64 scalar tensor 3 (the benchmark keeps it
constant for stability) and x is mapped into the polynomial's [-1, 1] domain
with `x * 2 - 1`.

float32 only: the chebyshev_polynomial CUDA kernels support no other dtype.
"""

import torch

OP_NAME = "special_chebyshev_polynomial_w"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.special.chebyshev_polynomial_w is unavailable on this torch build"
)

# ChebyshevPolynomial*Benchmark.set_shapes -- the special.* subset
_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 1280),
    (2, 19, 7),
]

_N = 3

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_INT_DTYPES = (torch.int16, torch.int32)
_BOOL_DTYPES = (torch.bool,)
_COMPLEX_DTYPES = (torch.complex64,)


def _generate_tensor_input(shape, dtype, device):
    """benchmark/utils.py::generate_tensor_input, verbatim semantics."""
    if dtype in _FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in _INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)
    if dtype in _BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(device)
    if dtype in _COMPLEX_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    raise TypeError(f"unsupported dtype: {dtype}")


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
    op = _resolve("torch.special.chebyshev_polynomial_w")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    x = _generate_tensor_input(shape, dtype, device)
    x = x * 2 - 1
    n = torch.tensor(_N, dtype=torch.int64, device=device)
    return (x, n), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": shape, "dtype": dt, "note": "x in [-1, 1]"},
                {"shape": [], "dtype": "torch.int64", "note": "degree n = 3"},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape, "dtype": dt}},
        "dims": {"shape": shape, "n": _N},
    }
