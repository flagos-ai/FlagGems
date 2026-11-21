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

"""NVIDIA baseline for igammac (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.igammac`` (the regularized upper incomplete gamma Q(a, x)); the
composed power-series fallback used by FlagGems on non-NVIDIA vendors is not
needed here, since baselines are collected on NVIDIA. Shapes, dtypes and the
domain-valid (a, x) in [0.1, 10.1] input construction are copied from
``benchmark/test_igammac.py``. Following the benchmark's
``MAX_FLOAT64_ELEMENTS``, fp64 is collected only for shapes of at most 2**24
elements.
"""

import math

import torch

OP_NAME = "igammac"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False

# benchmark/test_igammac.py::IgammacBenchmark.MAX_FLOAT64_ELEMENTS
_MAX_FLOAT64_ELEMENTS = 2**24


# benchmark/core_shapes.yaml::Benchmark
_CASES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


def native():
    # torch.igammac is the NVIDIA reference for the regularized upper
    # incomplete gamma Q(a, x).
    return torch.igammac


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    # Mirror benchmark/test_igammac.py::_igammac_input: Q(a, x) is only defined
    # for a > 0 and x >= 0, so both operands use torch.rand (uniform in
    # [0.1, 10.1]) instead of randn, which can produce negative values.
    x = torch.rand(binding["shape"], dtype=dtype, device=device) * 10 + 0.1
    a = torch.rand(binding["shape"], dtype=dtype, device=device) * 10 + 0.1
    return (a, x), {}


def unsupported_reason(binding, dtype):
    # Match benchmark/test_igammac.py: fp64 is collected only for shapes up to
    # MAX_FLOAT64_ELEMENTS (torch's fp64 igammac is very slow on large tensors).
    if dtype == torch.float64 and math.prod(binding["shape"]) > _MAX_FLOAT64_ELEMENTS:
        return "fp64 baseline is collected only for shapes <= 2**24 elements"
    return None


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    # Both operands and the result share the case shape and dtype.
    shape = list(binding["shape"])
    return {
        "inputs": {
            "a": {"shape": shape, "dtype": str(dtype)},
            "x": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {"out": {"shape": shape, "dtype": str(dtype)}},
        "dims": {},
    }
