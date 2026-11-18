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

"""NVIDIA baseline for pairwise_distance (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.pairwise_distance``; the composed fallback that FlagGems uses on Ascend
for ``p = +/-inf`` and non-standard ``p`` is not needed here, since baselines
are collected on NVIDIA and torch handles every benchmarked ``p`` on CUDA.
Shapes, dtypes, the ``p`` list (``-inf, inf, 0, 1, 2, 6.6``), ``eps=1e-6`` and
the input construction are copied from ``benchmark/test_pairwise_distance.py``.
"""

import torch

OP_NAME = "pairwise_distance"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False


# benchmark/core_shapes.yaml::GenericBenchmark2DOnly, plus the four extra shapes
# that PairwiseDistanceBenchmark.set_more_shapes adds at --level comprehensive
_CASES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
    (1, 65536),
    (8, 65536),
    (64, 65536),
    (1, 10000000),
]

# benchmark/test_pairwise_distance.py::_input_fn p values.
# Stored as strings: the collector embeds the binding dict in a generated
# profile script via repr(), and repr(float("-inf")) is not valid Python.
_P_LIST = ("-inf", "inf", "0.0", "1.0", "2.0", "6.6")


def native():
    return torch.pairwise_distance


def grid():
    return [{"shape": shape, "p": p} for shape in _CASES for p in _P_LIST]


def build_inputs(binding, dtype, device):
    x1 = torch.randn(binding["shape"], dtype=dtype, device=device)
    x2 = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (x1, x2), {"p": float(binding["p"])}


def key_shape(binding):
    shape = "x".join(str(dim) for dim in binding["shape"])
    return f"{shape}_p{binding['p']}"


def config(binding, dtype):
    # pairwise_distance reduces the last dimension; both operands share the case
    # shape and dtype.
    shape = list(binding["shape"])
    return {
        "inputs": {
            "x1": {"shape": shape, "dtype": str(dtype)},
            "x2": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {"out": {"shape": shape[:-1], "dtype": str(dtype)}},
        "dims": {"p": binding["p"], "eps": 1e-6},
    }
