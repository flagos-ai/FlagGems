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

"""NVIDIA baseline for linalg_matrix_exp (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.linalg.matrix_exp``; the ``_small_ops_matrix_exp`` scaling-and-squaring
composition that the benchmark substitutes on NPU (no native kernel) is not
needed here, since baselines are collected on NVIDIA. ``MatrixExpBenchmark``
overrides ``set_shapes``, so ``core_shapes.yaml`` is not consulted: the 14
``MATRIX_EXP_SHAPES`` entries are used, mixing square systems (16 to 256) with
batched small systems. Dtypes (``MATRIX_EXP_DTYPES``: fp32 plus fp64 when the
device supports it, i.e. both on NVIDIA) and the plain ``torch.randn`` input
construction are copied from ``benchmark/test_linalg_matrix_exp.py``; the
result has the same shape as the input.
"""

import torch

OP_NAME = "linalg_matrix_exp"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False


# benchmark/test_linalg_matrix_exp.py::MATRIX_EXP_SHAPES
_CASES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (4096, 4, 4),
    (1024, 8, 8),
    (1024, 16, 16),
    (128, 16, 16),
    (4, 32, 32),
    (512, 32, 32),
    (256, 64, 64),
    (32, 128, 128),
    (8, 256, 256),
]


def native():
    return torch.ops.aten.linalg_matrix_exp


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    x = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (x,), {}


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    # matrix_exp only accepts square matrices; the result is (*batch, n, n).
    shape = list(binding["shape"])
    return {
        "inputs": {
            "x": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": str(dtype)},
        },
        "dims": {},
    }
