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

"""NVIDIA baseline for linalg_lu (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.linalg.lu``; the manual pivot/unpack ``ops_lu`` composition that the
benchmark substitutes on Ascend (no native kernel) is not needed here, since
baselines are collected on NVIDIA. ``LinalgLuBenchmark`` overrides
``set_shapes``, so ``core_shapes.yaml`` is not consulted: the 13 ``LU_SHAPES``
entries are each run with ``pivot`` both ``True`` and ``False`` (26 cases; the
``pivot=False`` path is CUDA-only). Dtypes (``_TEST_DTYPES``: fp32 and fp64 on
NVIDIA) and the plain ``torch.randn`` input construction are copied from
``benchmark/test_linalg_lu.py``. For an ``(*batch, m, n)`` input the result is
``(P, L, U)`` with shapes ``(*batch, m, m)``, ``(*batch, m, k)`` and
``(*batch, k, n)``, where ``k = min(m, n)``.
"""

import torch

OP_NAME = "linalg_lu"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False


# benchmark/test_linalg_lu.py::LU_SHAPES
_CASES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 512),
    (32, 16),
    (16, 32),
    (128, 64),
    (64, 128),
    (4, 32, 32),
    (128, 16, 16),
    (1024, 512, 512),
]


def native():
    return torch.ops.aten.linalg_lu


def grid():
    return [
        {"shape": case, "pivot": pivot} for case in _CASES for pivot in (True, False)
    ]


def build_inputs(binding, dtype, device):
    x = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (x,), {"pivot": binding["pivot"]}


def key_shape(binding):
    shape = "x".join(str(dim) for dim in binding["shape"])
    return f"{shape}_pivot{binding['pivot']}"


def config(binding, dtype):
    # LU returns (P, L, U) for an (*batch, m, n) input: P is (*batch, m, m),
    # L is (*batch, m, k), U is (*batch, k, n), with k = min(m, n).
    shape = list(binding["shape"])
    m, n = shape[-2], shape[-1]
    k = min(m, n)
    batch = shape[:-2]
    return {
        "inputs": {
            "x": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {
            "P": {"shape": batch + [m, m], "dtype": str(dtype)},
            "L": {"shape": batch + [m, k], "dtype": str(dtype)},
            "U": {"shape": batch + [k, n], "dtype": str(dtype)},
        },
        "dims": {"pivot": binding["pivot"]},
    }
