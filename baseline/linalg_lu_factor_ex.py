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

"""NVIDIA baseline for linalg_lu_factor_ex (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.linalg.lu_factor_ex``; the manual pivot/info composition that the
benchmark substitutes on Ascend (no native kernel) is not needed here, since
baselines are collected on NVIDIA. ``LinalgLuFactorExBenchmark`` overrides
``set_shapes``, so ``core_shapes.yaml`` is not consulted: the 13
``LU_FACTOR_EX_SHAPES`` entries run with ``pivot`` both ``True`` and ``False``,
and ``check_errors`` both ``False`` and ``True`` only for ``pivot=True`` (39
cases; the ``pivot=False`` path is CUDA-only). Dtypes (``_TEST_DTYPES``: fp32
and fp64 on NVIDIA) and the plain ``torch.randn`` input construction are
copied from ``benchmark/test_linalg_lu_factor_ex.py``. For an
``(*batch, m, n)`` input the result is ``(LU, pivots, info)`` with shapes
``(*batch, m, n)``, ``(*batch, k)`` (int32) and ``(*batch)`` (int32), where
``k = min(m, n)``.
"""

import torch

OP_NAME = "linalg_lu_factor_ex"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False


# benchmark/test_linalg_lu_factor_ex.py::LU_FACTOR_EX_SHAPES
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
    return torch.ops.aten.linalg_lu_factor_ex


def grid():
    return [
        {"shape": shape, "pivot": pivot, "check_errors": check_errors}
        for shape in _CASES
        for pivot in (True, False)
        # check_errors only varies for pivot=True, mirroring the benchmark.
        for check_errors in ((False, True) if pivot else (False,))
    ]


def build_inputs(binding, dtype, device):
    x = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (x,), {"pivot": binding["pivot"], "check_errors": binding["check_errors"]}


def key_shape(binding):
    shape = "x".join(str(dim) for dim in binding["shape"])
    return f"{shape}_pivot{binding['pivot']}_check_errors{binding['check_errors']}"


def config(binding, dtype):
    # lu_factor_ex returns (LU, pivots, info) for an (*batch, m, n) input: LU
    # keeps the input shape, pivots is (*batch, k) with k = min(m, n), and
    # info is (*batch); pivots and info are int32.
    shape = list(binding["shape"])
    m, n = shape[-2], shape[-1]
    k = min(m, n)
    batch = shape[:-2]
    return {
        "inputs": {
            "x": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {
            "LU": {"shape": shape, "dtype": str(dtype)},
            "pivots": {"shape": batch + [k], "dtype": "torch.int32"},
            "info": {"shape": batch, "dtype": "torch.int32"},
        },
        "dims": {"pivot": binding["pivot"], "check_errors": binding["check_errors"]},
    }
