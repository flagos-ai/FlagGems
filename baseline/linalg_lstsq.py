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

"""NVIDIA baseline for linalg_lstsq (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.linalg.lstsq`` called with ``driver="gels"``; the ``qr`` /
``solve_triangular`` compositions that the benchmark substitutes on Ascend and
for the wide half on cuSOLVER-less builds are not needed here, since baselines
are collected on NVIDIA with a full cuSOLVER. ``LstsqBenchmark`` overrides
``set_shapes``, so ``core_shapes.yaml`` is not consulted: all 23
``_SHAPES_FULL`` entries are used (the ``_SHAPES_DEVICE`` reduction applies
only to the fully composed reference). Dtypes (``LINALG_LSTSQ_DTYPE``: fp32 and
fp64 on NVIDIA; ``gels`` supports no other type) and the ``A =
randn(*shape)`` / ``b = randn(shape[:-1])`` input construction are copied from
``benchmark/test_linalg_lstsq.py``. The result is ``(solution, residuals,
rank, singular_values)``: solution is ``(*batch, n)``, residuals is
``(*batch, 1)`` when ``m > n`` and empty otherwise, and
``rank``/``singular_values`` are empty for ``gels``.
"""

import torch

OP_NAME = "linalg_lstsq"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False


# benchmark/test_linalg_lstsq.py::_SHAPES_FULL
_CASES = [
    # tall (M >= N)
    (256, 32),
    (1024, 16),
    (4096, 8),
    (8, 4096, 8),
    (64, 2048, 16),
    (16, 8192, 16),
    # wide (M < N), within the tile budget (single-tile kernel)
    (8, 16, 512),
    (8, 64, 256),
    (16, 16, 1024),
    (64, 32, 256),
    # wide beyond the budget (blocked TSQR path)
    (64, 1024),
    (128, 512),
    (8, 64, 1024),
    (16, 32, 2048),
    # square / near-square -> compact-WY blocked QR
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (8, 256, 256),
    (8, 512, 512),
    (4096, 512),
    # underdetermined with large m -> the A^T QR also routes to WY
    (512, 1024),
    (1024, 2048),
]


def native():
    return torch.ops.aten.linalg_lstsq


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    A = torch.randn(binding["shape"], dtype=dtype, device=device)
    b = torch.randn(binding["shape"][:-1], dtype=dtype, device=device)
    return (A, b), {"driver": "gels"}


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    # lstsq (driver="gels") returns (solution, residuals, rank, singular_values):
    # solution is (*batch, n); residuals is (*batch, 1) when m > n and empty
    # otherwise; rank and singular_values are empty for gels.
    shape = list(binding["shape"])
    m, n = shape[-2], shape[-1]
    batch = shape[:-2]
    return {
        "inputs": {
            "x": {"shape": shape, "dtype": str(dtype)},
            "b": {"shape": batch + [m], "dtype": str(dtype)},
        },
        "outputs": {
            "solution": {"shape": batch + [n], "dtype": str(dtype)},
            "residuals": {
                "shape": batch + [1] if m > n else [0],
                "dtype": str(dtype),
            },
            "rank": {"shape": [0], "dtype": "torch.int64"},
            "singular_values": {"shape": [0], "dtype": str(dtype)},
        },
        "dims": {"driver": "gels"},
    }
