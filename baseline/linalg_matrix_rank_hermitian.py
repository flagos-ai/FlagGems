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

"""NVIDIA baseline for linalg_matrix_rank_hermitian (torch native).

Self-contained module; depends only on torch. ``native()`` returns
``torch.linalg.matrix_rank`` called with ``hermitian=True``; the ``svdvals`` +
threshold composition that the benchmark substitutes on Ascend (no NPU kernel
for ``aten::_linalg_eigh.eigenvalues``) is not needed here, since baselines are
collected on NVIDIA. ``MatrixRankHermitianBenchmark`` overrides ``set_shapes``,
so ``core_shapes.yaml`` is not consulted: the 8
``MATRIX_RANK_HERMITIAN_CORE_SHAPES`` entries are the default, and ``--level
comprehensive`` adds the 12 ``MATRIX_RANK_HERMITIAN_COMPREHENSIVE_SHAPES``
entries (20 total; the THead/Iluvatar filters do not apply on NVIDIA). Dtypes
(``MATRIX_RANK_DTYPES``: fp32 and fp64 on NVIDIA) and the ``A = randn(shape);
A = A + A.mT`` input construction are copied from
``benchmark/test_linalg_matrix_rank.py``. Every shape is a single or batched
square matrix (``hermitian=True`` requires it), and the result is an int64 rank
of the batch shape ``shape[:-2]``.
"""

import torch

OP_NAME = "linalg_matrix_rank_hermitian"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False


# benchmark/test_linalg_matrix_rank.py::MATRIX_RANK_HERMITIAN_CORE_SHAPES
# (+ MATRIX_RANK_HERMITIAN_COMPREHENSIVE_SHAPES at --level comprehensive)
_CASES = [
    (1, 1),
    (2, 2),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (33, 33),
    (64, 64),
    # Hermitian 65+ always uses one-sided tridiagonalization. 65/129/257 sample
    # the small/medium tile boundaries and the k >= 256 region.
    (65, 65),
    (128, 128),
    (129, 129),
    (256, 256),
    (257, 257),
    (512, 512),
    (1024, 1024),
    (32, 8, 8),
    (8, 16, 16),
    (4, 32, 32),
    (2, 64, 64),
    (2, 4, 16, 16),
]


def native():
    return torch.ops.aten.linalg_matrix_rank


def grid():
    return [{"shape": case} for case in _CASES]


def build_inputs(binding, dtype, device):
    # Mirrors benchmark/test_linalg_matrix_rank.py::matrix_rank_hermitian_input_fn.
    x = torch.randn(binding["shape"], dtype=dtype, device=device)
    x = x + x.mT
    return (x,), {"hermitian": True}


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    # matrix_rank(hermitian=True) reduces the last two dimensions to an int64
    # rank of shape (*batch).
    shape = list(binding["shape"])
    return {
        "inputs": {
            "x": {"shape": shape, "dtype": str(dtype)},
        },
        "outputs": {
            "out": {"shape": shape[:-2], "dtype": "torch.int64"},
        },
        "dims": {"hermitian": True},
    }
