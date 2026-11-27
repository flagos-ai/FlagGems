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

"""NVIDIA native baseline for `linalg_matrix_rank`.

Mirrors benchmark/test_linalg_matrix_rank.py (MatrixRankBenchmark, a
GenericBenchmark subclass):
    op_name  = "linalg_matrix_rank"
    torch_op = torch.linalg.matrix_rank
    dtypes   = [float32, float64] when the backend has FP64

The class overrides set_shapes with _select_shapes(CORE, COMPREHENSIVE), which
at --level core returns MATRIX_RANK_CORE_SHAPES only, so core_shapes.yaml is
not consulted and the long comprehensive list is not part of this grid.

The input is a plain randn matrix; the operator's tolerance defaults are used.
"""

import torch

OP_NAME = "linalg_matrix_rank"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.linalg.matrix_rank is unavailable on this torch build"
)

# test_linalg_matrix_rank.py::MATRIX_RANK_CORE_SHAPES -- shape_desc "*, M, N"
_SHAPES = [
    (1, 256),
    (256, 1),
    (2, 256),
    (256, 2),
    (8, 8),
    (16, 16),
    (17, 17),
    (32, 32),
    (33, 33),
    (64, 64),
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
    op = _resolve("torch.linalg.matrix_rank")
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    matrix = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (matrix,), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    dt = str(dtype)
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": shape[:-2], "dtype": "torch.int64"}},
        "dims": {"shape": shape, "M": shape[-2], "N": shape[-1]},
    }
