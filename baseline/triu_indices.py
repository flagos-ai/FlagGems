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

"""NVIDIA native baseline for `triu_indices`.

Mirrors benchmark/test_triangular_indices.py (TriangularIndicesBenchmark, a
GenericBenchmark subclass):
    op_name  = "triu_indices"
    torch_op = torch.triu_indices
    dtypes   = [torch.int32, torch.int64]

The class overrides set_shapes with its own (row, col) list and returns [] from
set_more_shapes, so core_shapes.yaml is not consulted.

This is a factory operator: it takes no input tensor. At --level core offset is
0; the +-min(row, col) // 4 variants are COMPREHENSIVE-only. All arguments are
keywords, as the input_fn yields them.
"""

import torch

OP_NAME = "triu_indices"
DTYPES = [torch.int32, torch.int64]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.triu_indices is unavailable on this torch build"

# TriangularIndicesBenchmark.set_shapes -- shape_desc "row, col"
_SHAPES = [
    (64, 64),
    (256, 1024),
    (1024, 256),
    (1024, 1024),
]

_OFFSET = 0


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
    op = _resolve("torch.triu_indices")
    return op if callable(op) else None


def grid():
    return [{"row": row, "col": col} for row, col in _SHAPES]


def build_inputs(binding, dtype, device):
    return (), {
        "row": binding["row"],
        "col": binding["col"],
        "offset": _OFFSET,
        "dtype": dtype,
        "device": device,
    }


def key_shape(binding):
    return [binding["row"], binding["col"]]


def config(binding, dtype):
    row, col = binding["row"], binding["col"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [],
            "kwargs": {
                "row": {"scalar": row},
                "col": {"scalar": col},
                "offset": {"scalar": _OFFSET},
                "dtype": dt,
                "device": "cuda:0",
            },
        },
        "outputs": {"out": {"dtype": dt, "note": "[2, nnz] upper-triangular indices"}},
        "dims": {"row": row, "col": col, "offset": _OFFSET},
    }
